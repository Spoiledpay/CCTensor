"""Integracao do CCT num modelo PyTorch existente.

O README mostrava `CCTLinear`, `cct_matmul` e `from_tensor`, mas nao o padrao
de substituicao num modelo real. E' este arquivo:

    # 1) treine/obtenha um modelo normal
    model = TinyGPT(...)

    # 2) troque as projecoes por versoes que consomem estado comprimido
    cctify_model(model, codec="l3_shuffle", min_features=64)

    # 3) depois de qualquer optimizer.step():
    refresh_all(model)

`cctify_model` nao toca em embeddings/head por padrao: elas sao tabelas de
lookup (nao GEMM) ou compartilham peso (weight tying). Escolha com `skip`.

O modo de inferencia (`model.eval()`) usa `cct_matmul`, decodificando tile a
tile. Em `model.train()` o `CCTLinear` usa o peso denso, porque o AdamW
precisa do gradiente e do estado do otimizador (ver `cct/training/__init__.py`).
"""

import torch

from .training import CCTLinear

DEFAULT_SKIP = ("head", "tok_emb", "pos_emb", "ln", "ln_")


def _should_skip(name, module, min_features, skip):
    if module.in_features < min_features or module.out_features < min_features:
        return True
    return any(token in name for token in skip)


def _resolve_parent(model, dotted):
    parts = dotted.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def cctify_model(model, codec="l3_shuffle", tile_rows=64, min_features=64,
                 skip=DEFAULT_SKIP, refresh=True):
    """Substitui `nn.Linear` por `CCTLinear` in-place.

    -> dict com `replaced` (nomes), `raw_bytes`, `compressed_bytes`, `CR`.
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not isinstance(module, CCTLinear):
            if name and not _should_skip(name, module, min_features, skip):
                targets.append((name, module))

    replaced = []
    for name, module in targets:
        parent, attr = _resolve_parent(model, name)
        new = CCTLinear.from_dense(module, codec=codec, tile_rows=tile_rows)
        setattr(parent, attr, new)
        replaced.append(name)

    if refresh:
        refresh_all(model)
    return {"replaced": replaced, **cct_stats(model)}


def refresh_all(model):
    """Recomprime todos os `CCTLinear` (chame depois de `optimizer.step()`)."""
    count = 0
    for module in model.modules():
        if isinstance(module, CCTLinear):
            module.refresh()
            count += 1
    return count


def cct_stats(model) -> dict:
    layers = [m for m in model.modules() if isinstance(m, CCTLinear)]
    raw = sum(m.raw_bytes for m in layers)
    comp = sum(m.compressed_bytes for m in layers)
    return {
        "layers": len(layers),
        "raw_bytes": raw,
        "compressed_bytes": comp,
        "CR": raw / max(1, comp),
        "encode_ms": sum(m.encode_ms for m in layers),
    }


def uncctify_model(model):
    """Volta a `nn.Linear` denso (util para comparar no mesmo processo)."""
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, CCTLinear)]
    replaced = []
    for name, module in targets:
        parent, attr = _resolve_parent(model, name)
        dense = torch.nn.Linear(module.in_features, module.out_features,
                                bias=module.bias is not None)
        with torch.no_grad():
            dense.weight.copy_(module.weight.detach())
            if module.bias is not None:
                dense.bias.copy_(module.bias.detach())
        setattr(parent, attr, dense)
        replaced.append(name)
    return replaced
