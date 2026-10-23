"""Integracao do CCT num modelo PyTorch (o exemplo de adocao do README).

Cobre:
    - `CCTLinear.from_dense` == `F.linear` com pesos BF16;
    - entrada 3-D (B, T, in) — usada na geracao token a token;
    - `cctify_model` substitui apenas as projecoes certas;
    - `uncctify_model` volta a nn.Linear;
    - `refresh_all` recomprime depois do optimizer.step().

Rodar:
    python cct/tests/test_integrate.py
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cct.integrate import (cct_stats, cctify_model, refresh_all,  # noqa: E402
                           uncctify_model)
from cct.training import CCTLinear                                # noqa: E402


def _bf16_weights(linear):
    return linear.weight.detach().to(torch.bfloat16).float()


def test_from_dense_matches_linear():
    torch.manual_seed(0)
    dense = nn.Linear(64, 128)
    cct = CCTLinear.from_dense(dense, codec="l3_shuffle", tile_rows=32)
    cct.eval()
    x = torch.randn(4, 64)
    with torch.no_grad():
        y = cct(x)
        ref = F.linear(x, _bf16_weights(dense), dense.bias)
    rel = float((y - ref).abs().max() / (ref.abs().max() + 1e-9))
    assert rel < 1e-2, rel
    assert cct.compression_ratio > 1.0


def test_three_dimensional_input():
    torch.manual_seed(1)
    cct = CCTLinear.from_dense(nn.Linear(64, 32), tile_rows=16)
    cct.eval()
    x = torch.randn(2, 5, 64)
    with torch.no_grad():
        y = cct(x)
        ref = F.linear(x, _bf16_weights_from(cct), cct.bias)
    assert y.shape == (2, 5, 32)
    assert torch.isfinite(y).all()
    rel = float((y - ref).abs().max() / (ref.abs().max() + 1e-9))
    assert rel < 1e-2, rel


def _bf16_weights_from(cct):
    return cct.weight.detach().to(torch.bfloat16).float()


def test_cctify_and_uncctify():
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
    model.eval()
    x = torch.randn(8, 64)
    with torch.no_grad():
        y_dense = model(x)

    info = cctify_model(model, codec="l3_shuffle", tile_rows=32, min_features=16)
    assert info["layers"] == 2, info
    assert info["CR"] > 1.0, info
    assert info["compressed_bytes"] < info["raw_bytes"]

    model.eval()
    with torch.no_grad():
        y_cct = model(x)
    rel = float((y_cct - y_dense).abs().max() / (y_dense.abs().max() + 1e-9))
    assert rel < 5e-2, rel

    n = refresh_all(model)
    assert n == 2

    uncctify_model(model)
    assert not any(isinstance(m, CCTLinear) for m in model.modules())
    with torch.no_grad():
        y_back = model(x)
    # uncctify recupera os pesos fp32 originais -> igual ao denso inicial
    rel = float((y_back - y_dense).abs().max() / (y_dense.abs().max() + 1e-9))
    assert rel < 1e-6, rel


def test_cctify_skips_embeddings_and_head():
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(64, 128), nn.Linear(128, 64))
    # nomes que o filtro padrao ignora
    model = nn.ModuleDict({
        "head": nn.Linear(64, 259),
        "fc": nn.Linear(64, 128),
    })
    info = cctify_model(model, min_features=16)
    assert info["replaced"] == ["fc"], info["replaced"]
    assert isinstance(model["head"], nn.Linear)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = []
    for fn in tests:
        try:
            fn()
            print(f"  [OK] {fn.__name__}")
        except AssertionError as exc:
            fails.append(fn.__name__)
            print(f"  [XX] {fn.__name__}: {exc}")
    print(f"\n{len(tests) - len(fails)} OK / {len(fails)} falhas")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
