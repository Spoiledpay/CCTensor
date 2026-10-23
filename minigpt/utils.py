"""Utilidades compartilhadas: checkpoints, geracao e avaliacao."""

import os

import torch

from .model import TinyGPT
from .tokenizer import ByteTokenizer, CharTokenizer, ItemChTokenizer


def save_checkpoint(path, model, tokenizer, config, extra=None):
    """Salva modelo + tokenizer + config. `extra` guarda estado extra de treino
    (epoca, optimizer, scheduler) para permitir `--resume`."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "tokenizer": tokenizer.state_dict(),
        "config": config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    tsd = ckpt["tokenizer"]
    if tsd.get("byte_level"):
        tokenizer = ByteTokenizer.from_state_dict(tsd)
    elif "items" in tsd:
        tokenizer = ItemChTokenizer.from_state_dict(tsd)
    else:
        tokenizer = CharTokenizer.from_state_dict(tsd)
    cfg = ckpt["config"]
    model = TinyGPT(
        tokenizer.vocab_size,
        block_size=cfg["block_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        dropout=cfg.get("dropout", 0.0),
        tie_embeddings=cfg.get("tie_embeddings", False),
        pos_encoding=cfg.get("pos_encoding", "learned"),
        copy_head=cfg.get("copy_head", False),
        normalize_embeddings=cfg.get("normalize_embeddings", False),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer, cfg


@torch.no_grad()
def generate_continuation(
    model, tokenizer, prompt, max_new_tokens=40, greedy=True, temperature=0.8, top_k=5
):
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)
    max_new = max(1, min(max_new_tokens, model.block_size - idx.size(1)))
    out = model.generate(
        idx,
        max_new_tokens=max_new,
        temperature=0 if greedy else temperature,
        top_k=None if greedy else top_k,
        eos_id=tokenizer.eos_id,
    )
    return tokenizer.decode(out[0, len(ids):].tolist())


def load_tokenizer_config(path, map_location="cpu"):
    """Carrega so tokenizer + config + num_params (sem instanciar o modelo).

    Usado pelo backend ONNX: a geracao usa o .onnx via onnxruntime, mas o
    tokenizer e os metadados (block_size, train_alphabets...) vêm do .pt.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    tsd = ckpt["tokenizer"]
    if tsd.get("byte_level"):
        tokenizer = ByteTokenizer.from_state_dict(tsd)
    elif "items" in tsd:
        tokenizer = ItemChTokenizer.from_state_dict(tsd)
    else:
        tokenizer = CharTokenizer.from_state_dict(tsd)
    cfg = ckpt["config"]
    state = ckpt["model"]
    num_params = sum(
        t.numel() for t in state.values() if hasattr(t, "numel")
    )
    # state_dict salva head.weight separado mesmo com tie_embeddings=True
    # (peso compartilhado); model.num_params conta 1x. Ajusta p/ display igual.
    if cfg.get("tie_embeddings") and "tok_emb.weight" in state and "head.weight" in state:
        num_params -= state["head.weight"].numel()
    return tokenizer, cfg, int(num_params)


def resolve_onnx_path(checkpoint_path, onnx_path=None):
    if onnx_path:
        return onnx_path
    base, _ext = os.path.splitext(checkpoint_path)
    return base + ".onnx"


def load_onnx_session(onnx_path, providers=None):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "onnxruntime nao instalado. Instale com: pip install onnxruntime"
        ) from exc
    if not os.path.exists(onnx_path):
        raise SystemExit(
            f"Arquivo ONNX nao encontrado: {onnx_path}\n"
            f"Reexporte com: python export_onnx.py --checkpoint <modelo.pt> --out {onnx_path}"
        )
    if providers is None:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, providers=providers)


def generate_continuation_onnx(
    session, tokenizer, prompt, block_size, max_new_tokens=None,
    greedy=True, temperature=0.8, top_k=5,
):
    """Loop greedy/amostragem via ONNX (forward de sequencia completa)."""
    import numpy as np

    ids = tokenizer.encode(prompt)
    input_name = session.get_inputs()[0].name
    if max_new_tokens is None:
        max_new_tokens = block_size
    max_new = max(1, min(int(max_new_tokens), max(1, block_size - len(ids))))
    for _ in range(max_new):
        window = ids[-block_size:]
        inp = np.array([window], dtype=np.int64)
        logits = session.run(None, {input_name: inp})[0]  # (1, T, V)
        last = logits[0, -1, :].astype(np.float64)
        if greedy or temperature == 0:
            next_id = int(np.argmax(last))
        else:
            scaled = last / max(1e-8, float(temperature))
            if top_k is not None:
                k = min(int(top_k), scaled.shape[0])
                thresh = np.sort(scaled)[-k]
                scaled = np.where(scaled < thresh, -np.inf, scaled)
            m = np.max(scaled)
            exp = np.exp(scaled - m)
            probs = exp / np.sum(exp)
            next_id = int(np.random.choice(len(probs), p=probs))
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    return tokenizer.decode(ids[len(tokenizer.encode(prompt)):])


def clean_answer(text):
    for stop in ("<eos>", "<bos>", "<pad>"):
        i = text.find(stop)
        if i != -1:
            text = text[:i]
    return text.strip()


@torch.no_grad()
def evaluate_exact_match(model, tokenizer, prompts, answers, cats=None, greedy=True):
    """Retorna lista de tuplas (prompt, esperado, previsto, ok, categoria)."""
    results = []
    for i, (prompt, gold) in enumerate(zip(prompts, answers)):
        cont = generate_continuation(model, tokenizer, prompt, greedy=greedy)
        pred = clean_answer(cont)
        cat = cats[i] if cats else "all"
        results.append((prompt, gold, pred, pred == gold.strip(), cat))
    return results


def summarize(results):
    """Agrega acuracia total e por categoria a partir de evaluate_exact_match."""
    total = len(results)
    correct = sum(int(r[3]) for r in results)
    per = {}
    for _p, _g, _pred, ok, cat in results:
        c, n = per.get(cat, (0, 0))
        per[cat] = (c + int(ok), n + 1)
    return correct, total, per


def evaluate_exact_match_onnx(session, tokenizer, block_size, prompts, answers, cats=None, greedy=True):
    """Mesmo que evaluate_exact_match, mas gerando via ONNX."""
    results = []
    for i, (prompt, gold) in enumerate(zip(prompts, answers)):
        cont = generate_continuation_onnx(
            session, tokenizer, prompt, block_size, greedy=greedy,
        )
        pred = clean_answer(cont)
        cat = cats[i] if cats else "all"
        results.append((prompt, gold, pred, pred == gold.strip(), cat))
    return results
