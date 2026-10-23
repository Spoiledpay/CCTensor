"""Cross-entropy HONESTA (held-out) do campo de expoente BF16.

Estima P(exp | contexto) na metade das linhas e avalia na outra metade.
Evita o vies do estimador plug-in, que faz H(exp|left,up)=2.5 parecer real.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "cct"))
import numpy as np
import torch


def cross_entropy(target, ctx, n_ctx, n_sym=256, alpha=1.0, backoff=None):
    """-> bits/simbolo (held-out), treinando a tabela na primeira metade."""
    n = target.size
    half = n // 2
    counts = np.zeros((n_ctx, n_sym), dtype=np.float64)
    np.add.at(counts, (ctx[:half], target[:half]), 1.0)
    # suavizacao de Laplace
    counts += alpha
    if backoff is not None:
        # mistura: P = 0.8 * P(ctx) + 0.2 * P(global)
        glob = np.bincount(target[:half], minlength=n_sym).astype(np.float64) + alpha
        glob /= glob.sum()
        row = counts / counts.sum(axis=1, keepdims=True)
        probs = 0.8 * row + 0.2 * glob[None, :]
    else:
        probs = counts / counts.sum(axis=1, keepdims=True)
    p = probs[ctx[half:], target[half:]]
    return float(-np.log2(np.maximum(p, 1e-12)).mean())


def main():
    ckpt = sys.argv[1]
    st = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    res = {k: [] for k in ("h0", "exp_up", "exp_left", "exp_up_bo", "sign_up")}
    for name, t in st.items():
        if t.dim() != 2 or t.numel() < 1024:
            continue
        a = t.detach().contiguous().numpy()
        u = a.view(np.uint16)
        sign = (u >> 15).astype(np.int64).ravel()
        exp = ((u >> 7) & 0xFF).astype(np.int64).ravel()
        up = np.roll(((u >> 7) & 0xFF).astype(np.int64), 1, axis=0).ravel()
        left = np.roll(((u >> 7) & 0xFF).astype(np.int64), 1, axis=1).ravel()
        sup = np.roll(sign.reshape(u.shape), 1, axis=0).ravel()
        res["h0"].append(cross_entropy(exp, np.zeros_like(exp), 1))
        res["exp_up"].append(cross_entropy(exp, up, 256))
        res["exp_left"].append(cross_entropy(exp, left, 256))
        res["exp_up_bo"].append(cross_entropy(exp, up, 256, backoff=True))
        res["sign_up"].append(cross_entropy(sign, sup, 2, n_sym=2))

    for k, v in res.items():
        print(f"  H_heldout[{k:11s}] = {np.mean(v):.3f} bits")
    print()
    h0 = np.mean(res["h0"])
    hup = np.mean(res["exp_up"])
    hubo = np.mean(res["exp_up_bo"])
    print("CR estimado (sign=1 bit, mantissa=7 bits crus):")
    print(f"  sem contexto no expoente : {16/(1+7+h0):.3f}x")
    print(f"  exp | up                 : {16/(1+7+hup):.3f}x")
    print(f"  exp | up + backoff       : {16/(1+7+hubo):.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
