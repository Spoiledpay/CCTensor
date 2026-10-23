"""Entropia condicional dos campos BF16 dados os vizinhos (esquerda/cima).

Se H(exp | vizinho) << H(exp), um codificador com CONTEXTO ganha o que o
Huffman ordem-0 (L1) e o RLE por bit-plane (L2) nao conseguem.
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "cct"))
import numpy as np
import torch


def h_order0(v):
    v = np.asarray(v).astype(np.int64).ravel()
    if v.size == 0:
        return 0.0
    c = np.bincount(v)
    p = c[c > 0] / v.size
    return float(-(p * np.log2(p)).sum())


def h_cond(target, ctx):
    """H(target | ctx) em bits."""
    target = np.asarray(target).astype(np.int64).ravel()
    ctx = np.asarray(ctx).astype(np.int64).ravel()
    n = target.size
    if n == 0:
        return 0.0
    h = 0.0
    # agrega por contexto
    for cv in np.unique(ctx):
        mask = ctx == cv
        w = mask.sum() / n
        h += w * h_order0(target[mask])
    return float(h)


def fields(a):
    u = np.ascontiguousarray(a).view(np.uint16)
    return ((u >> 15).astype(np.uint8),
            ((u >> 7) & 0xFF).astype(np.uint8),
            (u & 0x7F).astype(np.uint8))


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "modelos", "tinygpt_curado.pt")
    st = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    acc = {k: [] for k in ("sign", "exp", "mant", "mant_msb")}
    for name, t in st.items():
        if t.dim() != 2 or t.numel() < 1024:
            continue
        a = t.detach().contiguous().numpy()
        sign, exp, mant = fields(a)
        left = np.roll(exp, 1, axis=1)
        up = np.roll(exp, 1, axis=0)
        both = left.astype(np.int64) * 256 + up
        acc["sign"].append((h_order0(sign), h_cond(sign, left)))
        acc["exp"].append((h_order0(exp), h_cond(exp, left),
                           h_cond(exp, up), h_cond(exp, both)))
        acc["mant"].append((h_order0(mant), h_cond(mant, left)))
        msb = (mant >> 6).astype(np.uint8)
        acc["mant_msb"].append((h_order0(msb), h_cond(msb, exp)))

    print(f"checkpoint: {os.path.basename(ckpt)}")
    s = np.mean([x[0] for x in acc["sign"]])
    sl = np.mean([x[1] for x in acc["sign"]])
    print(f"sign : H={s:.3f}  H|left={sl:.3f}")
    e0 = np.mean([x[0] for x in acc["exp"]])
    el = np.mean([x[1] for x in acc["exp"]])
    eu = np.mean([x[2] for x in acc["exp"]])
    eb = np.mean([x[3] for x in acc["exp"]])
    print(f"exp  : H={e0:.3f}  H|left={el:.3f}  H|up={eu:.3f}  H|left,up={eb:.3f}")
    m0 = np.mean([x[0] for x in acc["mant"]])
    ml = np.mean([x[1] for x in acc["mant"]])
    print(f"mant : H={m0:.3f} (de 7)  H|left={ml:.3f}")
    b0 = np.mean([x[0] for x in acc["mant_msb"]])
    bl = np.mean([x[1] for x in acc["mant_msb"]])
    print(f"mant MSB (1 bit): H={b0:.3f}  H|exp={bl:.3f}")
    print()
    for name, bits in (
        ("ordem-0", s + e0 + m0),
        ("contexto left", sl + el + ml),
        ("contexto left+up (exp)", sl + eb + ml),
    ):
        print(f"CR com {name:24s}: {16/bits:.3f}x  ({bits:.2f} bits/valor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
