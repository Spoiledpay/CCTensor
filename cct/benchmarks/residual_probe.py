"""Entropia do RESIDUO do expoente/mantissa para varios preditores 2-D.
Decide o design do CCT-L3 (preditor + codificador de entropia).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "cct"))
import numpy as np
import torch


def h0(v):
    v = np.asarray(v).astype(np.int64).ravel()
    c = np.bincount(v)
    p = c[c > 0] / v.size
    return float(-(p * np.log2(p)).sum())


def main():
    ckpt = sys.argv[1]
    st = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    acc = {k: [] for k in ("up", "left", "paeth", "avg",
                           "mant_up", "mant_left", "mant_none", "sign_left")}
    nvals = 0
    for name, t in st.items():
        if t.dim() != 2 or t.numel() < 1024:
            continue
        a = t.detach().contiguous().numpy()
        u = a.view(np.uint16)
        sign = (u >> 15).astype(np.int16)
        exp = ((u >> 7) & 0xFF).astype(np.int16)
        mant = (u & 0x7F).astype(np.int16)
        nvals += exp.size
        L = np.roll(exp, 1, 1)
        U = np.roll(exp, 1, 0)
        UL = np.roll(np.roll(exp, 1, 0), 1, 1)
        P = np.where(np.abs(L - UL) < np.abs(U - UL), L, U)
        P = np.where((L == UL) & (U == UL), L, P)
        A = (L + U) // 2
        acc["up"].append(h0((exp - U) & 0xFF))
        acc["left"].append(h0((exp - L) & 0xFF))
        acc["paeth"].append(h0((exp - P) & 0xFF))
        acc["avg"].append(h0((exp - A) & 0xFF))
        ML = np.roll(mant, 1, 1)
        MU = np.roll(mant, 1, 0)
        acc["mant_up"].append(h0((mant - MU) & 0x7F))
        acc["mant_left"].append(h0((mant - ML) & 0x7F))
        acc["mant_none"].append(h0(mant))
        SL = np.roll(sign, 1, 1)
        acc["sign_left"].append(h0((sign - SL) & 1))

    print(f"checkpoint: {os.path.basename(ckpt)}  ({nvals:,} valores)")
    print("entropia do residuo (bits/valor):")
    for k, v in acc.items():
        print(f"  {k:10s} {np.mean(v):.3f}")
    print()
    print("CR estimado = 16 / (1 sign + exp_resid + 7 mantissa):")
    for k in ("up", "left", "paeth", "avg"):
        b = 1.0 + np.mean(acc[k]) + 7.0
        print(f"  pred={k:6s} -> {b:5.2f} bits -> CR {16/b:.3f}x")
    print()
    print("se tambem codificar a mantissa por contexto esquerdo:")
    for k in ("up", "left", "paeth", "avg"):
        b = 1.0 + np.mean(acc[k]) + np.mean(acc["mant_left"])
        print(f"  pred={k:6s} -> {b:5.2f} bits -> CR {16/b:.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
