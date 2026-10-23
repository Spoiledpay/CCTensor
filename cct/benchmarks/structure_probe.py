"""Analisa a estrutura BF16 dos pesos: cardinalidade de expoentes por tile e
entropia dos campos. Base para dimensionar o CCT-L3.
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "cct"))
import numpy as np
import torch


def bits_entropy(vals):
    vals = np.asarray(vals, dtype=np.uint8)
    n = vals.size
    if n == 0:
        return 0.0
    counts = np.bincount(vals, minlength=256)
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "modelos", "tinygpt_curado.pt")
    if not os.path.exists(ckpt):
        print(f"sem checkpoint: {ckpt}")
        return 1
    st = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]

    all_k = []
    ent = {"sign": [], "exp": [], "mant": []}
    tot_bits = {"raw": 0, "l3": 0}
    tile_rows = 16

    for name, t in st.items():
        if t.dim() != 2 or t.numel() < 1024:
            continue
        a = t.detach().contiguous().numpy()
        rows, cols = a.shape
        for r0 in range(0, rows, tile_rows):
            chunk = a[r0:r0 + tile_rows]
            u = chunk.view(np.uint16).reshape(-1)
            sign = (u >> 15).astype(np.uint8)
            exp = ((u >> 7) & 0xFF).astype(np.uint8)
            mant = (u & 0x7F).astype(np.uint8)
            k = int(np.unique(exp).size)
            all_k.append(k)
            for nm, v in (("sign", sign), ("exp", exp), ("mant", mant)):
                ent[nm].append(bits_entropy(v))
            we = 0 if k <= 1 else int(math.ceil(math.log2(k)))
            tot_bits["raw"] += u.size * 16
            tot_bits["l3"] += u.size * (1 + we + 7)

    ks = np.array(all_k)
    print(f"checkpoint: {os.path.basename(ckpt)}  |  tiles analisados: {ks.size}")
    print(f"expoentes distintos por tile: min={ks.min()} mediana={int(np.median(ks))} "
          f"p90={int(np.percentile(ks, 90))} max={ks.max()}")
    print(f"bits-por-expoente efetivos (ceil(log2 k)): {np.ceil(np.log2(ks)).mean():.2f}")
    for nm in ("sign", "exp", "mant"):
        print(f"entropia de {nm:5s}: {np.mean(ent[nm]):.3f} bits/valor "
              f"(de {8 if nm!='mant' else 7} bits)")
    cr = tot_bits["raw"] / tot_bits["l3"]
    print(f"\nCR estimado do L3 simples (sign 1 + exp ceil(log2k) + mant 7): {cr:.3f}x")
    print(f"  = economia de {100*(1-1/cr):.1f}%")

    # e se o expoente fosse codificado pela entropia (limite)
    e_exp = np.mean(ent["exp"])
    e_sign = np.mean(ent["sign"])
    best_bits = e_sign + e_exp + 7
    print(f"CR limite com entropia (sign+exp por entropia): {16/best_bits:.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
