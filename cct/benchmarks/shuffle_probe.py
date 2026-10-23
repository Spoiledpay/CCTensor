"""Testa 'field shuffle': separar sign/expoente/mantissa e comprimir cada fluxo
com zstd (em vez de zstd no BF16 interleaved). E' o que o Blosc chama de
shuffle e costuma ganhar muito em floats.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "cct"))
import numpy as np
import torch

from cct.codec import get_codec

Z = get_codec("l0_zstd")


def fields(u):
    return (u >> 15).astype(np.uint8), ((u >> 7) & 0xFF).astype(np.uint8), (u & 0x7F).astype(np.uint8)


def sz(blob):
    return len(blob)


def main():
    ckpt = sys.argv[1]
    st = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    tot = {"raw": 0, "zstd": 0, "shuffle3": 0, "shuffle3_res": 0, "exp_only": 0}
    rows = []
    for name, t in st.items():
        if t.dim() != 2 or t.numel() < 1024:
            continue
        u = t.detach().to(torch.bfloat16).contiguous().view(torch.uint16).numpy().reshape(-1)
        raw = u.nbytes
        sign, exp, mant = fields(u)
        N = u.size

        # 1) zstd no interleaved
        c_z = sz(Z.compress(u.tobytes()))

        # 3) shuffle: 3 fluxos
        c_s = (sz(Z.compress(sign.tobytes())) + sz(Z.compress(exp.tobytes()))
               + sz(Z.compress(mant.tobytes())))

        # 3b) expoente como residuo vertical
        e64 = exp.astype(np.int64)
        cols = t.shape[1]
        up = np.zeros_like(e64)
        up[cols:] = e64[:-cols]
        res = ((e64 - up) & 0xFF).astype(np.uint8)
        c_sr = (sz(Z.compress(sign.tobytes())) + sz(Z.compress(res.tobytes()))
                + sz(Z.compress(mant.tobytes())))

        # 4) so separar o expoente (sign+mant interleaved)
        c_e = sz(Z.compress(exp.tobytes())) + sz(Z.compress((u & 0x7FFF).astype(np.uint16).tobytes()))

        tot["raw"] += raw
        tot["zstd"] += c_z
        tot["shuffle3"] += c_s
        tot["shuffle3_res"] += c_sr
        tot["exp_only"] += c_e
        rows.append((name, raw, c_z, c_s, c_sr, c_e))

    print(f"{'tensor':<32}{'KB':>8}{'zstd':>8}{'shuf3':>8}{'shuf+res':>9}{'expsep':>8}")
    print("-" * 74)
    for n, raw, a, b, c, d in rows:
        print(f"{n[:32]:<32}{raw/1024:>8.1f}{raw/a:>8.3f}{raw/b:>8.3f}{raw/c:>9.3f}{raw/d:>8.3f}")
    print("-" * 74)
    for k in ("zstd", "shuffle3", "shuffle3_res", "exp_only"):
        print(f"GLOBAL {k:<14} CR={tot['raw']/tot[k]:.3f}x  "
              f"(economia {100*(1-tot[k]/tot['raw']):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

