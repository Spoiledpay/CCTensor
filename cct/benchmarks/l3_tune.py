"""Otimiza o CCT-L3 (field shuffle): tamanho de tile, nivel zstd, bit-packing."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "cct"))
import numpy as np
import torch
import zstandard as zstd

from cct.codec.l3_structured import _pack_bits


def zc(data, level):
    return zstd.ZstdCompressor(level=level).compress(data)


def pack7(mant):
    """7 bits por mantissa -> ceil(7N/8) bytes."""
    return _pack_bits(mant, 7)


def main():
    ckpt = sys.argv[1]
    st = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]

    def variants(t, tile_rows, level):
        u = t.detach().to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
        rows, cols = u.shape
        acc = {"shuffle3": 0, "packed": 0, "raw": 0}
        for r0 in range(0, rows, tile_rows):
            ch = u[r0:r0 + tile_rows].reshape(-1)
            if ch.size == 0:
                continue
            sign = (ch >> 15).astype(np.uint8)
            exp = ((ch >> 7) & 0xFF).astype(np.uint8)
            mant = (ch & 0x7F).astype(np.uint8)
            acc["raw"] += ch.nbytes
            acc["shuffle3"] += (len(zc(sign.tobytes(), level))
                                + len(zc(exp.tobytes(), level))
                                + len(zc(mant.tobytes(), level)))
            acc["packed"] += (len(zc(_pack_bits(sign, 1), level))
                              + len(zc(exp.tobytes(), level))
                              + len(pack7(mant)))
        return acc

    tensors = [(n, t) for n, t in st.items() if t.dim() == 2 and t.numel() >= 1024]
    for tile_rows in (16, 32, 64, 128, 10**9):
        for level in (6, 12, 19):
            tot = {"shuffle3": 0, "packed": 0, "raw": 0}
            for _n, t in tensors:
                v = variants(t, tile_rows if tile_rows < 10**9 else t.shape[0], level)
                for k in tot:
                    tot[k] += v[k]
            print(f"tile_rows={str(tile_rows)[:6]:>6} zstd={level:2d}  "
                  f"shuffle3 CR={tot['raw']/tot['shuffle3']:.3f}x  "
                  f"packed CR={tot['raw']/tot['packed']:.3f}x")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
