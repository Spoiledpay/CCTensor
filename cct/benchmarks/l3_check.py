"""Valida o CCT-L3: bit-exact + CR vs zstd em pesos BF16."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "cct"))
import numpy as np
import torch

from cct.tensor import CCTTensor
from cct.codec import get_codec


def check_roundtrip(name, t):
    """-> (raw, cmps) por codec, checando bit-exact."""
    out = {}
    for codec in ("l0_zstd", "l1_entropy", "l2_bitplane", "l3_shuffle"):
        t0 = time.time()
        c = CCTTensor.from_tensor(t, codec=codec, tile_rows=16)
        enc = time.time() - t0
        back = c.decompress()
        ok = torch.equal(back, t)
        out[codec] = (c.raw_bytes, c.compressed_bytes, enc, ok)
    return out


def main():
    ckpt = sys.argv[1]
    st = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    tot = {c: [0, 0, 0.0] for c in
           ("l0_zstd", "l1_entropy", "l2_bitplane", "l3_shuffle")}
    fails = 0
    print(f"{'tensor':<32}{'KB':>8}{'l0':>8}{'l1':>8}{'l2':>8}{'l3':>8}")
    print("-" * 72)
    for name, t in st.items():
        if t.dim() != 2 or t.numel() < 1024:
            continue
        tb = t.detach().to(torch.bfloat16).contiguous()
        r = check_roundtrip(name, tb)
        for c, (raw, cmp_, _e, ok) in r.items():
            tot[c][0] += raw
            tot[c][1] += cmp_
            fails += int(not ok)
        print(f"{name[:32]:<32}{tb.numel()*2/1024:>8.1f}"
              + "".join(f"{r[c][0]/r[c][1]:>8.3f}" for c in
                        ("l0_zstd", "l1_entropy", "l2_bitplane", "l3_shuffle")))
    print("-" * 72)
    print("GLOBAL (BF16):")
    for c in tot:
        raw, cmp_, _ = tot[c]
        print(f"  {c:<14} {raw/1e6:6.2f} MB -> {cmp_/1e6:6.2f} MB  CR={raw/cmp_:.3f}x")
    print(f"\nbit-exact: {'OK' if fails == 0 else f'{fails} FALHAS'}")

    # singleton / 1-D / fp32: L3 deve cair para hybrid sem quebrar
    for label, tt in (
        ("1-D", torch.randn(4096).to(torch.bfloat16)),
        ("fp32", torch.randn(64, 64)),
        ("linha unica", torch.randn(1, 1152).to(torch.bfloat16)),
        ("numel impar", torch.randn(3, 333).to(torch.bfloat16)),
        ("zeros", torch.zeros(32, 64).to(torch.bfloat16)),
    ):
        c = CCTTensor.from_tensor(tt, codec="l3_shuffle", tile_rows=16)
        ok = torch.equal(c.decompress(), tt)
        print(f"  {label:<14} codec={c.codec_name:<12} bit-exact={ok}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

