"""Mede CR por codec no checkpoint REAL (6.1M) e o decode GB/s."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "cct"))

import torch

from cct.tensor import CCTTensor
from cct.runtime import measure_decode_throughput

GROUPS = [
    ("tok_emb", "embedding"), ("pos_emb", "outros"), ("head", "output_head"),
    ("attn.c_attn", "attn_qkv"), ("attn.c_proj", "attn_out"),
    ("mlp.fc", "mlp_up"), ("mlp.proj", "mlp_down"), ("ln", "layernorm"),
]


def group_of(name):
    for key, g in GROUPS:
        if key in name:
            return g
    return "outros"


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "modelos", "tinygpt_curado.pt")
    st = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    codecs = ("l0_zstd", "l1_entropy", "l2_bitplane", "hybrid")
    tot = {c: [0, 0] for c in codecs}
    print(f"checkpoint: {ckpt}")
    print(f"{'tensor':<34}{'grupo':<12}{'KB':>8}{'l0':>8}{'l1':>8}{'l2':>8}{'hybrid':>8}")
    print("-" * 94)
    for name, t in st.items():
        if t.dim() != 2 or t.numel() < 1024:
            continue
        r = {}
        for c in codecs:
            x = CCTTensor.from_tensor(t, codec=c, tile_rows=16)
            r[c] = x.compression_ratio
            tot[c][0] += x.raw_bytes
            tot[c][1] += x.compressed_bytes
        print(f"{name[:34]:<34}{group_of(name):<12}"
              f"{t.numel() * t.element_size() / 1024:>8.1f}"
              f"{r['l0_zstd']:>8.3f}{r['l1_entropy']:>8.3f}"
              f"{r['l2_bitplane']:>8.3f}{r['hybrid']:>8.3f}")
    print("-" * 94)
    for c, (raw, comp) in tot.items():
        print(f"GLOBAL {c:<11} {raw / 1e6:6.2f} MB -> {comp / 1e6:6.2f} MB  "
              f"CR={raw / comp:.3f}x   (economia {100 * (1 - comp / raw):.1f}%)")

    # decode throughput dos codecs (tensor grande)
    big = max((t for t in st.values() if t.dim() == 2), key=lambda t: t.numel())
    print(f"\ndecode throughput (tensor {tuple(big.shape)}, "
          f"{big.numel() * big.element_size() / 1e6:.2f} MB):")
    for c in codecs:
        x = CCTTensor.from_tensor(big, codec=c, tile_rows=16)
        t0 = time.time()
        x.decompress()
        dt = time.time() - t0
        gb = (x.raw_bytes / dt) / 1e9 if dt else 0
        print(f"  {c:<12} CR={x.compression_ratio:.3f}x  decode={gb:.3f} GB/s")


if __name__ == "__main__":
    main()
