"""Tabela unica CR x T_decode x ECD (a lacuna do README §5/§6).

Mede, no MESMO tensor BF16:

    codec | CR | encode ms/MB | decode ms/MB | decode GB/s | ECD_mem | ECD_time | vs denso

`ECD_mem = ops / bytes_movidos` e' proporcional a 1/CR (por isso o "4.0 -> 6.1"
do README antigo era circular). `ECD_time = ops / (T_decode + T_compute)` e'
o numero que decide.

Tambem cobre tensores pequenos e dimensoes impares (padding do tile), que o
README nao reportava.

Uso:
    python cct/benchmarks/codec_table.py
    python cct/benchmarks/codec_table.py --checkpoint modelos/chat_en.pt
    python cct/benchmarks/codec_table.py --shape 2048 1024 --tile-rows 64
"""

import argparse
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
sys.path.insert(0, _PKG)
sys.path.insert(0, _HERE)

from baselines import format_table as format_baselines, measure as measure_baselines  # noqa: E402
from cct.codec import CODECS                                    # noqa: E402
from cct.tensor import CCTTensor                                # noqa: E402
from cct.training import ecd_report                             # noqa: E402


def _best(fn, arg, reps=3):
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(arg)
        best = min(best, time.perf_counter() - t0)
    return out, best


def _time_dense(t, x, dtype, reps=20):
    tt = t.detach().to(dtype)
    xx = x.detach().to(dtype)
    for _ in range(2):
        _ = tt @ xx
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = tt @ xx
        best = min(best, time.perf_counter() - t0)
    return best


def measure_cct(t, codec, tile_rows, reps=2):
    raw_mb = t.numel() * 2 / 1e6
    cct, t_enc = _best(lambda _: CCTTensor.from_tensor(
        t, codec=codec, tile_rows=tile_rows), None, reps=1)
    back, t_dec = _best(lambda c: c.decompress(), cct, reps=reps)
    ok = torch.equal(back, t)
    return {
        "name": codec,
        "cr": cct.compression_ratio,
        "enc_ms_per_mb": 1000.0 * t_enc / raw_mb,
        "dec_ms_per_mb": 1000.0 * t_dec / raw_mb,
        "dec_gbps": (cct.raw_bytes / t_dec) / 1e9 if t_dec else 0.0,
        "bit_exact": ok,
        "cct": cct,
        "t_dec": t_dec,
    }


def ecd_columns(row, t, t_compute):
    ops = 2.0 * t.shape[0] * t.shape[1]
    raw_bytes = t.numel() * 2
    comp_bytes = raw_bytes / max(1e-9, row["cr"])
    rep = ecd_report(ops, raw_bytes, comp_bytes, row["t_dec"], t_compute)
    return {
        "ecd_mem": rep["ecd_mem_cct"],
        "ecd_time": rep["ecd_time_cct"],
        "ecd_time_dense": rep["ecd_time_dense"],
        "slower": rep["cct_slower"],
        "overhead_pct": rep["overhead_pct"],
    }


def print_codec_table(t, tile_rows):
    print(f"\n== Tensor {tuple(t.shape)} BF16 "
          f"({t.numel() * 2 / 1e6:.2f} MB), tile_rows={tile_rows} ==")
    x = torch.randn(t.shape[1], 1)
    t_compute = _time_dense(t, x, torch.float32)
    t_compute_bf16 = _time_dense(t, x, torch.bfloat16)
    ops = 2.0 * t.shape[0] * t.shape[1]
    print(f"   GEMM de referencia ({t.shape[0]}x{t.shape[1]} @ {t.shape[1]}x1): "
          f"fp32 {t_compute*1e3:.2f} ms | bf16 {t_compute_bf16*1e3:.2f} ms")
    print(f"   ops uteis = {ops/1e6:.2f} Mops")
    print()
    print("| codec | CR | encode ms/MB | decode ms/MB | decode GB/s | "
          "ECD_mem ops/B | ECD_time Mops/s | vs denso |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")

    rows = []
    seen = set()
    for codec in list(CODECS):
        # `l3_structured` e' um alias de `l3_shuffle`
        if codec == "l3_structured":
            continue
        r = measure_cct(t, codec, tile_rows)
        if (r["cr"], round(r["enc_ms_per_mb"], 1)) in seen:
            continue
        seen.add((r["cr"], round(r["enc_ms_per_mb"], 1)))
        e = ecd_columns(r, t, t_compute)
        r.update(e)
        rows.append(r)
        mark = "" if r["bit_exact"] else " ⚠NAO-BITEXACT"
        slower = "lento" if e["slower"] else "GANHA"
        print(f"| {codec}{mark} | {r['cr']:.3f}x | {r['enc_ms_per_mb']:.1f} | "
              f"{r['dec_ms_per_mb']:.1f} | {r['dec_gbps']:.2f} | "
              f"{e['ecd_mem']:.2f} | {e['ecd_time']/1e6:.2f} | "
              f"{slower} ({e['overhead_pct']:+.0f}%) |")

    cct_best = max((r for r in rows if r["bit_exact"]), key=lambda r: r["cr"])
    print(f"\n   melhor codec CCT por CR: {cct_best['name']} "
          f"({cct_best['cr']:.3f}x); ECD_time {cct_best['ecd_time']/1e6:.2f} Mops/s "
          f"vs denso {cct_best['ecd_time_dense']/1e6:.2f} Mops/s")
    return rows, t_compute


def print_small_tensors(tile_rows):
    """Padding de tile / tensores pequenos (README §3.3)."""
    print("\n== Tensores pequenos e dimensoes impares (padding do tile) ==")
    print("| shape | KB | tempo ms | l3_shuffle | l0_zstd |")
    print("|---|---:|---:|---:|---:|")
    for shape in [(259, 192), (192, 768), (768, 192), (1, 1152), (3, 333),
                  (17, 65), (64, 64)]:
        t = (torch.randn(*shape) * 0.02).to(torch.bfloat16)
        out = {}
        total_ms = 0.0
        for codec in ("l3_shuffle", "l0_zstd"):
            t0 = time.perf_counter()
            c, _ = _best(lambda _: CCTTensor.from_tensor(
                t, codec=codec, tile_rows=tile_rows), None, reps=1)
            _ = c.decompress()
            total_ms += 1000 * (time.perf_counter() - t0)
            out[codec] = c.compression_ratio
        print(f"| {shape} | {t.numel()*2/1024:.1f} | {total_ms:.1f} | "
              f"{out['l3_shuffle']:.3f}x | {out['l0_zstd']:.3f}x |")
    print("\n   (tile_rows nao forca padding: o ultimo tile e' menor; o "
          "cabecalho por tile pesa mais em tensores pequenos.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(_ROOT, "modelos", "chat_en.pt"))
    ap.add_argument("--shape", nargs=2, type=int, default=(2048, 1024))
    ap.add_argument("--tile-rows", type=int, default=64)
    ap.add_argument("--baselines", action="store_true", default=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        t = max((v for v in state.values()
                 if hasattr(v, "dim") and v.dim() == 2 and v.numel() >= 4096),
                key=lambda v: v.numel())
        t = t.detach().to(torch.bfloat16).contiguous()
        print(f"fonte: {args.checkpoint} (maior tensor 2-D)")
    else:
        t = (torch.randn(*args.shape) * 0.02).to(torch.bfloat16)
        print(f"fonte: sintetico N(0,0.02) {tuple(args.shape)}")

    rows, t_compute = print_codec_table(t, args.tile_rows)

    if args.baselines:
        print("\n== Baselines externos no mesmo buffer ==")
        raw = t.view(torch.uint8).numpy().tobytes()
        b = measure_baselines(raw, reps=3)
        print(format_baselines(b))
        best = max((r for r in b if r.get("cr")), key=lambda r: r["cr"])
        cct = max(rows, key=lambda r: r["cr"])
        print(f"\n   CCT melhor: {cct['name']} {cct['cr']:.3f}x @ "
              f"{cct['dec_gbps']:.2f} GB/s")
        print(f"   baseline melhor CR: {best['name']} {best['cr']:.3f}x @ "
              f"{best['dec_gbps']:.2f} GB/s")

    print_small_tensors(args.tile_rows)
    print("\nLeitura: ECD_mem cresce com 1/CR (circular). A coluna que decide e' "
          "ECD_time,\nque inclui o decode — e em CPU pequena ela quase sempre "
          "perde para o denso.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
