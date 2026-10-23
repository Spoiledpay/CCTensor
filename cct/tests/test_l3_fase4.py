"""Testa o CCT-L3 (shuffle) e a Fase 4 (encode-on-update + ECD)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "cct"))
import numpy as np
import torch

from cct.codec import get_codec
from cct.tensor import CCTTensor
from cct.training import CCTLinear, ecd, ecd_report, ecd_time, measure_encode_on_update

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'XX'}] {name}  {detail}")


def test_l3_exact():
    print("\n== L3 bit-exact e CR ==")
    torch.manual_seed(0)
    cases = [
        ("bf16 2-D", torch.randn(64, 96).to(torch.bfloat16)),
        ("bf16 1-D", torch.randn(4096).to(torch.bfloat16)),
        ("bf16 impar", torch.randn(3, 333).to(torch.bfloat16)),
        ("zeros", torch.zeros(32, 64).to(torch.bfloat16)),
        ("constante", torch.full((16, 16), 0.03125).to(torch.bfloat16)),
        ("denormais", (torch.randn(32, 32) * 1e-40).to(torch.bfloat16)),
    ]
    for name, t in cases:
        c = CCTTensor.from_tensor(t, codec="l3_shuffle", tile_rows=64)
        check(f"L3 bit-exact: {name}", torch.equal(c.decompress(), t),
              f"{c.compression_ratio:.3f}x")

    # pesos com distribuicao realista (std 0.02, como pos-treino)
    w = (torch.randn(256, 288) * 0.02).to(torch.bfloat16)
    c3 = CCTTensor.from_tensor(w, codec="l3_shuffle", tile_rows=64)
    c0 = CCTTensor.from_tensor(w, codec="l0_zstd", tile_rows=64)
    check("L3 bit-exact em pesos realistas", torch.equal(c3.decompress(), w))
    check("L3 bate o zstd (CR)", c3.compression_ratio > c0.compression_ratio,
          f"L3={c3.compression_ratio:.3f}x zstd={c0.compression_ratio:.3f}x")
    check("L3 atinge a meta CR >= 1.3x", c3.compression_ratio >= 1.30,
          f"CR={c3.compression_ratio:.3f}x")
    # hybrid nunca e' pior que o melhor codec
    ch = CCTTensor.from_tensor(w, codec="hybrid", tile_rows=64)
    check("hybrid >= max(l0, l3)",
          ch.compression_ratio >= max(c0.compression_ratio, c3.compression_ratio) - 1e-9,
          f"hybrid={ch.compression_ratio:.3f}x")


def test_phase4():
    print("\n== Fase 4: encode-on-update + ECD ==")
    torch.manual_seed(0)
    blk = CCTLinear(64, 128, codec="l3_shuffle", tile_rows=64)
    blk.refresh()
    check("refresh produz CCTTensor", blk.cct is not None,
          f"CR={blk.compression_ratio:.3f}x encode={blk.encode_ms:.1f}ms")
    check("encode-on-update gera CR > 1", blk.compression_ratio > 1.0)

    # inferencia usa o estado comprimido e bate com F.linear
    blk.eval()
    x = torch.randn(4, 64)
    with torch.no_grad():
        y_cct = blk(x)
        y_ref = torch.nn.functional.linear(x, blk.weight.to(torch.bfloat16).float(), blk.bias)
    rel = float((y_cct - y_ref).abs().max() / (y_ref.abs().max() + 1e-9))
    check("inferencia comprimida == F.linear", rel < 1e-3, f"rel_err={rel:.2e}")

    # trade-off
    blk.train()
    opt = torch.optim.AdamW(blk.parameters(), lr=1e-3)
    x = torch.randn(32, 64)
    y = torch.randn(32, 128) * 0.1
    r = measure_encode_on_update(blk, opt, x, y, steps=15)
    check("mede o trade-off (step + encode)", r["step_ms_cct"] > 0 and r["CR"] > 1.0,
          f"denso={r['step_ms_denso']:.2f}ms cct={r['step_ms_cct']:.2f}ms "
          f"overhead={r['overhead_pct']:+.1f}% CR={r['CR']:.3f}x")
    # ECD: separar memoria de tempo. ECD_mem e' proporcional a 1/CR
    # (circular); quem decide e' ECD_time, que inclui o decode.
    flops = 2 * 128 * 64 * 4
    raw_b = 64 * 128 * 2
    rep = ecd_report(flops, raw_b, raw_b / r["CR"],
                     seconds_decode=r["step_ms_cct"] / 1000.0,
                     seconds_compute=r["step_ms_denso"] / 1000.0)
    check("ECD_mem cresce com a compressao (e' 1/CR)",
          rep["ecd_mem_cct"] > rep["ecd_mem_dense"],
          f"mem_cct={rep['ecd_mem_cct']:.1f} mem_denso={rep['ecd_mem_dense']:.1f}")
    check("ECD_time cai quando o decode domina",
          rep["ecd_time_cct"] < rep["ecd_time_dense"],
          f"time_cct={rep['ecd_time_cct']:.0f} time_denso={rep['ecd_time_dense']:.0f} "
          f"overhead={rep['overhead_pct']:+.0f}%")
    check("ecd_time e' ops/segundo", abs(ecd_time(1000, 0.5) - 2000.0) < 1e-9)
    check("ecd() continua ops/byte", abs(ecd(1000, 100) - 10.0) < 1e-9)


def main():
    print("CCT-L3 + Fase 4 — testes")
    test_l3_exact()
    test_phase4()
    print(f"\n{len(PASS)} OK / {len(FAIL)} falhas")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
