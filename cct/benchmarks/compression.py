"""Benchmark do CCT em escala PEQUENA (validacao da pipeline, nao do modelo real).

Objetivo: confirmar, em segundos, que a pipeline completa funciona:
    tensor -> tiles -> compressao (best-of) -> indice -> decode -> GEMM
e medir eficiencia (CR, encode, decode GB/s, CCT-GEMM vs denso) num
"mini-modelo" (TinyGPT reduzido) com tensores pequenos.

O teste pesado (pesos do modelo real de 6M/16M) fica para depois, com o
decoder otimizado — em Python puro o Huffman e lento demais.

Uso:
    .\\.venv\\Scripts\\python.exe cct\\benchmarks\\compression.py
    .\\.venv\\Scripts\\python.exe cct\\benchmarks\\compression.py --real   # modelo real (lento)
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cct.codec import compress_best, get_codec   # noqa: E402
from cct.ops import cct_matmul                    # noqa: E402
from cct.runtime import measure_decode_throughput  # noqa: E402
from cct.tensor import CCTTensor                  # noqa: E402

CANDIDATES = ("l0_zstd", "l1_entropy", "l2_bitplane", "l3_shuffle")


def group_of(name: str) -> str:
    if "tok_emb" in name or "position_embd" in name:
        return "embedding"
    if "c_attn" in name or "attn_qkv" in name:
        return "attn_qkv"
    if "c_proj" in name or "attn_output" in name:
        return "attn_out"
    if "mlp.fc" in name or "ffn_up" in name:
        return "mlp_up"
    if "mlp.proj" in name or "ffn_down" in name:
        return "mlp_down"
    if "ln1" in name or "ln2" in name or "ln_f" in name:
        return "layernorm"
    if "head" in name or "output" in name:
        return "output_head"
    return "outros"


def mini_model():
    """Mini TinyGPT (pequeno) para exercitar a pipeline em segundos."""
    from minigpt.model import TinyGPT
    torch.manual_seed(0)
    return TinyGPT(vocab_size=73, block_size=64, n_embd=64, n_head=4, n_layer=2,
                   tie_embeddings=True).state_dict()


def measure_tensor(tensor, tile_rows=16):
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    res = {}
    for cname in CANDIDATES:
        codec = get_codec(cname)
        t0 = time.time()
        blob = codec.compress(raw)
        t_enc = time.time() - t0
        t0 = time.time()
        back = codec.decompress(blob, len(raw))
        t_dec = time.time() - t0
        assert back == raw, f"{cname}: nao foi bit-exact"
        res[cname] = {"cr": len(raw) / max(1, len(blob)), "size": len(blob),
                      "enc": t_enc, "dec": t_dec,
                      "dec_gbps": (len(raw) / t_dec) / 1e9 if t_dec > 0 else 0.0}
    payload, best = compress_best(raw)
    res["hybrid"] = {"cr": len(raw) / max(1, len(payload)), "size": len(payload),
                     "enc": res[best]["enc"], "dec": res[best]["dec"],
                     "dec_gbps": res[best]["dec_gbps"]}
    res["_best"] = best
    return res


def run(state, title):
    print(f"\n=== {title} ===")
    print(f"{'tensor':34s} {'grupo':11s} {'KB':>7s} {'l0':>7s} {'l1':>7s} "
          f"{'l2':>7s} {'hyb':>7s}  {'venc':11s} {'dec GB/s':>9s}")
    print("-" * 112)
    groups = defaultdict(lambda: defaultdict(int))
    tot_raw = tot_cmp = 0
    for name, t in state.items():
        if t.dtype not in (torch.bfloat16, torch.float32) or t.dim() != 2:
            continue
        if t.numel() < 256:
            continue
        r = measure_tensor(t)
        g = group_of(name)
        groups[g][r["_best"]] += 1
        raw_kb = t.numel() * t.element_size() / 1024
        tot_raw += raw_kb * 1024
        tot_cmp += r["hybrid"]["size"]
        print(f"{name[:34]:34s} {g:11s} {raw_kb:7.1f} {r['l0_zstd']['cr']:7.3f} "
              f"{r['l1_entropy']['cr']:7.3f} {r['l2_bitplane']['cr']:7.3f} "
              f"{r['hybrid']['cr']:7.3f}  {r['_best']:11s} "
              f"{r['hybrid']['dec_gbps']:9.2f}")
    print("\n-- vencedor por grupo --")
    for g, wins in sorted(groups.items()):
        print(f"  {g:11s} " + ", ".join(f"{k}:{v}" for k, v in
                                        sorted(wins.items(), key=lambda kv: -kv[1])))
    print(f"\n  GLOBAL: {tot_raw/1024:.1f} KB -> {tot_cmp/1024:.1f} KB "
          f"(CR={tot_raw/max(1,tot_cmp):.3f}x)")
    return tot_raw, tot_cmp


def pipeline_check():
    """Valida a pipeline completa em tensores pequenos (segundos)."""
    print("\n=== Pipeline: tensor -> tiles -> compress -> index -> decode -> GEMM ===")
    torch.manual_seed(0)
    w = torch.randn(256, 128).to(torch.bfloat16)
    x = torch.randn(128, 16)

    t0 = time.time()
    cct = CCTTensor.from_tensor(w, codec="hybrid", tile_rows=32)
    t_comp = time.time() - t0
    print(f"  compressao (hybrid, 4 tiles): {t_comp*1e3:.1f} ms | "
          f"CR={cct.compression_ratio:.3f}x | {cct.n_tiles} tiles")
    print(f"  codecs por tile: {[e.codec for e in cct.index.entries]}")

    back = cct.decompress()
    print(f"  bit-exact: {torch.equal(back, w)}")

    t0 = time.time()
    out = cct_matmul(cct, x)
    t_gemm = time.time() - t0
    ref = w.to(torch.float32) @ x
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    print(f"  CCT-GEMM: {t_gemm*1e3:.1f} ms | rel_err={err:.2e}")

    t0 = time.time()
    for _ in range(200):
        _ = w.to(torch.float32) @ x
    t_dense = max((time.time() - t0) / 200, 1e-6)
    print(f"  GEMM denso (fp32): {t_dense*1e3:.3f} ms | "
          f"ratio CCT/dense = {t_gemm/t_dense:.1f}x")

    gbs = measure_decode_throughput(cct, iters=20)
    print(f"  decode: {gbs:.2f} GB/s (bytes crus)")

    # serializacao
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "w.cct")
        cct.save(p)
        cct2 = CCTTensor.load(p)
        ok = torch.equal(cct2.decompress(), w)
    print(f"  save/load roundtrip bit-exact: {ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="usar os pesos do modelo real (lento em Python puro)")
    ap.add_argument("--checkpoint", default="modelos/especialista_full_6m.pt")
    args = ap.parse_args()

    pipeline_check()
    run(mini_model(), "Mini-modelo (TinyGPT 2 camadas, 64 embd) — tensores pequenos")

    if args.real:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        run(ckpt["model"], f"Pesos reais ({args.checkpoint}) — lento")


if __name__ == "__main__":
    main()
