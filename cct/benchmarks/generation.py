"""Geracao autorregressiva (batch=1) com e sem CCT — o numero que decide.

Por que este benchmark existe (`Projeto.md` §3.2): na geracao token a token
com KV cache, cada peso e' lido A CADA TOKEN. O decode e' pago N_tokens
vezes; e' o pior caso possivel — e e' o caso de uso real de um SLM.

Mede:
    - ms/token e tokens/s de um modelo denso (fp32);
    - o mesmo modelo com as projecoes trocadas por CCT (L3, lossless bf16);
    - o custo PURO de decode por token (soma de load_tile de todos os tiles),
      que e' o piso teorico do CCT.

A comparacao de tokens usa um terceiro modelo "bf16-arredondado em fp32",
que e' matematicamente o que o CCT representa (pesos BF16, contas em fp32).

Uso:
    python cct/benchmarks/generation.py
    python cct/benchmarks/generation.py --tokens 16 --layers 2
    python cct/benchmarks/generation.py --checkpoint modelos/chat_en.pt
"""

import argparse
import copy
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
sys.path.insert(0, _PKG)
sys.path.insert(0, _ROOT)

from cct.integrate import cctify_model, cct_stats           # noqa: E402
from cct.training import CCTLinear                          # noqa: E402


def build_model(args):
    from minigpt.model import TinyGPT
    torch.manual_seed(0)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = TinyGPT(
            len(ckpt["tokenizer"].get("tokens", [])) or ckpt["model"]["tok_emb.weight"].shape[0],
            block_size=cfg["block_size"], n_embd=cfg["n_embd"],
            n_head=cfg["n_head"], n_layer=cfg["n_layer"],
            tie_embeddings=cfg.get("tie_embeddings", False),
            pos_encoding=cfg.get("pos_encoding", "learned"),
        )
        model.load_state_dict(ckpt["model"])
        return model.eval(), f"checkpoint {args.checkpoint}"
    model = TinyGPT(259, block_size=args.block, n_embd=args.embd,
                    n_head=args.heads, n_layer=args.layers, tie_embeddings=True)
    return model.eval(), f"sintetico {args.embd}/{args.heads}/{args.layers}"


def bf16_roundtrip(model):
    """Copia identica com pesos arredondados para BF16 (e de volta a fp32)."""
    ref = copy.deepcopy(model).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(p.detach().to(torch.bfloat16).float())
    return ref


@torch.no_grad()
def timed_generate(model, prompt, n_tokens, reps=3):
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = model.generate(prompt, max_new_tokens=n_tokens, temperature=0)
        best = min(best, time.perf_counter() - t0)
    return out, best


@torch.no_grad()
def decode_only_ms(model, reps=3):
    """Piso teorico: decodificar todos os tiles uma vez (1 token)."""
    layers = [m for m in model.modules() if isinstance(m, CCTLinear)]
    if not layers:
        return 0.0
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for m in layers:
            for tid in range(m.cct.n_tiles):
                m.cct.load_tile(tid)
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--embd", type=int, default=192)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--codec", default="l3_shuffle")
    ap.add_argument("--tile-rows", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    model, source = build_model(args)
    print(f"CCT — geracao autorregressiva (batch=1, KV cache)")
    print(f"  modelo: {source} | params={model.num_params/1e6:.2f}M | "
          f"tokens={args.tokens}")

    ref = bf16_roundtrip(model)
    prompt = torch.tensor([[1] + [ord(c) for c in "Hi!"]], dtype=torch.long)

    out_dense, t_dense = timed_generate(model, prompt, args.tokens, args.reps)
    out_ref, t_ref = timed_generate(ref, prompt, args.tokens, args.reps)

    info = cctify_model(model, codec=args.codec, tile_rows=args.tile_rows,
                        min_features=32)
    model.eval()
    out_cct, t_cct = timed_generate(model, prompt, args.tokens, args.reps)
    dec_ms = decode_only_ms(model, args.reps)

    agree_ref = bool(torch.equal(out_ref, out_cct))
    agree_dense = bool(torch.equal(out_dense, out_cct))
    p = lambda s: 1000.0 * s / args.tokens

    print(f"\n  camadas CCT: {info['layers']} | CR={info['CR']:.3f}x | "
          f"raw={info['raw_bytes']/1e6:.2f}MB -> comp={info['compressed_bytes']/1e6:.2f}MB | "
          f"encode={info['encode_ms']:.0f}ms")
    print()
    print("| variante | ms/token | tokens/s | vs denso |")
    print("|---|---:|---:|---|")
    print(f"| denso fp32 | {p(t_dense):.2f} | {args.tokens/t_dense:.2f} | 1.00x |")
    print(f"| denso (pesos bf16, contas fp32) | {p(t_ref):.2f} | "
          f"{args.tokens/t_ref:.2f} | {t_ref/t_dense:.2f}x |")
    print(f"| CCT-{args.codec} | {p(t_cct):.2f} | {args.tokens/t_cct:.2f} | "
          f"{t_cct/t_dense:.2f}x |")
    print()
    print(f"  decode puro por token (piso): {dec_ms:.1f} ms "
          f"({100*dec_ms/(p(t_cct)+1e-9):.0f}% do tempo CCT)")
    print(f"  tokens iguais vs bf16-arredondado: {agree_ref} "
          f"(vs fp32: {agree_dense})")
    print()
    print("Leitura: em CPU pequena o CCT perde em tempo — cada token paga o "
          "decode de\ntodos os pesos. Ele ganha em memoria; ganharia tempo so' "
          "se o workload fosse\nmemory-bound (modelo grande / banda de memoria "
          "como gargalo).")

    # Detalhe por camada (mostra que o custo e' proporcional aos bytes)
    print("\n  por camada:")
    for name, m in model.named_modules():
        if isinstance(m, CCTLinear):
            print(f"    {name:22s} {m.in_features:5d}->{m.out_features:5d} "
                  f"CR={m.compression_ratio:.3f}x tiles={m.cct.n_tiles}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
