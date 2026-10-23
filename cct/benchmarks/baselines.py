"""Baselines externos para o CCT-L3 (o comparativo que faltava no README).

O ponto do `Projeto.md` §2.3: comparar o L3 so' com zstd puro e' comparar
contra um adversario fraco. O padrao de fato para floats em HDF5/Zarr e'
**blosc + shuffle/bitshuffle + lz4/zstd**. Este script mede:

    zstd (niveis 3/12/19)
    zstd + dicionario treinado em held-out
    lz4 (frame)
    blosc2 x {lz4, zstd} x {shuffle, bitshuffle}
    fpzip (se instalado)

Metricas por metodo: CR, encode (ms/MB), decode (ms/MB), decode GB/s.
Todo decode e' verificado bit-exact antes de entrar na tabela.

Uso:
    python cct/benchmarks/baselines.py
    python cct/benchmarks/baselines.py --checkpoint modelos/chat_en.pt
    python cct/benchmarks/baselines.py --shape 2048 1024 --reps 5
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
sys.path.insert(0, _PKG)

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None

try:
    import lz4.frame as lz4frame
except ImportError:  # pragma: no cover
    lz4frame = None

try:
    import blosc2
except ImportError:  # pragma: no cover
    blosc2 = None


# --------------------------------------------------------------------------
# adaptadores: cada um devolve (compress_fn, decompress_fn, nota)
# --------------------------------------------------------------------------
def _mk_zstd(level):
    if zstd is None:
        return None
    c = zstd.ZstdCompressor(level=level)
    d = zstd.ZstdDecompressor()

    def comp(raw):
        return c.compress(raw)

    def dec(blob):
        return d.decompress(blob)

    return comp, dec, f"zstd level {level}"


def _mk_lz4():
    if lz4frame is None:
        return None

    def comp(raw):
        return lz4frame.compress(raw)

    def dec(blob):
        return lz4frame.decompress(blob)

    return comp, dec, "lz4 frame"


def _mk_blosc(codec, filt):
    if blosc2 is None:
        return None
    codec_enum = {"lz4": blosc2.Codec.LZ4, "zstd": blosc2.Codec.ZSTD,
                  "blosclz": blosc2.Codec.BLOSCLZ}[codec]
    filt_enum = {"shuffle": blosc2.Filter.SHUFFLE,
                 "bitshuffle": blosc2.Filter.BITSHUFFLE,
                 "nofilter": blosc2.Filter.NOFILTER}[filt]

    def comp(raw):
        return blosc2.compress(raw, typesize=2, clevel=9,
                               codec=codec_enum, filter=filt_enum)

    def dec(blob):
        return blosc2.decompress(blob)

    return comp, dec, f"blosc2 {codec}+{filt}"


def _mk_fpzip():
    try:
        import fpzip  # noqa: F401
    except ImportError:
        return None

    def comp(raw):
        a = np.frombuffer(raw, dtype="<u2")
        return fpzip.compress(a, precision=0, header=True).tobytes()

    def dec(blob):
        a = fpzip.decompress(blob)
        return np.ascontiguousarray(a).astype("<u2").tobytes()

    return comp, dec, "fpzip (lossless int16 view)"


def baseline_factories():
    """[(nome, factory)] — cada factory pode devolver None se indisponivel."""
    out = [
        ("zstd-19", lambda: _mk_zstd(19)),
        ("zstd-12", lambda: _mk_zstd(12)),
        ("zstd-3", lambda: _mk_zstd(3)),
        ("lz4", _mk_lz4),
        ("blosc-lz4-shuffle", lambda: _mk_blosc("lz4", "shuffle")),
        ("blosc-lz4-bitshuffle", lambda: _mk_blosc("lz4", "bitshuffle")),
        ("blosc-zstd-shuffle", lambda: _mk_blosc("zstd", "shuffle")),
        ("blosc-zstd-bitshuffle", lambda: _mk_blosc("zstd", "bitshuffle")),
        ("blosc-zstd-nofilter", lambda: _mk_blosc("zstd", "nofilter")),
        ("fpzip", _mk_fpzip),
    ]
    return out


def zstd_dict_row(raw, dict_size=112 * 1024, level=19):
    """CR de zstd com dicionario treinado na METADE 0 e medido na METADE 1."""
    if zstd is None:
        return None
    half = len(raw) // 2
    half -= half % (2 * 4096)
    train, test = raw[:half], raw[half:]
    tile = 4096
    samples = [train[i:i + tile] for i in range(0, len(train), tile)]
    samples = [s for s in samples if len(s) >= 512]
    if len(samples) < 8:
        return None
    try:
        d = zstd.train_dictionary(dict_size, samples)
    except Exception:
        return None
    c = zstd.ZstdCompressor(level=level, dict_data=d)
    dctx = zstd.ZstdDecompressor(dict_data=d)

    t0 = time.perf_counter()
    blob = c.compress(test)
    t1 = time.perf_counter()
    back = dctx.decompress(blob, max_output_size=len(test))
    t2 = time.perf_counter()
    if back != test:
        return None
    return {
        "name": "zstd-19+dict (held-out)",
        "cr": len(test) / max(1, len(blob)),
        "enc_ms_per_mb": 1000.0 * (t1 - t0) / (len(test) / 1e6),
        "dec_ms_per_mb": 1000.0 * (t2 - t1) / (len(test) / 1e6),
        "dec_gbps": (len(test) / (t2 - t1)) / 1e9 if t2 > t1 else 0.0,
        "note": "treinado na metade 0, medido na metade 1",
    }


# --------------------------------------------------------------------------
# medicao
# --------------------------------------------------------------------------
def _timed(fn, arg, reps):
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(arg)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return out, best


def measure(raw, reps=3):
    """-> lista de dicts {name, cr, enc_ms_per_mb, dec_ms_per_mb, dec_gbps, note}."""
    rows = []
    raw_mb = len(raw) / 1e6
    for name, factory in baseline_factories():
        made = factory()
        if made is None:
            rows.append({"name": name, "cr": None, "note": "indisponivel"})
            continue
        comp, dec, note = made
        blob, t_enc = _timed(comp, raw, reps)
        back, t_dec = _timed(dec, blob, reps)
        ok = back == raw
        rows.append({
            "name": name,
            "cr": len(raw) / max(1, len(blob)),
            "enc_ms_per_mb": 1000.0 * t_enc / raw_mb,
            "dec_ms_per_mb": 1000.0 * t_dec / raw_mb,
            "dec_gbps": (len(raw) / t_dec) / 1e9 if t_dec > 0 else 0.0,
            "note": note,
            "bit_exact": ok,
        })
    dr = zstd_dict_row(raw)
    if dr:
        rows.append(dr)
    return rows


def format_table(rows, title=""):
    lines = []
    if title:
        lines.append(f"### {title}")
        lines.append("")
    lines.append("| metodo | CR | encode ms/MB | decode ms/MB | decode GB/s |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        if r.get("cr") is None:
            lines.append(f"| {r['name']} | — | — | — | — |")
            continue
        ex = "" if r.get("bit_exact", True) else " ⚠"
        lines.append(
            f"| {r['name']}{ex} | {r['cr']:.3f}x | {r['enc_ms_per_mb']:.1f} | "
            f"{r['dec_ms_per_mb']:.2f} | {r['dec_gbps']:.2f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# dados
# --------------------------------------------------------------------------
def load_tensors(checkpoint=None, shape=(2048, 1024)):
    """-> [(label, bf16 tensor)]. Usa checkpoint se existir, senao sintetiza."""
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        tensors = [(n, t.detach().to(torch.bfloat16))
                   for n, t in state.items()
                   if hasattr(t, "dim") and t.dim() == 2 and t.numel() >= 4096]
        if tensors:
            return [("checkpoint", t) for _n, t in tensors], os.path.basename(checkpoint)
    torch.manual_seed(0)
    # realista: pesos pos-treino tendem a N(0, 0.02)
    realistic = (torch.randn(*shape) * 0.02).to(torch.bfloat16)
    return [("N(0,0.02) bf16", realistic)], "sintetico"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(_ROOT, "modelos", "chat_en.pt"))
    ap.add_argument("--shape", nargs=2, type=int, default=(2048, 1024))
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    print("CCT — baselines externos (L3 vs blosc/bitshuffle/lz4/zstd)")
    if zstd is None:
        print("  AVISO: python-zstandard ausente (pip install zstandard)")
    if blosc2 is None:
        print("  AVISO: blosc2 ausente (pip install blosc2)")
    if lz4frame is None:
        print("  AVISO: lz4 ausente (pip install lz4)")

    tensors, source = load_tensors(args.checkpoint, tuple(args.shape))
    print(f"  fonte: {source} ({len(tensors)} tensor(es))")

    # Um tensor de referencia: concatena para uma medida estavel.
    raw = b"".join(t.view(torch.uint8).numpy().tobytes() for _l, t in tensors)
    if len(raw) > 32 * 1024 * 1024:
        raw = raw[:32 * 1024 * 1024]
    print(f"  amostra: {len(raw)/1e6:.2f} MB de BF16\n")
    rows = measure(raw, reps=args.reps)
    print(format_table(rows))
    print()
    print("Notas:")
    print("  * `bit_exact` verificado no decode; sem marcador = passou.")
    print("  * zstd+dict e' treinado na metade 0 e medido na metade 1 (held-out).")
    print("  * blosc2 usa clevel=9 fixo; o decode ja e' o do C-Blosc.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
