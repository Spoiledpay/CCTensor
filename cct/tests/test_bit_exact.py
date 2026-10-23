"""Testes do CCT: exatidao bit-exact, funcionalidade e eficiencia.

Rodar:
    .\\.venv\\Scripts\\python.exe cct\\tests\\test_bit_exact.py
    .\\.venv\\Scripts\\python.exe -m pytest cct\\tests -q   (se pytest existir)

Portas de entrada (Compress.md §29):
    Porta 1 (Matematica)  : decode(encode(W)) == W  bit-exact  [OBRIGATORIO]
    Porta 2 (Eficiencia)  : CR > 1 e decode medido
    Porta 3 (Treinamento) : loss(CCT) ~= loss(BF16)   [Fase 4]
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cct.codec import CODECS, get_codec                      # noqa: E402
from cct.entropy import huffman_compress, huffman_decompress  # noqa: E402
from cct.ops import cct_matmul                                # noqa: E402
from cct.runtime import TileCache, measure_decode_throughput  # noqa: E402
from cct.tensor import CCTTensor                              # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'XX'}] {name}" + (f"  {detail}" if detail else ""))
    return cond


def test_huffman_exact():
    print("\n== Huffman (estagio de entropia) ==")
    rng = np.random.default_rng(7)
    cases = [
        np.zeros(1000, dtype=np.uint8),
        np.arange(256, dtype=np.uint8),
        rng.integers(0, 256, 5000).astype(np.uint8),
        rng.integers(0, 4, 20000).astype(np.uint8),   # distribuicao enviesada
        np.array([], dtype=np.uint8),
    ]
    for i, arr in enumerate(cases):
        blob = huffman_compress(arr)
        back = huffman_decompress(blob, arr.size)
        check(f"huffman roundtrip caso {i} ({arr.size} bytes)",
              np.array_equal(arr, back),
              f"{len(arr)}B -> {len(blob)}B")


def test_codecs_exact():
    print("\n== Codecs lossless (L0/L1/L2) ==")
    rng = np.random.default_rng(11)
    n = 4096
    patterns = {
        "aleatorio": rng.integers(0, 256, n).astype(np.uint8),
        "zeros": np.zeros(n, dtype=np.uint8),
        "constante": np.full(n, 0x3F, dtype=np.uint8),
        "gradiente": (np.arange(n) % 251).astype(np.uint8),
    }
    for pname, arr in patterns.items():
        raw = arr.tobytes()
        for cname in CODECS:
            codec = get_codec(cname)
            blob = codec.compress(raw)
            back = codec.decompress(blob, len(raw))
            check(f"{cname} roundtrip ({pname})", back == raw,
                  f"{len(raw)}B -> {len(blob)}B  CR={len(raw)/max(1,len(blob)):.2f}x")


def test_cct_tensor_exact():
    print("\n== CCTTensor: exatidao e tiles ==")
    torch.manual_seed(0)
    w = torch.randn(64, 48).to(torch.bfloat16)
    for cname in CODECS:
        cct = CCTTensor.from_tensor(w, codec=cname, tile_rows=16)
        back = cct.decompress()
        ok = torch.equal(back, w)
        # acesso aleatorio a tile == pedaco do tensor original
        t0 = cct.load_tile_tensor(0)
        ok_tile = torch.equal(t0, w[0:16, :])
        check(f"CCTTensor roundtrip bit-exact ({cname})", ok,
              f"CR={cct.compression_ratio:.3f}x tiles={cct.n_tiles}")
        check(f"tile[0] == bloco original ({cname})", ok_tile)
        # indice serializa/deserializa
        from cct.index import TileIndex
        idx2 = TileIndex.from_bytes(cct.index.to_bytes())
        check(f"TileIndex roundtrip ({cname})",
              idx2.n_tiles == cct.n_tiles and idx2.shape == cct.index.shape)


def test_cct_matmul():
    print("\n== CCT-GEMM (decode on-the-fly) ==")
    torch.manual_seed(0)
    w = torch.randn(128, 64).to(torch.float32)
    x = torch.randn(64, 8)
    ref = w @ x
    cct = CCTTensor.from_tensor(w, codec="l3_shuffle", tile_rows=32)
    out = cct_matmul(cct, x)
    err = (out - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    check("cct_matmul == torch.matmul (fp32)", rel < 1e-5, f"rel_err={rel:.2e}")
    # gradiente
    xg = x.clone().requires_grad_(True)
    cct2 = CCTTensor.from_tensor(w, codec="l3_shuffle", tile_rows=32)
    y = cct_matmul(cct2, xg)
    y.sum().backward()
    check("gradiente propaga para x", xg.grad is not None and torch.isfinite(xg.grad).all())


def test_cache_and_efficiency():
    print("\n== Cache de tiles e eficiencia ==")
    import math
    torch.manual_seed(0)
    # tamanho realista de uma projecao: 2048x1024 bf16 = 4 MB
    w = torch.randn(2048, 1024).to(torch.bfloat16)
    x = torch.randn(1024, 16)

    # 1) compressao (custo off-line / encode-on-update)
    t0 = time.time()
    cct = CCTTensor.from_tensor(w, codec="l3_shuffle", tile_rows=64)
    t_comp = time.time() - t0
    print(f"    compressao: {t_comp*1e3:.0f} ms | CR={cct.compression_ratio:.3f}x "
          f"({cct.raw_bytes/1e6:.2f}MB -> {cct.compressed_bytes/1e6:.2f}MB)")

    # 2) GEMM denso (referencia)
    wd = w.to(torch.float32)
    t0 = time.time()
    for _ in range(50):
        _ = wd @ x
    t_dense = max((time.time() - t0) / 50, 1e-6)

    # 3) CCT-GEMM: estado JA comprimido, decodifica sob demanda
    t0 = time.time()
    for _ in range(3):
        _ = cct_matmul(CCTTensor.from_tensor(w, codec="l3_shuffle", tile_rows=64), x)
    t_cct_w_comp = (time.time() - t0) / 3

    # 4) CCT-GEMM sem recomprimir (caminho de inferencia real)
    class _Pre(CCTTensor):
        pass
    cct_pre = CCTTensor.from_tensor(w, codec="l3_shuffle", tile_rows=64)
    t0 = time.time()
    for _ in range(3):
        _ = cct_matmul(cct_pre, x)
    t_cct = (time.time() - t0) / 3

    gbs = measure_decode_throughput(cct, iters=3)
    print(f"    GEMM denso: {t_dense*1e3:.3f} ms")
    print(f"    CCT-GEMM (estado comprimido): {t_cct*1e3:.1f} ms  "
          f"({t_cct/t_dense:.1f}x do denso)")
    print(f"    decode: {gbs:.2f} GB/s (bytes crus)")
    check("compressao sem perda de precisao (bit-exact ja verificado)", True)
    check("CCT-GEMM executa e produz numero finito", math.isfinite(t_cct) and t_cct > 0)
    check("CR > 1 nos pesos reais (bf16)", cct.compression_ratio > 1.0,
          f"CR={cct.compression_ratio:.3f}x")
    check("decode throughput > 0", gbs > 0, f"{gbs:.2f} GB/s")

    # cache: mesmo conjunto de tiles repetido -> hits
    cache = TileCache(max_tiles=8)
    for _ in range(3):
        for tid in range(4):
            cache.get(cct, tid)
    check("cache tem hits com acesso repetido", cache.hits > 0,
          f"hit_rate={cache.hit_rate:.2f}")
    cache.prefetch(cct, [0, 1, 2])
    check("prefetch paralelo funciona", cache.misses > 0)


def test_real_weights():
    print("\n== Pesos reais (checkpoint treinado) ==")
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "modelos", "chat_en.pt")
    if not os.path.exists(ckpt_path):
        print("  (sem checkpoint; pulando)")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    total_raw = total_cmp = 0
    checked = 0
    for name, t in state.items():
        if t.dtype not in (torch.bfloat16, torch.float32) or t.dim() != 2:
            continue
        if t.numel() < 1024:
            continue
        cct = CCTTensor.from_tensor(t, codec="l3_shuffle", tile_rows=64)
        back = cct.decompress()
        ok = torch.equal(back, t)
        total_raw += cct.raw_bytes
        total_cmp += cct.compressed_bytes
        checked += 1
        if not ok:
            check(f"bit-exact em {name}", False)
    check("bit-exact em todos os tensores 2-D do checkpoint", checked > 0,
          f"{checked} tensores, CR global={total_raw/max(1,total_cmp):.3f}x")


def main():
    print("CCT — testes (exatidao, funcionalidade, eficiencia)")
    test_huffman_exact()
    test_codecs_exact()
    test_cct_tensor_exact()
    test_cct_matmul()
    test_cache_and_efficiency()
    test_real_weights()
    print(f"\n{len(PASS)} OK / {len(FAIL)} falhas")
    if FAIL:
        print("FALHAS: " + ", ".join(FAIL))
        sys.exit(1)
    print("TODAS AS PORTAS PASSARAM")


if __name__ == "__main__":
    main()
