"""Testes do codificador de entropia (Huffman) do CCT.

Cobrem os dois bugs corrigidos em `cct/entropy.py`:

  1. a soma de Kraft era 0.5 (1 bit desperdicado por simbolo);
  2. codigos com mais de 16 bits eram truncados no payload v1, corrompendo
     o decode em distribuicoes enviesadas (ex.: plano de expoente BF16).

Rodar:
    python cct/tests/test_entropy.py
    python -m pytest cct/tests/test_entropy.py -q
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cct.codec import EntropyCodec                       # noqa: E402
from cct.entropy import (_canonical_codes, _huffman_lengths,  # noqa: E402
                         huffman_compress, huffman_decompress)


def _lengths(data):
    counts = np.bincount(data, minlength=256)
    freqs = {i: int(c) for i, c in enumerate(counts) if c > 0}
    return _huffman_lengths(freqs)


def _kraft(lengths):
    return sum(2.0 ** -l for l in lengths.values())


def test_kraft_is_one():
    rng = np.random.default_rng(0)
    for nsym in (2, 3, 4, 16, 24, 256):
        data = rng.integers(0, nsym, 50_000).astype(np.uint8)
        lengths = _lengths(data)
        assert abs(_kraft(lengths) - 1.0) < 1e-9, (nsym, _kraft(lengths))
        codes = _canonical_codes(lengths)
        assert len({c for c, _l in codes.values()}) == len(codes)


def test_roundtrip_all_symbols():
    rng = np.random.default_rng(1)
    data = np.arange(256, dtype=np.uint8)
    data = np.concatenate([data, rng.integers(0, 256, 20_000).astype(np.uint8)])
    blob = huffman_compress(data)
    assert np.array_equal(huffman_decompress(blob, data.size), data)


def test_roundtrip_skewed_and_long_codes():
    """Frequencias de Fibonacci forcam codigos > 16 bits (o bug do v1)."""
    fib = [1, 1]
    while len(fib) < 22:
        fib.append(fib[-1] + fib[-2])
    data = np.concatenate([
        np.full(f, i, dtype=np.uint8) for i, f in enumerate(fib)
    ])
    lengths = _lengths(data)
    assert max(lengths.values()) > 16, max(lengths.values())
    blob = huffman_compress(data)
    assert np.array_equal(huffman_decompress(blob, data.size), data)


def test_skewed_single_symbol_dominates():
    data = np.zeros(200_000, dtype=np.uint8)
    data[:50] = np.arange(50, dtype=np.uint8) + 1
    blob = huffman_compress(data)
    assert np.array_equal(huffman_decompress(blob, data.size), data)
    assert len(blob) < len(data) / 5


def test_roundtrip_empty_and_single():
    empty = np.empty(0, dtype=np.uint8)
    blob = huffman_compress(empty)
    assert huffman_decompress(blob, 0).size == 0
    one = np.full(1000, 7, dtype=np.uint8)
    blob = huffman_compress(one)
    assert np.array_equal(huffman_decompress(blob, one.size), one)


def test_bf16_planes_large():
    """Regressao do bug real: 4 MB de BF16 quebravam o L1 no plano alto."""
    torch.manual_seed(0)
    w = torch.randn(2048, 1024).to(torch.bfloat16)
    raw = w.view(torch.uint8).numpy().tobytes()
    blob = EntropyCodec.compress(raw)
    back = EntropyCodec.decompress(blob, len(raw))
    assert back == raw
    assert len(raw) / len(blob) > 1.3


def test_legacy_payload_still_reads():
    """Payload v1 (1 byte n-1, 5 bytes/simbolo) continua legivel."""
    syms = {0: (0, 1), 1: (1, 1)}          # sym -> (code, len)
    payload = bytearray([1])                # n_syms - 1
    for s, (c, ln) in syms.items():
        payload += ln.to_bytes(2, "big") + c.to_bytes(2, "big") + bytes([s])
    body = bytes([0b01010000, 0b10000000])  # bits: 0,1,0,1,0,0,0,0 | 1,0...
    out = huffman_decompress(bytes(payload) + body, 4)
    assert out.tolist() == [0, 1, 0, 1]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = []
    for fn in tests:
        try:
            fn()
            print(f"  [OK] {fn.__name__}")
        except AssertionError as exc:
            fails.append(fn.__name__)
            print(f"  [XX] {fn.__name__}: {exc}")
    print(f"\n{len(tests) - len(fails)} OK / {len(fails)} falhas")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
