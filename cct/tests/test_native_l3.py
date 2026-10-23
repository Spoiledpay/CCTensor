"""Decoder nativo (C) do L3 vs a referencia numpy.

Se a biblioteca nao estiver compilada, os testes sao pulados (nao falham) —
o codec usa o caminho numpy automaticamente.

Rodar:
    python -c "import cct.native as n; n.build()"
    python cct/tests/test_native_l3.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cct.native as native                          # noqa: E402
import cct.codec.l3_structured as l3                 # noqa: E402
from cct.native import _pack_bits                    # noqa: E402
from cct.tensor import CCTTensor                     # noqa: E402

SIZES = (1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 255, 1000, 4097)


def _ref(sign, exp, mant):
    u = ((sign.astype(np.uint16) << 15)
         | (exp.astype(np.uint16) << 7)
         | mant.astype(np.uint16))
    return u.astype("<u2").tobytes()


def test_direct_against_numpy():
    rng = np.random.default_rng(7)
    for n in SIZES:
        sign = rng.integers(0, 2, n).astype(np.uint8)
        exp = rng.integers(0, 256, n).astype(np.uint8)
        mant = rng.integers(0, 128, n).astype(np.uint8)
        got = native.unpack_bf16(_pack_bits(sign, 1), exp.tobytes(),
                                 _pack_bits(mant, 7), n)
        assert got == _ref(sign, exp, mant), n


def test_l3_codec_matches_numpy_path():
    torch.manual_seed(0)
    shapes = [(1, 1), (1, 7), (3, 333), (17, 65), (64, 64), (8, 1000)]
    saved = l3._native
    try:
        for shape in shapes:
            w = (torch.randn(*shape) * 0.02).to(torch.bfloat16)
            cct = CCTTensor.from_tensor(w, codec="l3_shuffle", tile_rows=64)
            blob = b"".join(cct.tiles)
            entry = cct.index.entries[0]
            assert entry.raw_size == w.numel() * 2

            l3._native = saved
            fast = cct.decompress()
            l3._native = None
            slow = cct.decompress()
            l3._native = saved

            assert torch.equal(fast, slow), shape
            assert torch.equal(fast, w), shape
    finally:
        l3._native = saved


def test_native_bit_exact_on_edge_values():
    """Zeros, denormais, constantes, +-inf e NaN continuam bit-exact."""
    vals = torch.tensor([
        0.0, -0.0, 1.0, -1.0, 0.03125,
        float("inf"), float("-inf"), float("nan"),
        1e-40, -1e-40,
    ], dtype=torch.bfloat16)
    w = vals.repeat(37).reshape(37, len(vals))
    cct = CCTTensor.from_tensor(w, codec="l3_shuffle", tile_rows=8)
    # NaN != NaN; compara pelos bytes
    a = w.view(torch.uint16).numpy().tobytes()
    b = cct.decompress().view(torch.uint16).numpy().tobytes()
    assert a == b


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    if not native.available():
        print(f"  (nativo indisponivel: {native.error()}; pulando)")
        print("   compile com: python -c \"import cct.native as n; n.build()\"")
        return 0
    print(f"  biblioteca: {native.library_path()}")
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
