"""Aceleracao nativa (C) do decode do CCT-L3.

O codec L3 precisa de:
    1) zstd decompress dos fluxos de sinal e expoente  (ja em C, via
       python-zstandard);
    2) remontar BF16 a partir de 3 fluxos de bits         (aqui).

O passo (2) e' o gargalo em numpy (~100 ms para 4 MB de pesos). Esta
extensao faz isso perto da largura de banda de memoria.

Build:
    python -c "import cct.native as n; n.build()"
    # ou
    python cct/tools/build_native.py

Se a biblioteca nao existir (ou falhar a auto-verificacao), `available()`
retorna False e o codec usa o caminho numpy — o resultado e' bit-exact nos
dois casos.
"""

import ctypes
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "l3_native.c")
_ABI = 4

_LIB = None
_AVAILABLE = False
_ERROR = None
_LOADED = False


def _candidates():
    if sys.platform == "win32":
        names = ["l3_native.dll"]
    elif sys.platform == "darwin":
        names = ["l3_native.dylib"]
    else:
        names = ["l3_native.so"]
    return [os.path.join(_HERE, n) for n in names]


def _bind(lib):
    lib.l3_native_abi.restype = ctypes.c_int
    lib.l3_unpack_bf16.restype = ctypes.c_int
    lib.l3_unpack_bf16.argtypes = [
        ctypes.c_void_p, ctypes.c_int64,   # sign bits, len
        ctypes.c_void_p, ctypes.c_int64,   # exp, len
        ctypes.c_void_p, ctypes.c_int64,   # mant bits, len
        ctypes.c_int64,                    # n
        ctypes.c_void_p,                   # out (2n bytes)
    ]


def _load():
    global _LIB, _AVAILABLE, _ERROR
    for path in _candidates():
        if not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            _bind(lib)
            if lib.l3_native_abi() != _ABI:
                _ERROR = f"ABI inesperada em {path}"
                continue
            if _selftest(lib):
                _LIB = lib
                _AVAILABLE = True
                _ERROR = None
                return
            _ERROR = f"auto-verificacao falhou em {path}"
        except OSError as exc:  # pragma: no cover - ambiente
            _ERROR = str(exc)
    _AVAILABLE = False


def _pack_bits(vals: np.ndarray, width: int) -> bytes:
    if width <= 0 or vals.size == 0:
        return b""
    bits = np.unpackbits(vals.astype(np.uint8)[:, None], axis=1)[:, 8 - width:]
    return np.packbits(bits.reshape(-1)).tobytes()


def _numpy_ref(sign: np.ndarray, exp: np.ndarray, mant: np.ndarray) -> bytes:
    u = ((sign.astype(np.uint16) << 15)
         | (exp.astype(np.uint16) << 7)
         | mant.astype(np.uint16))
    return u.astype("<u2").tobytes()


def _call(lib, sign_bits, exp, mant_bits, n) -> bytes:
    out = bytearray(2 * n)
    obuf = (ctypes.c_char * len(out)).from_buffer(out)
    rc = lib.l3_unpack_bf16(
        ctypes.c_char_p(sign_bits), len(sign_bits),
        ctypes.c_char_p(exp), len(exp),
        ctypes.c_char_p(mant_bits), len(mant_bits),
        n, ctypes.byref(obuf),
    )
    if rc != 0:
        raise RuntimeError(f"l3_unpack_bf16 retornou {rc}")
    return bytes(out)


def _selftest(lib) -> bool:
    """Compara a saida nativa com a referencia numpy em dados aleatorios."""
    rng = np.random.default_rng(0)
    for n in (1, 7, 8, 13, 255, 1024):
        sign = rng.integers(0, 2, n).astype(np.uint8)
        exp = rng.integers(0, 256, n).astype(np.uint8)
        mant = rng.integers(0, 128, n).astype(np.uint8)
        got = _call(lib, _pack_bits(sign, 1), exp.tobytes(),
                    _pack_bits(mant, 7), n)
        if got != _numpy_ref(sign, exp, mant):
            return False
    return True


def unpack_bf16(sign_bits: bytes, exp: bytes, mant_bits: bytes, n: int) -> bytes:
    """Remonta n valores BF16 (2n bytes, little-endian).

    sign_bits: n bits empacotados MSB-first
    exp      : n bytes
    mant_bits: 7n bits empacotados MSB-first
    """
    _load_if_needed()
    if not _AVAILABLE:
        raise RuntimeError("biblioteca nativa indisponivel")
    return _call(_LIB, sign_bits, exp, mant_bits, n)


def available() -> bool:
    _load_if_needed()
    return _AVAILABLE


def library_path():
    for p in _candidates():
        if os.path.exists(p):
            return p
    return None


def error():
    _load_if_needed()
    return _ERROR


def _load_if_needed():
    global _LOADED
    if not _LOADED:
        _load()
        _LOADED = True


def build(verbose: bool = True) -> str:
    """Compila a extensao com o primeiro compilador C encontrado no PATH."""
    import shutil

    out = _candidates()[0]
    cc = None
    for name in ("gcc", "clang", "cc"):
        path = shutil.which(name)
        if path:
            cc = path
            break
    if cc is None:
        raise RuntimeError("nenhum compilador C encontrado (gcc/clang/cc)")

    tmp = tempfile.mkdtemp(prefix="cct-native-")
    obj = os.path.join(tmp, "l3_native.o")
    cmd = [cc, "-O3", "-march=native", "-fPIC", "-c", _SRC, "-o", obj]
    if verbose:
        print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    link = [cc, "-O3", "-shared", obj, "-o", out]
    if verbose:
        print("  " + " ".join(link))
    subprocess.run(link, check=True)

    global _LOADED
    _LOADED = False
    _load_if_needed()
    if not _AVAILABLE:
        raise RuntimeError(f"build ok mas carga falhou: {_ERROR}")
    return out


def main():
    print(f"fonte: {_SRC}")
    path = build()
    print(f"biblioteca: {path}")
    print(f"autoteste: {'OK' if available() else 'FALHOU'} ({error()})")


if __name__ == "__main__":
    main()
