"""CCT-L3: de-interleave de campos BF16 + entropia por fluxo.

Por que os codecs anteriores perdiam (medido, nao suposto):

  - L1/L2 codificam o BF16 *interleaved* (byte baixo = mantissa + 1 bit de
    expoente; byte alto = sinal + 7 bits de expoente). A mantissa e' ~aleatoria
    (6.98 de 7 bits) e quebra os matches do LZ; pior, o Huffman estatico por
    tile paga ate' 5 bytes/simbolo de tabela (~28% num tile de 16 linhas).
  - A entropia ordem-0 do expoente e' 6.18 bits e H(exp|vizinho de cima) 5.54
    (held-out). Ou seja: nao existe ganho de 1.3x "puro" pela entropia; o ganho
    real vem de REORGANIZAR os bytes para o compressor enxergar estrutura.

L3 faz o "shuffle" de floats: separa os tres campos em fluxos contiguos e
comprime cada um com o que lhe convem:

    sign     : 1 bit/valor empacotado -> zstd   (redundante -> comprime bem)
    expoente : 1 byte/valor          -> zstd   (poucos valores -> comprime bem)
    mantissa : 7 bits/valor empacotado -> cru   (~aleatoria -> nada a ganhar)

Medido no checkpoint de referencia: CR ~1.50x (33% de economia) contra 1.27x do
zstd no BF16 interleaved. E' tudo vetorizado em numpy + 2 chamadas de zstd por
tile: sem laco Python por simbolo.

Formato do tile:
    b"M3" | n(4) | tail_len(1) | l_sign(4) | l_exp(4) | tail | sign | exp | mant
"""

import threading

import numpy as np

try:
    from .. import native as _native
except Exception:  # pragma: no cover - ambiente sem a extensao
    _native = None

try:
    import zstandard as _zstd
except ImportError:  # pragma: no cover
    _zstd = None

MAGIC = b"M3"
# Nivel 19 vs 12: +1.5% de CR no checkpoint real, decode identico (o nivel
# afeta so' o encode, que e' offline). Custo: encode ~3x mais lento.
ZSTD_LEVEL = 19


def _pack_bits(vals: np.ndarray, width: int) -> bytes:
    """Empacota `vals` (< 2**width) em `width` bits cada, MSB-first."""
    if width <= 0 or vals.size == 0:
        return b""
    bits = np.unpackbits(vals.astype(np.uint8)[:, None], axis=1)[:, 8 - width:]
    return np.packbits(bits.reshape(-1)).tobytes()


def _unpack_bits(data: bytes, n: int, width: int) -> np.ndarray:
    """Inverso exato de _pack_bits."""
    if width <= 0 or n == 0:
        return np.zeros(n, dtype=np.uint8)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[: n * width]
    bits = bits.reshape(n, width).astype(np.uint16)
    weights = (1 << np.arange(width - 1, -1, -1)).astype(np.uint16)
    return (bits * weights).sum(axis=1).astype(np.uint8)


_TLS = threading.local()


def _z(data: bytes) -> bytes:
    """Comprime um fluxo. O compressor e' cacheado POR THREAD (criar um por
    tile custa caro: era o principal overhead do encode do L3) e as
    instancias do zstandard nao sao thread-safe."""
    if _zstd is not None:
        c = getattr(_TLS, "compressor", None)
        if c is None:
            c = _TLS.compressor = _zstd.ZstdCompressor(level=ZSTD_LEVEL)
        return c.compress(data)
    import lzma
    return lzma.compress(data, preset=6)


def _unz(data: bytes) -> bytes:
    if _zstd is not None:
        d = getattr(_TLS, "decompressor", None)
        if d is None:
            d = _TLS.decompressor = _zstd.ZstdDecompressor()
        return d.decompress(data)
    import lzma
    return lzma.decompress(data)


class L3Codec:
    """CCT-L3 — shuffle de campos BF16 + zstd por fluxo."""

    name = "l3_shuffle"
    # so faz sentido com elementos de 2 bytes (BF16/FP16)
    requires_itemsize = 2

    @staticmethod
    def compress(data: bytes) -> bytes:
        n = len(data) // 2
        tail = data[2 * n:]
        u = np.frombuffer(data[: 2 * n], dtype="<u2")
        sign = (u >> 15).astype(np.uint8)
        exp = ((u >> 7) & 0xFF).astype(np.uint8)
        mant = (u & 0x7F).astype(np.uint8)

        s_sign = _z(_pack_bits(sign, 1)) if n else b""
        s_exp = _z(exp.tobytes()) if n else b""
        s_mant = _pack_bits(mant, 7)

        head = (MAGIC + n.to_bytes(4, "little") + len(tail).to_bytes(1, "little")
                + len(s_sign).to_bytes(4, "little")
                + len(s_exp).to_bytes(4, "little"))
        return head + tail + s_sign + s_exp + s_mant

    @staticmethod
    def decompress(data: bytes, raw_size: int) -> bytes:
        assert data[:2] == MAGIC, "payload L3 invalido"
        off = 2
        n = int.from_bytes(data[off:off + 4], "little"); off += 4
        tail_len = data[off]; off += 1
        l_sign = int.from_bytes(data[off:off + 4], "little"); off += 4
        l_exp = int.from_bytes(data[off:off + 4], "little"); off += 4
        tail = data[off:off + tail_len]; off += tail_len

        if n == 0:
            return tail
        sign_bits = _unz(data[off:off + l_sign]); off += l_sign
        exp = _unz(data[off:off + l_exp]); off += l_exp
        mant_bits = data[off:]
        return _assemble(sign_bits, exp, mant_bits, n) + tail


def _assemble(sign_bits: bytes, exp: bytes, mant_bits: bytes, n: int) -> bytes:
    """Remonta n valores BF16 a partir dos 3 fluxos (nativo se disponivel)."""
    if _native is not None:
        try:
            if _native.available():
                return _native.unpack_bf16(sign_bits, exp, mant_bits, n)
        except Exception:  # pragma: no cover - cai no numpy
            pass
    sign = _unpack_bits(sign_bits, n, 1)
    expa = np.frombuffer(exp, dtype=np.uint8)[:n]
    mant = _unpack_bits(mant_bits, n, 7)
    u = ((sign.astype(np.uint16) << 15)
         | (expa.astype(np.uint16) << 7)
         | mant.astype(np.uint16))
    return u.astype("<u2").tobytes()


CODEC = L3Codec
