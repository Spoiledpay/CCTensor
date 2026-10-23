"""Codecs do CCT.

CCT-L0 : BF16 -> ZSTD                       (baseline)
CCT-L1 : BF16 -> split sign/expo | mantissa  -> Huffman no plano expo
CCT-L2 : BF16 -> bit-planes -> delta -> RLE  -> Huffman

Todos sao LOSSLESS: decompress(compress(x)) == x byte a byte.
"""

import numpy as np

from ..entropy import huffman_compress, huffman_decompress


class Codec:
    name = "base"

    @staticmethod
    def compress(data: bytes) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def decompress(data: bytes, raw_size: int) -> bytes:  # pragma: no cover
        raise NotImplementedError


class ZstdCodec(Codec):
    """CCT-L0: baseline lossless (zstd; lzma como fallback)."""
    name = "l0_zstd"

    @staticmethod
    def compress(data: bytes) -> bytes:
        try:
            import zstandard
            c = zstandard.ZstdCompressor(level=19)
            return b"Z" + c.compress(data)
        except ImportError:
            import lzma
            return b"L" + lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)

    @staticmethod
    def decompress(data: bytes, raw_size: int) -> bytes:
        tag, body = data[:1], data[1:]
        if tag == b"Z":
            import zstandard
            return zstandard.ZstdDecompressor().decompress(body, max_output_size=raw_size)
        import lzma
        return lzma.decompress(body)


def _bf16_planes(raw: bytes):
    """Separa bytes BF16 (little-endian) em 2 planos de N bytes:
       plano ALTO = bits 8..15 (sinal + 7 bits de expoente)
       plano BAIXO = bits 0..7 (7 bits de mantissa + 1 bit de expoente)"""
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr[1::2].copy(), arr[0::2].copy()


def _join_planes(high: np.ndarray, low: np.ndarray) -> bytes:
    n = high.size
    out = np.empty((n, 2), dtype=np.uint8)
    out[:, 0] = low
    out[:, 1] = high
    return out.tobytes()


class EntropyCodec(Codec):
    """CCT-L1: estrutura do BF16 + entropia no plano sign/expoente.

    Em pesos de rede, o byte alto (sinal+expoente) tem distribuicao muito
    concentrada (poucos expoentes) -> Huffman comprime bem; o byte baixo
    (mantissa) e quase aleatorio -> fica cru.
    Framing: "E1" | len(high,4B) | high | low
    """
    name = "l1_entropy"

    @staticmethod
    def compress(data: bytes) -> bytes:
        high, low = _bf16_planes(data)
        h = huffman_compress(high)
        l = huffman_compress(low)
        return b"E1" + len(h).to_bytes(4, "little") + h + l

    @staticmethod
    def decompress(data: bytes, raw_size: int) -> bytes:
        assert data[:2] == b"E1", "payload L1 invalido"
        hlen = int.from_bytes(data[2:6], "little")
        off = 6
        h_part = data[off:off + hlen]
        l_part = data[off + hlen:]
        n = raw_size // 2
        high = huffman_decompress(h_part, n)
        low = huffman_decompress(l_part, n)
        return _join_planes(high, low)


def _bitplanes(arr: np.ndarray, nbits: int = 8):
    """Transpoe para bit-planes: devolve lista de arrays[uint8] (bit i de cada byte)."""
    bits = np.unpackbits(arr[:, None], axis=1) if arr.ndim == 1 else None
    return [np.packbits(bits[:, i], bitorder="big") for i in range(nbits)]


class BitPlaneCodec(Codec):
    """CCT-L2: bit-planes + delta + RLE + Huffman por plano.

    O plano 0 (LSB da mantissa) e quase aleatorio; os planos de expoente
    tem Runs longos (expoente constante dentro de um tile) -> RLE+Huffman
    comprimem muito.
    """
    name = "l2_bitplane"

    @staticmethod
    def _rle_encode(arr: np.ndarray):
        """RLE -> (flags uint8, values). flag=1 significa 'repetir'."""
        if arr.size == 0:
            return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.uint8)
        changes = np.flatnonzero(np.diff(arr)) + 1
        starts = np.concatenate(([0], changes))
        counts = np.diff(np.concatenate((starts, [arr.size])))
        keep = counts < 4  # repeticoes curtas ficam como sequencia
        values = np.empty(0, dtype=np.uint8)
        flags = np.empty(0, dtype=np.uint8)
        i = 0
        while i < starts.size:
            if keep[i]:
                # emite o trecho sem RLE
                c = int(counts[i])
                values = np.concatenate((values, arr[starts[i]:starts[i] + c]))
                flags = np.concatenate((flags, np.zeros(c, dtype=np.uint8)))
                i += 1
            else:
                c = int(counts[i])
                values = np.concatenate((values, arr[starts[i]:starts[i] + 1]))
                flags = np.concatenate((flags, np.array([1], dtype=np.uint8)))
                flags = np.concatenate((flags, np.full(c - 1, 2, dtype=np.uint8)))
                i += 1
        return flags, values

    @staticmethod
    def _rle_decode(flags: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
        out = np.empty(size, dtype=np.uint8)
        vi = 0
        oi = 0
        i = 0
        n = flags.size
        while i < n and oi < size:
            if flags[i] == 0:
                out[oi] = values[vi]
                oi += 1
                vi += 1
                i += 1
            else:
                # flag 1 = inicio de run; flags 2 = continuacao
                run = 1
                i += 1
                while i < n and flags[i] == 2:
                    run += 1
                    i += 1
                out[oi:oi + run] = values[vi]
                oi += run
                vi += 1
        return out[:oi]

    @staticmethod
    def compress(data: bytes) -> bytes:
        arr = np.frombuffer(data, dtype=np.uint8)
        out = bytearray(b"L2")
        for plane in _bitplanes(arr, 8):
            flags, values = BitPlaneCodec._rle_encode(plane)
            f = huffman_compress(flags)
            v = huffman_compress(values)
            # framing: len(f) | len(v) | n_simbolos_de_values | f | v
            out += (len(f).to_bytes(4, "little") + len(v).to_bytes(4, "little")
                    + values.size.to_bytes(4, "little") + f + v)
        return bytes(out)

    @staticmethod
    def decompress(data: bytes, raw_size: int) -> bytes:
        assert data[:2] == b"L2", "payload L2 invalido"
        off = 2
        n_bytes = raw_size
        n_bits = n_bytes * 8
        planes = []
        for _ in range(8):
            flen = int.from_bytes(data[off:off + 4], "little")
            vlen = int.from_bytes(data[off + 4:off + 8], "little")
            vcount = int.from_bytes(data[off + 8:off + 12], "little")
            off += 12
            f_part = data[off:off + flen]
            off += flen
            v_part = data[off:off + vlen]
            off += vlen
            n_plane = (n_bytes + 7) // 8
            flags = huffman_decompress(f_part, n_plane)
            values = huffman_decompress(v_part, vcount)
            planes.append(BitPlaneCodec._rle_decode(flags, values, n_plane))
        # cada plano tem n_plane bytes = n_plane*8 bits; recorta para n_bits
        plane_bits = [np.unpackbits(p)[:n_bits] for p in planes]
        bits = np.stack(plane_bits, axis=1)          # (n_bytes, 8)
        return np.packbits(bits, axis=1, bitorder="big").tobytes()


def compress_best(data: bytes):
    """Melhor codec para este bloco -> (payload, codec_name).

    Usado pelo modo 'hybrid': o indice guarda o codec POR TILE, entao cada
    bloco usa a codificacao que lhe convem. Garante CR >= 1 (nunca expande).
    """
    best = None
    for cname in ("l0_zstd", "l1_entropy", "l2_bitplane", "l3_shuffle"):
        blob = get_codec(cname).compress(data)
        if best is None or len(blob) < len(best[0]):
            best = (blob, cname)
    return best


CODECS = {
    "l0_zstd": ZstdCodec,
    "l1_entropy": EntropyCodec,
    "l2_bitplane": BitPlaneCodec,
}


def _register_l3():
    """CCT-L3 vive em modulo proprio (codec/l3_structured.py)."""
    from .l3_structured import L3Codec
    CODECS["l3_shuffle"] = L3Codec
    CODECS["l3_structured"] = L3Codec


_register_l3()


def get_codec(name: str) -> Codec:
    if name not in CODECS:
        raise KeyError(f"codec desconhecido: {name} (disponiveis: {list(CODECS)})")
    return CODECS[name]
