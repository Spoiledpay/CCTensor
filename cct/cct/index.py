"""TileIndex — indice de acesso aleatorio aos tiles comprimidos.

Cada tile e comprimido de forma independente; o indice guarda onde cada um
esta no storage e com que codec foi codificado. E o que permite ao
"Tensor Core" pedir SO o tile 17 em vez de descomprimir a matriz inteira.
"""

import struct
from dataclasses import dataclass, field


@dataclass
class TileEntry:
    tile_id: int
    offset: int          # posicao no storage comprimido
    compressed_size: int
    raw_size: int
    codec: str


@dataclass
class TileIndex:
    entries: list = field(default_factory=list)
    tile_size: int = 0
    shape: tuple = ()
    dtype: str = "bfloat16"
    tile_cols: int = 0        # colunas por tile (modo row_tiles)
    row_tiles: bool = False   # True = cada tile e um bloco de linhas

    def add(self, entry: TileEntry):
        self.entries.append(entry)

    def get(self, tile_id: int) -> TileEntry:
        return self.entries[tile_id]

    @property
    def n_tiles(self) -> int:
        return len(self.entries)

    @property
    def compressed_bytes(self) -> int:
        return sum(e.compressed_size for e in self.entries)

    @property
    def raw_bytes(self) -> int:
        return sum(e.raw_size for e in self.entries)

    @property
    def compression_ratio(self) -> float:
        raw = self.raw_bytes
        return (raw / self.compressed_bytes) if self.compressed_bytes else 1.0

    # ---------- serializacao ----------
    MAGIC = b"CCTI"

    def to_bytes(self) -> bytes:
        out = bytearray(self.MAGIC)
        out += struct.pack("<III", self.tile_size, len(self.entries), len(self.shape))
        out += struct.pack("<" + "I" * len(self.shape), *self.shape)
        out += self.dtype.encode("utf-8") + b"\x00"
        out += struct.pack("<I", 1 if getattr(self, "row_tiles", False) else 0)
        out += struct.pack("<I", int(getattr(self, "tile_cols", 0) or 0))
        for e in self.entries:
            out += struct.pack("<IIIII", e.tile_id, e.offset, e.compressed_size,
                               e.raw_size, len(e.codec))
            out += e.codec.encode("utf-8")
        return bytes(out)

    @classmethod
    def from_bytes(cls, blob: bytes):
        assert blob[:4] == cls.MAGIC, "indice CCT invalido"
        off = 4
        tile_size, n_entries, ndim = struct.unpack_from("<III", blob, off)
        off += 12
        shape = struct.unpack_from("<" + "I" * ndim, blob, off)
        off += 4 * ndim
        end = blob.index(b"\x00", off)
        dtype = blob[off:end].decode("utf-8")
        off = end + 1
        row_tiles, tile_cols = struct.unpack_from("<II", blob, off)
        off += 8
        entries = []
        for _ in range(n_entries):
            tid, o, csz, rsz, clen = struct.unpack_from("<IIIII", blob, off)
            off += 20
            codec = blob[off:off + clen].decode("utf-8")
            off += clen
            entries.append(TileEntry(tid, o, csz, rsz, codec))
        idx = cls(entries=entries, tile_size=tile_size, shape=tuple(shape), dtype=dtype)
        idx.row_tiles = bool(row_tiles)
        idx.tile_cols = int(tile_cols)
        return idx
