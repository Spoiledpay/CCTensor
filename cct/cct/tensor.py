"""CCTTensor — tensor com estado matematicamente igual, fisicamente comprimido.

O tensor NUNCA e materializado em BF16 inteiro durante a inferencia:
    - decompress() so existe para verificacao/testes
    - matmul() decodifica tile a tile, sob demanda (CFCS)
"""

import numpy as np
import torch

from .codec import compress_best, get_codec
from .index import TileEntry, TileIndex

_DTYPES = {
    torch.bfloat16: ("bfloat16", 2),
    torch.float16: ("float16", 2),
    torch.float32: ("float32", 4),
    torch.uint8: ("uint8", 1),
    torch.int8: ("int8", 1),
}


class CCTTensor:
    """Estado comprimido de um tensor.

    from_tensor(w)  ->  tiles comprimidos + indice
    load_tile(i)    ->  bytes crus do tile i (decodificacao sob demanda)
    matmul(x)       ->  Y = W @ x decodificando tile a tile
    """

    def __init__(self, shape, dtype, tiles, index, codec_name):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.tiles = tiles            # list[bytes]
        self.index = index
        self.codec_name = codec_name

    # ---------------- construcao ----------------
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, tile_size: int = 4096,
                    codec: str = "l1_entropy", tile_rows: int = None) -> "CCTTensor":
        """Comprime o tensor.

        tile_rows: se dado (e o tensor 2-D), cada tile = um bloco de LINHAS
        (rows, cols), o que torna o CCT-GEMM possivel (acumula tile a tile
        sem materializar W). Se None, tiles sao blocos flat de tile_size
        elementos.
        """
        tensor = tensor.detach().contiguous()
        dt_name, itemsize = _DTYPES[tensor.dtype]
        codec_impl = get_codec(codec) if codec != "hybrid" else None
        hybrid = codec == "hybrid"
        # L3 depende de elementos de 2 bytes (BF16/FP16); fora disso, hybrid.
        req = getattr(codec_impl, "requires_itemsize", None)
        if req is not None and itemsize != req:
            codec_impl, hybrid, codec = None, True, "hybrid"

        tiles = []
        entries = []
        offset = 0
        tid = 0

        def encode(data: bytes, cols=None):
            """-> (payload, codec_name)"""
            if hybrid:
                return compress_best(data)
            return codec_impl.compress(data), codec

        if tile_rows is not None and tensor.dim() == 2:
            rows, cols = tensor.shape
            for r0 in range(0, rows, tile_rows):
                chunk = tensor[r0:r0 + tile_rows]
                data = (chunk.view(torch.uint8).numpy().tobytes()
                        if itemsize > 1 else chunk.numpy().tobytes())
                comp, cname = encode(data, cols)
                tiles.append(comp)
                entries.append(TileEntry(tid, offset, len(comp), len(data), cname))
                offset += len(comp)
                tid += 1
            index = TileIndex(entries=entries, tile_size=tile_rows,
                              shape=tuple(tensor.shape), dtype=dt_name)
            index.tile_cols = cols
            index.row_tiles = True
            return cls(tensor.shape, tensor.dtype, tiles, index, codec)

        flat = tensor.reshape(-1)
        n = flat.numel()
        for start in range(0, n, tile_size):
            chunk = flat[start:start + tile_size]
            data = (chunk.view(torch.uint8).numpy().tobytes()
                    if itemsize > 1 else chunk.numpy().tobytes())
            comp, cname = encode(data)
            tiles.append(comp)
            entries.append(TileEntry(tid, offset, len(comp), len(data), cname))
            offset += len(comp)
            tid += 1

        index = TileIndex(entries=entries, tile_size=tile_size,
                          shape=tuple(tensor.shape), dtype=dt_name)
        index.row_tiles = False
        return cls(tensor.shape, tensor.dtype, tiles, index, codec)

    # ---------------- leitura ----------------
    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    @property
    def raw_bytes(self) -> int:
        return self.index.raw_bytes

    @property
    def compressed_bytes(self) -> int:
        return sum(len(t) for t in self.tiles)

    @property
    def compression_ratio(self) -> float:
        return self.raw_bytes / max(1, self.compressed_bytes)

    def load_tile(self, tile_id: int) -> bytes:
        """Decodifica SO o tile pedido (random access)."""
        codec = get_codec(self.index.get(tile_id).codec)
        entry = self.index.get(tile_id)
        return codec.decompress(self.tiles[tile_id], entry.raw_size)

    def load_tile_tensor(self, tile_id: int) -> torch.Tensor:
        data = self.load_tile(tile_id)
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32, "uint8": torch.uint8,
              "int8": torch.int8}[self.index.dtype]
        n = len(data) // torch.tensor([], dtype=dt).element_size()
        t = torch.frombuffer(bytearray(data), dtype=dt)[:n]
        if getattr(self.index, "row_tiles", False):
            t = t.view(-1, self.index.tile_cols)
        return t

    def decompress(self) -> torch.Tensor:
        """Materializa o tensor INTEIRO (verificacao/testes — nao e o caminho
        de inferencia do CCT)."""
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32, "uint8": torch.uint8,
              "int8": torch.int8}[self.index.dtype]
        out = torch.empty(self.shape, dtype=dt)
        flat = out.reshape(-1)
        pos = 0
        for tid in range(self.n_tiles):
            t = self.load_tile_tensor(tid).reshape(-1)
            flat[pos:pos + t.numel()] = t
            pos += t.numel()
        return out

    # ---------------- serializacao ----------------
    # layout: [indice][tiles concatenados]
    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.index.to_bytes())
            f.write(b"".join(self.tiles))

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            blob = f.read()
        index = TileIndex.from_bytes(blob)
        storage_start = len(index.to_bytes())
        tiles = []
        for e in index.entries:
            o = storage_start + e.offset
            tiles.append(blob[o:o + e.compressed_size])
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32, "uint8": torch.uint8,
              "int8": torch.int8}[index.dtype]
        return cls(index.shape, dt, tiles, index,
                   index.entries[0].codec if index.entries else "l1_entropy")

    def __repr__(self):
        return (f"CCTTensor(shape={self.shape}, dtype={self.dtype}, "
                f"tiles={self.n_tiles}, raw={self.raw_bytes/1024:.1f}KB, "
                f"compressed={self.compressed_bytes/1024:.1f}KB, "
                f"CR={self.compression_ratio:.3f}x)")
