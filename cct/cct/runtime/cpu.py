"""Runtime CPU: cache de tiles decodificados + prefetch paralelo.

Objetivo (Compress.md §16): o decode tem de acompanhar a largura de banda.
Medimos: decode GB/s, cache hit rate, tempo de GEMM com e sem cache.
"""

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import torch


class TileCache:
    """Cache LRU de tiles decodificados (FIFO/LRU por acesso)."""

    def __init__(self, max_tiles: int = 8):
        self.max_tiles = max_tiles
        self._data = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, cct, tile_id: int) -> torch.Tensor:
        key = (id(cct), tile_id)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
        self.misses += 1
        tile = cct.load_tile_tensor(tile_id)
        with self._lock:
            self._data[key] = tile
            self._data.move_to_end(key)
            while len(self._data) > self.max_tiles:
                self._data.popitem(last=False)
        return tile

    def prefetch(self, cct, tile_ids):
        """Decodifica tiles em paralelo (threads)."""
        with ThreadPoolExecutor(max_workers=min(8, len(tile_ids) or 1)) as ex:
            list(ex.map(lambda tid: self.get(cct, tid), tile_ids))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def clear(self):
        with self._lock:
            self._data.clear()


def measure_decode_throughput(cct, iters: int = 5) -> float:
    """GB/s de dados decodificados (raw bytes / tempo)."""
    import time
    t0 = time.time()
    for _ in range(iters):
        for tid in range(cct.n_tiles):
            cct.load_tile(tid)
    dt = time.time() - t0
    raw = cct.raw_bytes * iters
    return (raw / dt) / 1e9 if dt > 0 else 0.0
