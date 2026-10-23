"""CCT-GEMM: Y = W @ x consumindo o estado COMPRIMIDO de W.

O caminho prohibited (Compress.md §9):
    compressed -> decompress -> matriz BF16 inteira -> GEMM
Aqui fazemos:
    compressed tile -> decode -> registers -> multiplicacao -> acumulacao

O peso W e tratado como constante (o gradiente vai para o estado
descomprimido, que o optimizer atualiza; depois re-comprimimos =
encode-on-update, Fase 4).
"""

import torch

from .tensor import CCTTensor


class _CCTMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cct: CCTTensor, x: torch.Tensor):
        rows, cols = cct.shape
        out = torch.empty((rows, x.shape[1]), dtype=torch.float32)
        r = 0
        for tid in range(cct.n_tiles):
            tile = cct.load_tile_tensor(tid).to(torch.float32)   # (rows_tile, cols)
            out[r:r + tile.shape[0]] = tile @ x
            r += tile.shape[0]
        ctx.cct = cct
        ctx.save_for_backward(x)
        return out.to(x.dtype if x.dtype.is_floating_point else torch.float32)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        cct = ctx.cct
        # dL/dx = grad_out @ W  -> decodifica W tile a tile
        rows, cols = cct.shape
        gx = torch.zeros((cols, x.shape[1]), dtype=torch.float32)
        r = 0
        for tid in range(cct.n_tiles):
            tile = cct.load_tile_tensor(tid).to(torch.float32)
            # dL/dx = W^T @ dL/dY   (x e (cols, batch))
            gx += tile.T @ grad_out[r:r + tile.shape[0]].to(torch.float32)
            r += tile.shape[0]
        return None, gx.to(x.dtype)


def cct_matmul(cct: CCTTensor, x: torch.Tensor) -> torch.Tensor:
    """Y = W @ x, com W em estado comprimido (decodifica sob demanda)."""
    return _CCTMatMul.apply(cct, x)


def cct_matmul_streaming(cct: CCTTensor, x: torch.Tensor, cache=None):
    """Variante que usa o cache de tiles (runtime CPU) e mede o tempo."""
    rows, cols = cct.shape
    out = torch.empty((rows, x.shape[1]), dtype=torch.float32)
    r = 0
    for tid in range(cct.n_tiles):
        if cache is not None:
            tile = cache.get(cct, tid)
        else:
            tile = cct.load_tile_tensor(tid)
        out[r:r + tile.shape[0]] = tile.to(torch.float32) @ x
        r += tile.shape[0]
    return out.to(x.dtype)
