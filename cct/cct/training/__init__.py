"""Fase 4 do Compress.md (§15): treino com encode-on-update + metrica ECD (§25).

Honestidade do que isto e' (e do que nao e'):

  - `CCTLinear.forward` em modo TREINO usa o peso em bf16 "decodificado"
    (nn.Parameter), porque o AdamW precisa do gradiente e do estado do
    otimizador. Ou seja: durante o treino NAO ha economia de memoria real —
    seria preciso guardar so' o comprimido, o que conflita com o estado do
    otimizador (e' exatamente o trade-off que o LCT mede).
  - O que a harness mede, como o §15 pede: tempo de encode, tempo de step
    com e sem recompressao, tamanho cru vs comprimido e o CR resultante.
  - Em modo INFERENCIA, `forward` pode consumir o estado comprimido
    (cct_matmul), sem materializar o peso inteiro.

Metrica ECD (§25) — corrigida para nao ser circular:

    ECD_mem  = ops uteis / bytes movidos          (densidade de memoria)
    ECD_time = ops uteis / tempo total            (vazao efetiva)

`ECD_mem` sozinho e' proporcional a 1/CR — reportar "ECD 4.0 -> 6.1" com o
tempo triplicando era comparar a mesma coisa duas vezes. A decisao depende
de `ECD_time` (ou do tempo total), que inclui o decode. Ver `ecd_report`.
"""

import time

import torch
import torch.nn.functional as F

from ..ops import cct_matmul
from ..tensor import CCTTensor


def ecd(flops: float, bytes_moved: float) -> float:
    """ECD de memoria — operacoes uteis por byte movido.

    Atencao: com `bytes_moved = bytes_crus / CR`, este numero e' apenas
    `ops / bytes_crus * CR`; ele NAO incorpora o custo do decode. Use
    `ecd_report` para a comparacao que decide (ECD_time).
    """
    return float(flops) / max(1.0, float(bytes_moved))


def ecd_time(flops: float, seconds: float) -> float:
    """ECD de tempo — operacoes uteis por segundo (vazao efetiva)."""
    return float(flops) / max(1e-12, float(seconds))


def ecd_report(flops: float, bytes_raw: float, bytes_compressed: float,
               seconds_decode: float, seconds_compute: float) -> dict:
    """Compara denso vs comprimido incluindo o custo do decode.

    `seconds_compute` e' o tempo de GEMM de referencia (dense).
    """
    t_dense = max(1e-12, float(seconds_compute))
    t_cct = max(1e-12, float(seconds_decode) + float(seconds_compute))
    return {
        "ecd_mem_dense": ecd(flops, bytes_raw),
        "ecd_mem_cct": ecd(flops, bytes_compressed),
        "ecd_time_dense": ecd_time(flops, t_dense),
        "ecd_time_cct": ecd_time(flops, t_cct),
        "decode_s": float(seconds_decode),
        "compute_s": float(seconds_compute),
        "overhead_pct": 100.0 * (t_cct / t_dense - 1.0),
        "cct_slower": t_cct > t_dense,
    }


class CCTLinear(torch.nn.Module):
    """Linear cujo peso e' re-comprimido depois de cada `optimizer.step()`."""

    def __init__(self, in_features, out_features, bias=True,
                 codec="l3_shuffle", tile_rows=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.codec = codec
        self.tile_rows = tile_rows
        w = torch.empty(out_features, in_features)
        torch.nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.weight = torch.nn.Parameter(w)
        if bias:
            b = torch.empty(out_features)
            bound = 1 / (in_features ** 0.5)
            torch.nn.init.uniform_(b, -bound, bound)
            self.bias = torch.nn.Parameter(b)
        else:
            self.register_parameter("bias", None)
        self.cct = None
        self.encode_ms = 0.0
        self.raw_bytes = 0
        self.compressed_bytes = 0

    @classmethod
    def from_dense(cls, linear, codec="l3_shuffle", tile_rows=64):
        """Substitui um `torch.nn.Linear` por um Linear que consome CCT.

        Exemplo (o padrao de adocao que faltava no README):
            # antes
            self.fc = nn.Linear(192, 768)
            # depois
            self.fc = CCTLinear.from_dense(nn.Linear(192, 768))
        """
        m = cls(linear.in_features, linear.out_features,
                bias=linear.bias is not None, codec=codec, tile_rows=tile_rows)
        with torch.no_grad():
            m.weight.copy_(linear.weight.detach())
            if linear.bias is not None:
                m.bias.copy_(linear.bias.detach())
        m.refresh()          # ja nasce pronto para inferencia
        return m

    # ---- Fase 4 ----
    def refresh(self):
        """Recomprime o peso (chame depois de optimizer.step())."""
        t0 = time.perf_counter()
        self.cct = CCTTensor.from_tensor(
            self.weight.detach().to(torch.bfloat16), codec=self.codec,
            tile_rows=self.tile_rows)
        self.encode_ms = (time.perf_counter() - t0) * 1000.0
        self.raw_bytes = self.cct.raw_bytes
        self.compressed_bytes = self.cct.compressed_bytes
        return self

    @property
    def compression_ratio(self):
        return self.raw_bytes / max(1, self.compressed_bytes)

    def forward(self, x):
        if self.training or self.cct is None:
            # treino: peso "decodificado" (gradiente flui normalmente)
            return F.linear(x, self.weight, self.bias)
        # inferencia: consome o estado comprimido, tile a tile.
        # x pode ser (..., in_features); cct_matmul trabalha em 2-D.
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        y = cct_matmul(self.cct, x2.t().contiguous()).t()
        y = y.reshape(*lead, self.out_features)
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"codec={self.codec}, CR={self.compression_ratio:.3f}x")


def measure_encode_on_update(module, opt, x, y, steps=20, refresh_every=1,
                             loss_fn=None):
    """Compara step denso vs step com recompressao (o trade-off do §15).

    -> dict com: step_ms_denso, step_ms_cct, overhead_pct, encode_ms_medio,
       CR, raw_bytes, compressed_bytes, final_loss
    """
    loss_fn = loss_fn or F.mse_loss
    cct_layers = [m for m in module.modules() if isinstance(m, CCTLinear)]
    if not cct_layers:
        raise ValueError("modulo sem CCTLinear")

    def one_step(do_refresh, i):
        opt.zero_grad(set_to_none=True)
        out = module(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        if do_refresh and (i % refresh_every == 0):
            for m in cct_layers:
                m.refresh()
        return float(loss.detach())

    # aquecimento
    for i in range(2):
        one_step(False, i)

    t0 = time.perf_counter()
    for i in range(steps):
        one_step(False, i)
    step_denso = (time.perf_counter() - t0) / steps * 1000.0

    t0 = time.perf_counter()
    loss = 0.0
    for i in range(steps):
        loss = one_step(True, i)
    step_cct = (time.perf_counter() - t0) / steps * 1000.0

    return {
        "step_ms_denso": step_denso,
        "step_ms_cct": step_cct,
        "overhead_pct": 100.0 * (step_cct / max(step_denso, 1e-9) - 1.0),
        "encode_ms_medio": sum(m.encode_ms for m in cct_layers) / len(cct_layers),
        "CR": sum(m.compression_ratio for m in cct_layers) / len(cct_layers),
        "raw_bytes": sum(m.raw_bytes for m in cct_layers),
        "compressed_bytes": sum(m.compressed_bytes for m in cct_layers),
        "final_loss": loss,
    }


class MLPBlock(torch.nn.Module):
    """Bloco minimo para a harness: Linear -> GELU -> Linear."""

    def __init__(self, d_in, d_hidden, d_out=None, **kw):
        super().__init__()
        d_out = d_out or d_in
        self.fc1 = CCTLinear(d_in, d_hidden, **kw)
        self.fc2 = CCTLinear(d_hidden, d_out, **kw)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
