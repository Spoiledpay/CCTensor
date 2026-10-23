"""Exporta o TinyGPT para ONNX (caminho tipico para deploy em NPU).

    .venv\\Scripts\\python.exe export_onnx.py
    .venv\\Scripts\\python.exe export_onnx.py --int8

Fluxo NPU comum: PyTorch -> ONNX -> (quantizacao INT8) -> runtime da NPU
(OpenVINO/Intel NPU, ONNX Runtime QNN/Qualcomm Hexagon, RKNN/Rockchip, Core ML...).

Observacoes:
  - O grafo exportado cobre o forward de sequencia completa (sem KV cache).
    Para inferencia incremental (token a token), exporte um segundo grafo com
    entradas/saidas de cache, ou use o runtime que faca o split automaticamente.
  - Verifique no runtime da NPU o suporte a MultiHeadAttention, LayerNorm,
    GELU e ao masked_fill da mascara causal.
"""

import argparse

import torch

from minigpt.utils import load_checkpoint


class LogitsOnly(torch.nn.Module):
    """Wrapper para exportar so os logits (sem KV cache/loss)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx):
        logits, _, _ = self.model(idx)
        return logits


def parse_args():
    p = argparse.ArgumentParser(description="Exportacao ONNX do TinyGPT")
    p.add_argument("--checkpoint", type=str, default="modelos/especialista.pt")
    p.add_argument("--out", type=str, default="modelos/especialista.onnx")
    p.add_argument("--seq-len", type=int, default=None, help="tamanho de sequencia do exemplo (default: block_size)")
    p.add_argument("--int8", action="store_true", help="quantiza para INT8 (requer onnxruntime)")
    return p.parse_args()


def export_checkpoint_to_onnx(checkpoint, out, seq_len=None, int8=False):
    """Exporta um checkpoint .pt para .onnx. Reusavel por train*.py."""
    import os

    from minigpt.utils import load_checkpoint

    model, tokenizer, cfg = load_checkpoint(checkpoint)
    model.eval()

    seq_len = seq_len or cfg["block_size"]
    dummy = torch.randint(0, tokenizer.vocab_size, (1, seq_len), dtype=torch.long)

    wrapper = LogitsOnly(model).eval()
    directory = os.path.dirname(out)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        out,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        opset_version=17,
        dynamo=False,
    )

    with torch.no_grad():
        logits = wrapper(dummy)

    if int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError as exc:
            raise SystemExit(
                "Instale onnxruntime para quantizar: pip install onnxruntime"
            ) from exc
        int8_path = out.replace(".onnx", ".int8.onnx")
        quantize_dynamic(out, int8_path, weight_type=QuantType.QInt8)
        print(f"Modelo INT8 salvo em: {int8_path}")
    return out


def main():
    args = parse_args()
    out = export_checkpoint_to_onnx(args.checkpoint, args.out, args.seq_len, args.int8)
    print(f"ONNX salvo em: {out}")


if __name__ == "__main__":
    main()
