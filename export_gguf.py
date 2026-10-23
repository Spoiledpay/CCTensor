"""Exportador UNICO de GGUF (Ollama).

Dois caminhos, detetados pelo tipo de entrada:

  1) TinyGPT (.pt deste repo) -> GGUF na arquitetura `gpt2` (llama.cpp).
     O TinyGPT e estruturalmente um GPT-2 (LayerNorm pre-norm, posicao
     aprendida, GELU, causal, head empatada), por isso exportamos nesse
     formato REAL — o Ollama recusa arquiteturas inventadas.
     Chave: os nomes dos tokens vao no byte-encode do GPT-2 (espaco -> 'G'),
     que e o que o llama.cpp aplica antes de tokenizar. Sem isso, o espaco
     vira [UNK_BYTE_0x20] e o modelo ve um prompt diferente do treino.

  2) SLM HF (pasta com config.json, ex.: Qwen fine-tuned) -> GGUF via
     convert_hf_to_gguf.py do llama.cpp (F16) + quantizacao opcional.

Ambos: geram o Modelfile oficial em ollama/ e podem registar no Ollama.

Uso:
    python export_gguf.py --check
    python export_gguf.py modelos/tinygpt_curado.pt
    python export_gguf.py modelos/tinygpt_curado.pt --create
    python export_gguf.py --from-hf modelos/slm-qwen --out modelos/slm.gguf
    python export_gguf.py --from-hf modelos/slm-qwen --quant q4_k_m --create
"""

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONVERT_SCRIPT_CANDIDATES = [
    "convert_hf_to_gguf.py",
    "convert-hf-to-gguf.py",
]

SYSTEM_DEFAULT = (
    "Voce e um assistente especialista no alfabeto, que responde em portugues. "
    "O usuario informa um alfabeto em JSON e faz perguntas sobre ele. "
    "Responda de forma curta e exata."
)

# Template do SLM de conversa (o mesmo do treino em `train_chat.py`).
# O `\n` e' uma quebra de linha real quando o Modelfile e' escrito.
CHAT_TEMPLATE = "<bos>User: {{ .Prompt }}\nAssistant: "


# ======================================================================
# caminho 1: TinyGPT (.pt) -> gpt2
# ======================================================================
def _gpt2_byte_encoder():
    """bytes_to_unicode() do GPT-2 (o "byte encoder").

    O tokenizer BPE do llama.cpp aplica esta representacao ANTES de dividir
    o texto: espaco -> 'G' (U+0120), newline -> 'J' (U+010A), e os
    printaveis ASCII ficam iguais. Sem esta tabela, o fallback do llama.cpp
    nao encontra o espaco e o prompt chega ao modelo diferente do treino.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _byteenc_vocab(old_tokens):
    """Nomes dos tokens no byte-encode do GPT-2 (ids do modelo preservados)."""
    enc = _gpt2_byte_encoder()
    tokens = []
    for tok in old_tokens:
        tokens.append(enc.get(ord(tok), tok) if len(tok) == 1 else tok)
    return tokens


def export_tinygpt_to_gguf(checkpoint, out, arch="gpt2", tokenizer_model="gpt2",
                           add_bos=True, add_eos=False, vocab_mode="byteenc"):
    """TinyGPT .pt -> GGUF (gpt2)."""
    import gguf
    import numpy as np
    import torch

    from minigpt.utils import load_checkpoint

    model, tokenizer, cfg = load_checkpoint(checkpoint)
    state = model.state_dict()
    n_layer, n_embd, n_head = cfg["n_layer"], cfg["n_embd"], cfg["n_head"]
    ctx = cfg["block_size"]

    old_tokens = [tokenizer.itos[i] for i in range(tokenizer.vocab_size)]
    byte_level = bool(getattr(tokenizer, "byte_level", False))
    if byte_level or vocab_mode == "original":
        tokens = old_tokens            # ja sao byte-level (1:1)
    else:
        tokens = _byteenc_vocab(old_tokens)
    v_new = len(tokens)

    writer = gguf.GGUFWriter(out, arch=arch)
    writer.add_name("TinyGPT-alfabeto-gpt2")
    writer.add_description(
        "TinyGPT especialista no alfabeto exportado no formato gpt2 "
        "(LayerNorm pre-norm, GELU, posicao aprendida) para rodar no Ollama.")
    writer.add_uint32(f"{arch}.block_count", n_layer)
    writer.add_uint32(f"{arch}.context_length", ctx)
    writer.add_uint32(f"{arch}.embedding_length", n_embd)
    writer.add_uint32(f"{arch}.attention.head_count", n_head)
    writer.add_uint32(f"{arch}.feed_forward_length", 4 * n_embd)
    writer.add_float32(f"{arch}.attention.layer_norm_epsilon", 1e-5)
    writer.add_string(f"{arch}.pos_encoding", cfg.get("pos_encoding", "learned"))
    tails = cfg.get("train_alphabets")
    if tails:
        writer.add_string(f"{arch}.train_alphabets", ",".join(tails))
    tasks = cfg.get("tasks")
    if tasks:
        writer.add_string(f"{arch}.tasks", ",".join(tasks))

    specials = {0, 1, 2}
    writer.add_tokenizer_model(tokenizer_model)
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * len(tokens))
    writer.add_token_types([3 if i in specials else 1 for i in range(v_new)])
    if tokenizer_model == "gpt2":
        writer.add_token_merges(["<eos> <pad>"])   # merge inerte (chave exigida)
    writer.add_bos_token_id(tokenizer.bos_id)
    writer.add_eos_token_id(tokenizer.eos_id)
    writer.add_pad_token_id(tokenizer.pad_id)
    writer.add_unk_token_id(tokenizer.pad_id)
    writer.add_add_bos_token(bool(add_bos))
    writer.add_add_eos_token(bool(add_eos))

    def np_of(t, transpose=False):
        a = t.detach().cpu().float().numpy()
        return np.ascontiguousarray(a.T) if transpose else a

    writer.add_tensor("token_embd.weight", np_of(state["tok_emb.weight"]))
    if "pos_emb.weight" in state:
        writer.add_tensor("position_embd.weight", np_of(state["pos_emb.weight"]))
    for i in range(n_layer):
        p = f"blocks.{i}"
        writer.add_tensor(f"blk.{i}.attn_qkv.weight", np_of(state[f"{p}.attn.c_attn.weight"]))
        writer.add_tensor(f"blk.{i}.attn_qkv.bias", np_of(state[f"{p}.attn.c_attn.bias"], False))
        writer.add_tensor(f"blk.{i}.attn_output.weight", np_of(state[f"{p}.attn.c_proj.weight"]))
        writer.add_tensor(f"blk.{i}.attn_output.bias", np_of(state[f"{p}.attn.c_proj.bias"], False))
        writer.add_tensor(f"blk.{i}.attn_norm.weight", np_of(state[f"{p}.ln1.weight"], False))
        writer.add_tensor(f"blk.{i}.attn_norm.bias", np_of(state[f"{p}.ln1.bias"], False))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", np_of(state[f"{p}.mlp.fc.weight"]))
        writer.add_tensor(f"blk.{i}.ffn_up.bias", np_of(state[f"{p}.mlp.fc.bias"], False))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", np_of(state[f"{p}.mlp.proj.weight"]))
        writer.add_tensor(f"blk.{i}.ffn_down.bias", np_of(state[f"{p}.mlp.proj.bias"], False))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", np_of(state[f"{p}.ln2.weight"], False))
        writer.add_tensor(f"blk.{i}.ffn_norm.bias", np_of(state[f"{p}.ln2.bias"], False))
    writer.add_tensor("output_norm.weight", np_of(state["ln_f.weight"], False))
    writer.add_tensor("output_norm.bias", np_of(state["ln_f.bias"], False))
    writer.add_tensor("output.weight", np_of(state["head.weight"]))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"GGUF salvo: {out} ({os.path.getsize(out)/1e6:.2f} MB, arch={arch}, "
          f"vocab={v_new}, add_bos={add_bos})")
    return out


def verify_gguf(path, show=5):
    import gguf
    r = gguf.GGUFReader(path)
    arch = r.fields.get("general.architecture")
    print(f"magic OK | arch={arch.contents() if arch else '?'} | "
          f"tensores={len(r.tensors)} | kvs={len(r.fields)}")
    for ten in r.tensors[:show]:
        print(f"  {ten.name}: {list(ten.shape)}")
    return True


# ======================================================================
# caminho 2: SLM HF -> GGUF (llama.cpp)
# ======================================================================
def run(cmd, cwd=None):
    print("+ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"comando falhou ({r.returncode}): {' '.join(cmd)}")
    return r


def ollama_bin():
    p = shutil.which("ollama")
    if p:
        return p
    for cand in [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
        r"C:\Program Files\Ollama\ollama.exe",
    ]:
        if os.path.exists(cand):
            return cand
    return None


def check_env():
    print("== check ==")
    ok = True
    ob = ollama_bin()
    print(f"ollama: {ob or 'NAO ENCONTRADO no PATH'}")
    try:
        if ob:
            run([ob, "--version"])
    except SystemExit:
        ok = False
    for pkg in ("torch", "numpy", "gguf"):
        try:
            __import__(pkg)
            print(f"{pkg}: ok")
        except ImportError:
            print(f"{pkg}: ausente")
            ok = False
    print("llama.cpp: " + ("presente em ./llama.cpp" if os.path.isdir("llama.cpp")
                           else "ausente (clonado sob demanda para o caminho HF)"))
    print("RESULTADO: " + ("OK" if ok else "PENDENCIAS acima"))
    return ok


def ensure_llama_cpp(dest="llama.cpp"):
    if os.path.isdir(dest):
        return dest
    if not shutil.which("git"):
        raise SystemExit("git nao encontrado: clone manual de llama.cpp para ./llama.cpp")
    run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", dest])
    return dest


def find_convert_script(llama_dir):
    for root, _dirs, files in os.walk(llama_dir):
        for cand in CONVERT_SCRIPT_CANDIDATES:
            if cand in files:
                return os.path.join(root, cand)
    raise SystemExit("convert_hf_to_gguf.py nao encontrado dentro de ./llama.cpp")


def convert_hf_to_gguf(hf_dir, out_gguf, llama_dir="llama.cpp", extra_args=None):
    cfg = os.path.join(hf_dir, "config.json")
    if not os.path.exists(cfg):
        raise SystemExit(f"diretorio HF invalido (sem config.json): {hf_dir}")
    import json
    with open(cfg, encoding="utf-8") as f:
        arch = json.load(f).get("model_type", "?")
    print(f"model_type detectado: {arch}")
    llama_dir = ensure_llama_cpp(llama_dir)
    script = find_convert_script(llama_dir)
    cmd = [sys.executable, script, hf_dir, "--outfile", out_gguf, "--outtype", "f16"]
    if extra_args:
        cmd += extra_args
    run(cmd)
    print(f"GGUF F16 salvo em: {out_gguf}")
    return out_gguf


def quantize_gguf(src, quant="q4_k_m", llama_dir="llama.cpp"):
    base, _ = os.path.splitext(src)
    dst = f"{base}.{quant.upper()}.gguf"
    cands = [
        os.path.join(llama_dir, "build", "bin", "llama-quantize"),
        os.path.join(llama_dir, "build", "bin", "llama-quantize.exe"),
        os.path.join(llama_dir, "llama-quantize"),
        os.path.join(llama_dir, "llama-quantize.exe"),
    ]
    exe = next((c for c in cands if os.path.exists(c)), shutil.which("llama-quantize"))
    if not exe:
        print("AVISO: llama-quantize nao encontrado (pulei --quant).")
        print(f"  O GGUF F16 ({src}) ja funciona no Ollama.")
        return src
    run([exe, src, dst, quant.upper()])
    print(f"GGUF quantizado salvo em: {dst}")
    return dst


# ======================================================================
# Modelfile + Ollama (comum aos dois caminhos)
# ======================================================================
def write_modelfile(gguf_path, out, name, template=None, system=None,
                    temperature=0, num_ctx=512, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Modelfile oficial.

    Para o TinyGPT, o TEMPLATE monta o prompt no formato de TREINO:
        <bos>{"alfabeto":"A-Z"}\nQ: <pergunta>\nA:
    Sem isto o modelo nao reconhece a pergunta e devolve o eco do JSON.
    """
    if template is None and system is None:
        template = ('<bos>{"alfabeto":"%s"}\n'
                    "Q: {{ .Prompt }}\n"
                    "A: ") % alphabet
    gguf_abs = os.path.abspath(gguf_path).replace("\\", "/")
    content = f"FROM {gguf_abs}\n"
    if template is not None:
        content += f'TEMPLATE """{template}"""\n'
    if system is not None:
        content += f'SYSTEM """{system}"""\n'
    content += (f"PARAMETER temperature {temperature}\n"
                f"PARAMETER num_ctx {num_ctx}\n")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # newline="\n": em Windows o modo texto traduz \n -> \r\n e o "\r" entra
    # no prompt do modelo (template multi-linha) -> resposta corrompida.
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"Modelfile escrito: {out} (ollama create {name} -f {out})")
    return out


def ollama_create(name, modelfile):
    ob = ollama_bin()
    if not ob:
        raise SystemExit("ollama nao encontrado. Instale https://ollama.com/download")
    run([ob, "create", name, "-f", modelfile])
    print(f"Modelo criado no Ollama: {name}")
    print(f"Teste: ollama run {name} \"Depois de C?\"")


def build_modelfile_from_dataset(base="qwen3:8b", name="alfabeto-slm",
                                 out="ollama/Modelfile.alfabeto", system=None,
                                 temperature=0.0, num_ctx=4096):
    """Modelfile do SLM (base HF) com few-shot tirado do dataset curado.

    O dataset curado (input/dialogos.jsonl) vira exemplos no SYSTEM prompt:
    assim o modelo base ganha o "sabor" do dominio sem treino.
    """
    from minigpt.dialogues import load_curated

    entries = load_curated()
    if system is None:
        system = SYSTEM_DEFAULT
    seen, shots = set(), []
    for e in entries:
        t = e.get("task", "?")
        if e.get("alphabet") and t not in seen:
            seen.add(t)
            shots.append(f"Pergunta: {e['user'][0]}\nResposta: {e['assistant'][0]}")
    for e in entries:
        if not e.get("alphabet") and e["id"] in ("greet_oi_01", "cap_exemplos_01"):
            shots.append(f"Pergunta: {e['user'][0]}\nResposta: {e['assistant'][0]}")
    content = (
        f"FROM {base}\n"
        f"PARAMETER temperature {temperature}\n"
        f"PARAMETER num_ctx {num_ctx}\n"
        f'PARAMETER stop "Pergunta:"\n'
        'SYSTEM """\n' + system + "\n\n"
        "Voce recebe o alfabeto em JSON e UMA pergunta. Responda APENAS a resposta, "
        "curta e exata, sem explicacoes (salvo se pedirem explicacao). "
        "Conversa geral em portugues: natural e breve.\n\n"
        "Exemplos:\n" + "\n\n".join(shots) + '\n"""\n'
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"Modelfile escrito: {out} (base {base}, {len(entries)} dialogos curados)")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Exporta para GGUF (Ollama)")
    p.add_argument("input", nargs="?", default=None,
                   help="checkpoint .pt (TinyGPT) — o tipo e detetado")
    p.add_argument("--from-hf", default=None, help="pasta de um modelo HF (Qwen etc.)")
    p.add_argument("--from-dataset", action="store_true",
                   help="cria so o Modelfile do SLM (few-shot do dataset curado)")
    p.add_argument("--base-model", default="qwen3:8b",
                   help="base HF ja no ollama (para --from-dataset)")
    p.add_argument("--out", default=None, help="GGUF de saida")
    p.add_argument("--check", action="store_true")
    p.add_argument("--vocab", default="byteenc", choices=["byteenc", "original"])
    p.add_argument("--no-add-bos", action="store_true")
    p.add_argument("--llama-cpp-dir", default="llama.cpp")
    p.add_argument("--quant", default=None, help="ex.: q4_k_m (caminho HF)")
    p.add_argument("--modelfile", default=None)
    p.add_argument("--system", default=SYSTEM_DEFAULT)
    p.add_argument("--template", default=None)
    p.add_argument("--chat", action="store_true",
                   help="template de conversa (<bos>User: ...\\nAssistant: )")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-ctx", type=int, default=512)
    p.add_argument("--ollama-name", default=None)
    p.add_argument("--create", action="store_true", help="roda 'ollama create' ao final")
    return p.parse_args(argv)


def main():
    args = parse_args()
    if args.check:
        check_env()
        return

    if args.from_dataset:
        mf = build_modelfile_from_dataset(base=args.base_model,
                                         name=args.ollama_name or "alfabeto-slm",
                                         out=args.modelfile or "ollama/Modelfile.alfabeto",
                                         system=args.system,
                                         temperature=args.temperature,
                                         num_ctx=args.num_ctx)
        if args.create:
            ollama_create(args.ollama_name or "alfabeto-slm", mf)
        return

    is_hf = bool(args.from_hf)
    src = args.from_hf if is_hf else args.input
    if not src:
        raise SystemExit("informe o checkpoint: export_gguf.py modelos/x.pt  "
                         "(ou --from-hf <dir HF>)")

    if is_hf:
        out = args.out or "modelos/slm.gguf"
        gguf_path = convert_hf_to_gguf(src, out, args.llama_cpp_dir)
        if args.quant:
            gguf_path = quantize_gguf(gguf_path, args.quant, args.llama_cpp_dir)
        name = args.ollama_name or "alfabeto-slm"
        modelfile = args.modelfile or "ollama/Modelfile.alfabeto"
        write_modelfile(gguf_path, modelfile, name, template=args.template,
                        system=args.system, temperature=args.temperature,
                        num_ctx=args.num_ctx)
    else:
        if not src.endswith(".pt"):
            raise SystemExit(f"entrada invalida: {src} (esperado .pt do TinyGPT ou --from-hf)")
        out = args.out or "modelos/tinygpt.gguf"
        gguf_path = export_tinygpt_to_gguf(src, out, add_bos=not args.no_add_bos,
                                            vocab_mode=args.vocab)
        verify_gguf(gguf_path)
        name = args.ollama_name or "tinygpt"
        modelfile = args.modelfile or "ollama/Modelfile.tinygpt"
        template, system = args.template, args.system
        if args.chat:
            # conversa: o modelo foi treinado sem SYSTEM e com <bos>User: ...
            template, system = CHAT_TEMPLATE, None
        write_modelfile(gguf_path, modelfile, name, template=template,
                        system=system, temperature=args.temperature,
                        num_ctx=args.num_ctx)

    if args.create:
        ollama_create(name, modelfile)
    else:
        print("\nPara registrar no Ollama:")
        print(f"  ollama create {name} -f {modelfile}")
        print(f"  ollama run {name}")


if __name__ == "__main__":
    main()
