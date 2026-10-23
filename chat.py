"""Chat com o modelo especialista em alfabetos.

O modelo NAO e um chatbot de linguagem geral: ele le um alfabeto (JSON) e
responde perguntas sobre ele (total, primeira, ultima, depois/antes de uma
letra, posicao, ultimas N).

O alfabeto NAO precisa ser passado na linha de comando: o chat usa
automaticamente o alfabeto do dataset gravado no checkpoint
(`train_alphabets`). O JSON e montado internamente a cada pergunta.

Uso:
    .venv\\Scripts\\python.exe chat.py
    .venv\\Scripts\\python.exe chat.py --checkpoint modelos/especialista_full.pt
    .venv\\Scripts\\python.exe chat.py --checkpoint modelos/especialista.pt --backend onnx
    .venv\\Scripts\\python.exe chat.py --checkpoint modelos/especialista_full.pt --backend onnx

Comandos no chat:
    /alfabeto              mostra o alfabeto atual (vem do dataset)
    /alfabetos             lista os alfabetos do dataset (para trocar)
    /usar 2                troca para o alfabeto nº 2 do dataset
    /usar ABCDEFGHIJ       troca para um alfabeto do dataset pelas letras
    /exemplos              mostra perguntas que o modelo entende
    /help                  ajuda
    /sair                  encerra

Perguntas tipicas (pode falar livre — o chat traduz para a forma do treino):
    Total? ("quantas letras tem?")
    Primeira? / Ultima? ("mostre a ultima letra")
    Depois de M? ("qual vem depois do m?")
    Antes de M? / Posicao de M?
    Ultimas 3? / Mostre o alfabeto?
"""

import argparse

from minigpt.data import ALPHABETS, build_json_block, canonicalize_question, make_examples
from minigpt.utils import (
    clean_answer,
    generate_continuation,
    generate_continuation_onnx,
    load_checkpoint,
    load_onnx_session,
    load_tokenizer_config,
    resolve_onnx_path,
)

HELP = r"""
Comandos:
  /alfabeto        mostra o alfabeto atual (vem do dataset do checkpoint)
  /alfabetos       lista os alfabetos do dataset (com numero para /usar)
  /usar N          troca para o alfabeto nº N do dataset (ex.: /usar 2)
  /usar XXXX       troca para um alfabeto do dataset pelas letras
  /exemplos        sugestoes de perguntas
  /help            esta ajuda
  /sair            encerrar

Perguntas: Mostre o alfabeto? | Mostre as N primeiras? | Ultimas N? |
           Total? | Primeira? | Ultima? | Depois de X? | Antes de X? |
           Posicao de X?

Nota: o alfabeto e injetado automaticamente como JSON no prompt.
Voce so digita a pergunta. Troque de alfabeto com /usar.
"""


def parse_args():
    p = argparse.ArgumentParser(description="Chat com o modelo especialista (CPU)")
    p.add_argument("--checkpoint", type=str, default="modelos/especialista.pt")
    p.add_argument("--backend", choices=["pt", "onnx"], default="pt",
                   help="pt = PyTorch (.pt); onnx = onnxruntime (.onnx, prova de paridade/deploy)")
    p.add_argument("--onnx-path", type=str, default=None,
                   help="caminho do .onnx (default: mesmo nome do checkpoint com .onnx)")
    p.add_argument("--alfabeto", type=str, default=None,
                   help="(opcional, avancado) sobrescreve o alfabeto do dataset. "
                        "Default: usa o alfabeto nº 1 gravado no checkpoint.")
    p.add_argument("--usar", type=str, default=None,
                   help="(opcional) indice (1-based) ou letras do alfabeto do dataset para iniciar. "
                        "Ex.: --usar 2")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="max de tokens gerados (default: automatico = block_size, cabe a maior resposta)")
    p.add_argument("--raw", action="store_true",
                   help="FASE 1: envia o texto cru ao modelo, sem traduzir via regex. "
                        "Exige modelo treinado com parafrases.")
    return p.parse_args()


def normalize_alphabet(text):
    letters = [c for c in text.upper() if c.isalpha()]
    if not letters:
        raise ValueError("alfabeto vazio")
    if len(set(letters)) != len(letters):
        raise ValueError("o alfabeto tem letras repetidas")
    return letters


def ask_pt(model, tokenizer, alphabet, question, json_style, greedy, temperature, top_k, max_new):
    json_block = build_json_block(alphabet, json_style)
    prompt = f"<bos>{json_block}\nQ: {question}\nA: "
    cont = generate_continuation(
        model, tokenizer, prompt, max_new_tokens=max_new,
        greedy=greedy, temperature=temperature, top_k=top_k,
    )
    return clean_answer(cont)


def ask_onnx(session, tokenizer, block_size, alphabet, question, json_style, greedy, temperature, top_k, max_new):
    json_block = build_json_block(alphabet, json_style)
    prompt = f"<bos>{json_block}\nQ: {question}\nA: "
    cont = generate_continuation_onnx(
        session, tokenizer, prompt, block_size, max_new_tokens=max_new,
        greedy=greedy, temperature=temperature, top_k=top_k,
    )
    return clean_answer(cont)


def main():
    args = parse_args()
    backend = args.backend
    if backend == "onnx":
        onnx_path = resolve_onnx_path(args.checkpoint, args.onnx_path)
        tokenizer, cfg, num_params = load_tokenizer_config(args.checkpoint)
        session = load_onnx_session(onnx_path)
        model = None
        block_size = cfg["block_size"]
    else:
        model, tokenizer, cfg = load_checkpoint(args.checkpoint)
        session = None
        block_size = model.block_size
        num_params = model.num_params
        onnx_path = resolve_onnx_path(args.checkpoint, args.onnx_path)
    json_style = cfg.get("json_style", "compact")
    greedy = args.temperature == 0.0
    min_len = cfg.get("min_length", 1)
    max_len = cfg.get("max_length", 26)
    known = cfg.get("train_alphabets")
    # max tokens automatico: usa o block_size para sempre caber a maior
    # resposta (generate_continuation limita pelo espaco restante).
    default_max_new = block_size

    def resolve_default_alphabet():
        # 1) override explicito tem prioridade (modo avancado)
        if args.alfabeto:
            try:
                return normalize_alphabet(args.alfabeto), "(override --alfabeto)"
            except ValueError:
                pass
        # 2) --usar N ou letras
        if args.usar and known:
            sel = args.usar.strip().upper()
            if sel.isdigit():
                idx = int(sel) - 1
                if 0 <= idx < len(known):
                    return list(known[idx]), f"(dataset nº {idx + 1})"
            else:
                letters = [c for c in sel if c.isalpha()]
                if "".join(letters) in known:
                    return letters, "(dataset)"
        # 3) padrao: primeiro alfabeto do dataset gravado no checkpoint
        if known:
            return list(known[0]), "(dataset nº 1)"
        # 4) generalista sem train_alphabets: cai no canonico
        return list(ALPHABETS["canonical"])[:max_len], "(canonico)"

    alphabet, origem = resolve_default_alphabet()

    def warn_if_out_of_range(letters):
        if not (min_len <= len(letters) <= max_len):
            print(f"  aviso: o modelo foi treinado com {min_len}-{max_len} letras; "
                  f"{len(letters)} pode dar respostas ruins.")
        if known and "".join(letters) not in known:
            print("  aviso: este alfabeto nao esta entre os conhecidos pelo especialista "
                  "(veja /alfabetos).")

    print("=" * 60)
    print("Modelo especialista em alfabetos")
    print(f"checkpoint: {args.checkpoint} | backend: {backend}"
          + (f" ({onnx_path})" if backend == "onnx" else "")
          + f" | params: {num_params:,} | vocab {tokenizer.vocab_size}")
    print(f"treinado p/ alfabetos de {min_len} a {max_len} letras")
    if known:
        print(f"dataset do checkpoint ({len(known)} alfabetos): "
              f"{', '.join(known[:6])}{' ...' if len(known) > 6 else ''}")
    print(f"alfabeto em uso: {''.join(alphabet)} {origem}  <- JSON montado sozinho, so digite a pergunta")
    print(f"max-new-tokens: {'auto (' + str(default_max_new) + ')' if args.max_new_tokens is None else args.max_new_tokens}")
    print("digite /help para a lista de comandos")
    print("=" * 60)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            cmd = cmd.lower()
            if cmd in ("/sair", "/exit", "/quit"):
                break
            if cmd == "/help":
                print(HELP)
            elif cmd == "/alfabeto":
                # sem argumento: so mostra. Com argumento: modo avancado livre.
                if rest.strip():
                    try:
                        alphabet = normalize_alphabet(rest)
                        origem = "(livre, fora do dataset)"
                        print("alfabeto atual:", "".join(alphabet))
                        warn_if_out_of_range(alphabet)
                    except ValueError as exc:
                        print("erro:", exc)
                else:
                    print("alfabeto atual:", "".join(alphabet), origem)
            elif cmd == "/usar":
                if not known:
                    print("este checkpoint nao declarou alfabetos do dataset (modelo generalista).")
                elif not rest.strip():
                    print("uso: /usar N  (ex.: /usar 2) — veja /alfabetos")
                else:
                    sel = rest.strip().upper()
                    if sel.isdigit():
                        idx = int(sel) - 1
                        if 0 <= idx < len(known):
                            alphabet = list(known[idx])
                            origem = f"(dataset nº {idx + 1})"
                            print("alfabeto atual:", "".join(alphabet), origem)
                        else:
                            print(f"erro: use 1..{len(known)} (veja /alfabetos)")
                    else:
                        letters = [c for c in sel if c.isalpha()]
                        key = "".join(letters)
                        if key in known:
                            alphabet = letters
                            origem = "(dataset)"
                            print("alfabeto atual:", key, origem)
                        else:
                            print("erro: esse alfabeto nao esta no dataset. Veja /alfabetos.")
                            warn_if_out_of_range(letters) if letters else None
            elif cmd == "/exemplos":
                for cat, q, _a in make_examples(alphabet, variants=False):
                    print(f"  ({cat}) {q}")
            elif cmd in ("/alfabetos", "/list-alfabetos"):
                if known:
                    print("alfabetos do dataset (use /usar N):")
                    for i, a in enumerate(known, start=1):
                        mark = " <-- em uso" if "".join(alphabet) == a else ""
                        print(f"  {i}. {a}{mark}")
                else:
                    print("este checkpoint nao declarou alfabetos conhecidos (modelo generalista).")
            else:
                print("comando desconhecido. use /help")
            continue

        max_new = args.max_new_tokens if args.max_new_tokens is not None else default_max_new
        if args.raw:
            canon = line
        else:
            canon = canonicalize_question(line)
            if canon is None:
                print("nao entendi. Tente uma destas (ou veja /exemplos):")
                print("  Total? | Primeira? | Ultima? | Mostre o alfabeto?")
                print("  Mostre as 3 primeiras? | Ultimas 3? | Depois de C? | Antes de B? | Posicao de C?")
                continue
            if canon != line:
                print(f"  (entendi como: {canon})")
        try:
            if backend == "onnx":
                answer = ask_onnx(
                    session, tokenizer, block_size, alphabet, canon, json_style,
                    greedy, args.temperature, args.top_k, max_new,
                )
            else:
                answer = ask_pt(
                    model, tokenizer, alphabet, canon, json_style,
                    greedy, args.temperature, args.top_k, max_new,
                )
        except KeyError as exc:
            print(f"erro: nao consigo ler {exc} (fora do vocabulario do modelo).")
            continue
        print("A:", answer)


if __name__ == "__main__":
    main()
