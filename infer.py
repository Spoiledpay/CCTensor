"""Inferencia/avaliacao do TinyGPT treinado.

O alfabeto NAO precisa ser passado: por padrao usa o alfabeto do dataset
gravado no checkpoint (`train_alphabets[0]`). O JSON e montado sozinho.

Exemplos:
    .venv\\Scripts\\python.exe infer.py --prompt "Depois de C?"
    .venv\\Scripts\\python.exe infer.py --prompt "Mostre o alfabeto?"
    .venv\\Scripts\\python.exe infer.py --eval

Avancado (override):
    .venv\\Scripts\\python.exe infer.py --letters MNYBI --prompt "Ultimas 3?"

Por padrao usa decodificacao greedy (deterministica). Use --temperature/--top-k
para amostragem. Para conversar, use chat.py.
"""

import argparse

from minigpt.data import ALPHABETS, TASK_GROUPS, build_json_block, canonicalize_question, format_prefix, make_examples, val_prompts_for
from minigpt.utils import (
    clean_answer,
    evaluate_exact_match,
    evaluate_exact_match_onnx,
    generate_continuation,
    generate_continuation_onnx,
    load_checkpoint,
    load_onnx_session,
    load_tokenizer_config,
    resolve_onnx_path,
    summarize,
)


def parse_args():
    p = argparse.ArgumentParser(description="Inferencia do TinyGPT (CPU)")
    p.add_argument("--checkpoint", type=str, default="modelos/especialista.pt")
    p.add_argument("--backend", choices=["pt", "onnx"], default="pt",
                   help="pt = PyTorch (.pt); onnx = onnxruntime (.onnx)")
    p.add_argument("--onnx-path", type=str, default=None,
                   help="caminho do .onnx (default: mesmo nome do checkpoint com .onnx)")
    p.add_argument("--prompt", type=str, default=None, help="ex: 'Depois de C?' (usa o alfabeto do dataset)")
    p.add_argument("--alphabet", type=str, default=None, choices=list(ALPHABETS.keys()),
                   help="(avancado) preset de alfabeto; ignora o dataset")
    p.add_argument("--letters", type=str, default=None, help="(avancado) alfabeto explicito, ex: MNYBI (ignora o dataset)")
    p.add_argument("--usar", type=str, default=None, help="(opcional) indice 1-based do alfabeto do dataset. Ex.: --usar 2")
    p.add_argument("--eval", action="store_true", help="avalia em alfabetos nao vistos (por categoria)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="max de tokens gerados (default: automatico = block_size)")
    p.add_argument("--raw", action="store_true",
                   help="FASE 1: envia o texto cru ao modelo, sem traduzir via regex.")
    return p.parse_args()


def ask_pt(model, tokenizer, alphabet, question, json_style, greedy, temperature, top_k, max_new):
    json_block = build_json_block(alphabet, json_style)
    prompt = format_prefix(json_block, question)
    cont = generate_continuation(
        model, tokenizer, prompt, max_new_tokens=max_new,
        greedy=greedy, temperature=temperature, top_k=top_k,
    )
    return clean_answer(cont)


def ask_onnx(session, tokenizer, block_size, alphabet, question, json_style, greedy, temperature, top_k, max_new):
    json_block = build_json_block(alphabet, json_style)
    prompt = format_prefix(json_block, question)
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
    print(f"Modelo: {args.checkpoint} | backend: {backend}"
          + (f" ({onnx_path})" if backend == "onnx" else "")
          + f" | params {num_params:,} | vocab {tokenizer.vocab_size} | json {json_style}")
    greedy = args.temperature == 0.0

    if args.eval:
        allowed = cfg.get("tasks")  # especialista: avalia so o que foi treinado
        prompts, answers, cats = val_prompts_for(
            cfg.get("n_alphabets", 100),
            cfg.get("n_val_alphabets", 15),
            cfg.get("seed", 1337),
            json_style,
            cfg.get("min_length", 6),
            cfg.get("max_length", 10),
            categories=allowed,
        )
        if allowed:
            print(f"(eval filtrado p/ tarefas do checkpoint: {', '.join(allowed)})")
        if backend == "onnx":
            results = evaluate_exact_match_onnx(
                session, tokenizer, block_size, prompts, answers, cats, greedy=greedy)
        else:
            results = evaluate_exact_match(model, tokenizer, prompts, answers, cats, greedy=greedy)
        correct, total, per = summarize(results)
        print(f"\nAcuracia exata em alfabetos NAO vistos: {correct}/{total} = {correct / total:.1%}")
        for group in TASK_GROUPS:
            c, n = per.get(group, (0, 0))
            if n:
                print(f"  {group:9s}: {c:4d}/{n:<4d} = {c / n:5.1%}")
        print("\nErros (ate 8):")
        shown = 0
        for prompt, gold, pred, ok, cat in results:
            if not ok and shown < 8:
                q = prompt.split("Q: ")[1].split("\nA:")[0]
                print(f"  [XX] ({cat}) {q} -> {pred!r} (esperado {gold!r})")
                shown += 1

    if args.prompt:
        known = cfg.get("train_alphabets")
        if args.letters:
            alphabet = list(args.letters.upper())
            origem = "(override --letters)"
        elif args.alphabet:
            alphabet = ALPHABETS[args.alphabet]
            origem = f"(preset --alphabet {args.alphabet})"
        elif args.usar and known:
            try:
                idx = int(args.usar) - 1
                alphabet = list(known[idx])
                origem = f"(dataset nº {idx + 1})"
            except (ValueError, IndexError):
                raise SystemExit(f"--usar invalido: use 1..{len(known)}")
        elif known:
            alphabet = list(known[0])
            origem = "(dataset nº 1)"
        else:
            alphabet = ALPHABETS["canonical"]
            origem = "(canonico)"
        max_new = args.max_new_tokens if args.max_new_tokens is not None else block_size
        if args.raw:
            canon = args.prompt
        else:
            canon = canonicalize_question(args.prompt)
            if canon is None:
                raise SystemExit(
                    f"nao entendi: {args.prompt!r}\n"
                    "Tente: 'Total?', 'Primeira?', 'Ultima?', 'Mostre o alfabeto?', "
                    "'Mostre as 3 primeiras?', 'Ultimas 3?', 'Depois de C?', 'Antes de B?', 'Posicao de C?'"
                )
            if canon != args.prompt:
                print(f"(entendi como: {canon})")
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
            raise SystemExit(
                f"erro: nao consigo ler {exc} (fora do vocabulario do modelo). "
                "Reformule com as palavras dos exemplos."
            )
        print(f"\nalfabeto {origem}: {''.join(alphabet)}\nPergunta: {canon}\nResposta: {answer}")

    if not args.eval and not args.prompt:
        print("Nada a fazer. Use --eval e/ou --prompt (ou chat.py para conversar).")


if __name__ == "__main__":
    main()
