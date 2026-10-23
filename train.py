"""Treino unico do projeto (le tudo de input/). Escolha o modo:

  --mode especialista  TinyGPT que DECORA alfabetos fixos de input/alfabeto.json
                       (rapido, CPU, p/ chat sobre alfabetos conhecidos).
  --mode generalista   TinyGPT que busca GENERALIZAR p/ alfabetos novos
                       (on-the-fly, val disjunta, CPU/GPU).
  --mode slm           Fine-tune LoRA de SLM padrao (Qwen) no dataset CURADO
                       input/dialogos_*.jsonl (alfabeto + cotidiano PT-BR),
                       pronto p/ GGUF/Ollama via export_gguf.py.

Exemplos:
    .\\.venv\\Scripts\\python.exe train.py --mode especialista
    .\\.venv\\Scripts\\python.exe train.py --mode especialista --min-length 26 --max-length 26 --block-size 192 --steps 400 --out modelos/especialista_full.pt
    .\\.venv\\Scripts\\python.exe train.py --mode generalista
    .\\.venv\\Scripts\\python.exe train.py --mode slm --build-dataset-only
    .\\.venv\\Scripts\\python.exe train.py --mode slm

O config.json da os defaults (secoes generalista/especialista/slm); a CLI sobrepoe.
"""

import argparse
import math
import random
import re
import sys
import time

import torch

from minigpt.config import DEFAULT_CONFIG, load_config, section
from minigpt.data import (
    TASK_GROUPS,
    build_dataset,
    build_json_block,
    encode_example,
    format_prefix,
    load_base_alphabet,
    make_examples,
    sample_batch,
    val_prompts_for,
)
from minigpt.dialogues import load_curated, to_messages
from minigpt.model import TinyGPT
from minigpt.tokenizer import ItemChTokenizer
from minigpt.utils import evaluate_exact_match, save_checkpoint, summarize

MODES = ("especialista", "generalista", "slm")

# tarefas de copia/extracao + neighbor (next/prev). `position` fica de fora por
# padrao no especialista (exige emitir numero, nao e copia).
DEFAULT_TASKS = ["show", "first_n", "last_n", "count", "first", "last", "next", "prev"]


def specialist_alphabets(base, n, min_length, max_length, seed=0):
    """Conjunto FIXO de alfabetos com LETRAS DISTINTAS derivados de input."""
    rng = random.Random(seed)
    out = [list(base[:max_length])]
    pool = list(base[max_length:])
    rng.shuffle(pool)
    i = 0
    while i < len(pool) and len(out) < n:
        size = rng.randint(min_length, max_length)
        chunk = pool[i:i + size]
        i += size
        if len(chunk) >= min_length:
            out.append(chunk)
    return out


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=DEFAULT_CONFIG)
    pre.add_argument("--mode", choices=MODES, default="generalista")
    known, _ = pre.parse_known_args(argv)
    cfg = load_config(known.config)
    m, d = section(cfg, "model"), section(cfg, "data")

    p = argparse.ArgumentParser(description="Treino unico (especialista|generalista|slm)")
    p.add_argument("--config", default=known.config, help="arquivo de config json")
    p.add_argument("--mode", choices=MODES, default=known.mode)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--input", type=str, default=d.get("input", "input/alfabeto.json"))
    p.add_argument("--no-onnx", action="store_true",
                   help="nao exportar o .onnx ao final (default: exporta sempre)")
    p.add_argument("--onnx-int8", action="store_true",
                   help="tambem gerar versao quantizada INT8 do .onnx")

    if known.mode == "especialista":
        e = section(cfg, "especialista")
        tasks_cfg = e.get("tasks", DEFAULT_TASKS)
        p.add_argument("--steps", type=int, default=e.get("steps", 300))
        p.add_argument("--batch-size", type=int, default=e.get("batch_size", 64))
        p.add_argument("--block-size", type=int, default=m.get("block_size", 96))
        p.add_argument("--n-embd", type=int, default=m.get("n_embd", 96))
        p.add_argument("--n-head", type=int, default=m.get("n_head", 4))
        p.add_argument("--n-layer", type=int, default=m.get("n_layer", 4))
        p.add_argument("--lr", type=float, default=e.get("lr", 3e-3))
        p.add_argument("--weight-decay", type=float, default=e.get("weight_decay", 0.01))
        p.add_argument("--answer-weight", type=float, default=e.get("answer_weight", 20.0))
        p.add_argument("--prompt-weight", type=float, default=e.get("prompt_weight", 0.0),
                       help="peso da loss no PROMPT (JSON+pergunta). 0 = so a resposta "
                            "e supervisionada. Com >0 o modelo gasta capacidade "
                            "copiando a pergunta (era 83% da massa da loss).")
        p.add_argument("--stop-weight", type=float, default=e.get("stop_weight", 1.0),
                       help="peso da loss no '\\n' e '<eos>' (ensina a parar)")
        p.add_argument("--fresh-alphabets", action=argparse.BooleanOptionalAction,
                       default=e.get("fresh_alphabets", True),
                       help="regerar os alfabetos de treino a cada epoca: cada "
                            "alfabeto aparece 1x, logo a memorizacao dos pares "
                            "pergunta->resposta deixa de ser uma opcao.")
        p.add_argument("--n-alphabets", type=int, default=e.get("n_alphabets", 8))
        p.add_argument("--min-length", type=int, default=d.get("min_length", 6))
        p.add_argument("--max-length", type=int, default=d.get("max_length", 10))
        p.add_argument("--tasks", type=str,
                       default=",".join(tasks_cfg) if isinstance(tasks_cfg, list) else tasks_cfg)
        p.add_argument("--tokenizer", choices=["item", "byte"], default=e.get("tokenizer", "byte"),
                       help="byte = byte-level (sem char desconhecido, export GGUF 1:1); "
                            "item = char/item (vocab pequena)")
        p.add_argument("--dataset", default=e.get("dataset"),
                       help="dataset curado (input/dialogos.jsonl). Se definido, "
                            "o treino usa os dialogos curados em vez dos templates.")
        p.add_argument("--intents", default=e.get("intents"),
                       help="catalogo de intents (input/intents.jsonl): o treino "
                            "instancia os templates curados sobre varios "
                            "alfabetos (respostas verificadas) -> generaliza.")
        p.add_argument("--n-daily-repeat", type=int, default=e.get("n_daily_repeat", 1))
        p.add_argument("--no-daily-variants", action="store_true",
                       help="nao gerar variantes de caixa/pontuacao do cotidiano")
        p.add_argument("--epochs", type=int, default=e.get("epochs", 0),
                       help="0 = modo steps (legado); N>0 = N epocas com DataLoader, "
                            "barra tqdm, eval de respostas e historico CSV")
        p.add_argument("--num-threads", type=int, default=0,
                       help="threads CPU do torch (0 = auto; benchmark: 8 ideal aqui)")
        p.add_argument("--eval-every", type=int, default=5,
                       help="eval de respostas exatas a cada N epocas (modo epocas)")
        p.add_argument("--probe-per-task", type=int, default=3,
                       help="n de perguntas por tarefa no probe (modo epocas)")
        p.add_argument("--resume", nargs="?", const="auto", default=None,
                       help="retoma o treino de um checkpoint (modelo+optimizer+"
                            "scheduler+epoca). Sem valor: usa o --out. Com valor: "
                            "caminho do checkpoint.")
        p.add_argument("--out", type=str, default=e.get("out", "modelos/especialista.pt"))
    elif known.mode == "slm":
        from minigpt.dialogues import DIALOGUES_FILE
        s = section(cfg, "slm")
        p.add_argument("--base", default=s.get("base", "Qwen/Qwen2.5-0.5B-Instruct"))
        p.add_argument("--datasets", nargs="+",
                       default=s.get("datasets", [DIALOGUES_FILE]))
        p.add_argument("--train-out", default=s.get("train_out", "slm_data/train.jsonl"))
        p.add_argument("--val-out", default=s.get("val_out", "slm_data/val.jsonl"))
        p.add_argument("--val-frac", type=float, default=s.get("val_frac", 0.05))
        p.add_argument("--build-dataset-only", action="store_true")
        p.add_argument("--out", default=s.get("finetuned_dir", "modelos/slm-qwen"))
        p.add_argument("--epochs", type=int, default=s.get("epochs", 2))
        p.add_argument("--lr", type=float, default=s.get("lr", 2e-4))
        p.add_argument("--batch-size", type=int, default=s.get("batch_size", 4))
        p.add_argument("--lora-r", type=int, default=s.get("lora_r", 16))
    else:
        g = section(cfg, "generalista")
        tasks_cfg = d.get("tasks")
        p.add_argument("--steps", type=int, default=g.get("steps", 3000))
        p.add_argument("--batch-size", type=int, default=g.get("batch_size", 64))
        p.add_argument("--block-size", type=int, default=m.get("block_size", 96))
        p.add_argument("--n-embd", type=int, default=m.get("n_embd", 96))
        p.add_argument("--n-head", type=int, default=m.get("n_head", 4))
        p.add_argument("--n-layer", type=int, default=m.get("n_layer", 4))
        p.add_argument("--dropout", type=float, default=m.get("dropout", 0.0))
        p.add_argument("--tie-embeddings", action=argparse.BooleanOptionalAction,
                       default=m.get("tie_embeddings", True))
        p.add_argument("--pos-encoding", choices=["learned", "rope"],
                       default=m.get("pos_encoding", "learned"))
        p.add_argument("--lr", type=float, default=g.get("lr", 3e-3))
        p.add_argument("--warmup", type=int, default=g.get("warmup", 200))
        p.add_argument("--weight-decay", type=float, default=g.get("weight_decay", 0.01))
        p.add_argument("--grad-clip", type=float, default=g.get("grad_clip", 1.0))
        p.add_argument("--n-alphabets", type=int, default=g.get("n_alphabets", 100))
        p.add_argument("--n-val-alphabets", type=int, default=g.get("n_val_alphabets", 15))
        p.add_argument("--min-length", type=int, default=d.get("min_length", 6))
        p.add_argument("--max-length", type=int, default=d.get("max_length", 10))
        p.add_argument("--loss", choices=["answer", "full"], default=g.get("loss", "full"))
        p.add_argument("--answer-weight", type=float, default=g.get("answer_weight", 10.0))
        p.add_argument("--aux-weight", type=float, default=g.get("aux_weight", 1.0))
        p.add_argument("--copy-head", action=argparse.BooleanOptionalAction,
                       default=m.get("copy_head", False))
        p.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction,
                       default=m.get("normalize_embeddings", False))
        p.add_argument("--tasks", type=str,
                       default=",".join(tasks_cfg) if isinstance(tasks_cfg, list) else tasks_cfg)
        p.add_argument("--on-the-fly", action=argparse.BooleanOptionalAction,
                       default=g.get("on_the_fly", True))
        p.add_argument("--json-style", choices=["compact", "list"],
                       default=d.get("json_style", "compact"))
        p.add_argument("--eval-interval", type=int, default=g.get("eval_interval", 250))
        p.add_argument("--eval-iters", type=int, default=g.get("eval_iters", 20))
        p.add_argument("--out", type=str, default=g.get("out", "modelos/especialista.pt"))
    return p.parse_args(argv)


def export_onnx_after(out_path, no_onnx, onnx_int8):
    if no_onnx:
        return
    import os
    try:
        from export_onnx import export_checkpoint_to_onnx
        onnx_out = os.path.splitext(out_path)[0] + ".onnx"
        export_checkpoint_to_onnx(out_path, onnx_out, int8=onnx_int8)
        print(f"ONNX reexportado em: {onnx_out}")
    except SystemExit as exc:
        print(f"aviso: falha ao exportar ONNX: {exc}")
    except Exception as exc:  # noqa: BLE001 - treino nao pode quebrar por causa do ONNX
        print(f"aviso: falha ao exportar ONNX: {exc}")


def encode_with_answer_span(tokenizer, jb, turns):
    """-> (ids, ans_lo, ans_hi).

    ByteTokenizer e consistente por prefixo (encode(a+b)==encode(a)+encode(b)),
    mas o trecho da resposta pode conter '\\n' (que vira o byte 10 -> token 10)
    e o tokenizador pode fundir o '\\n' final com o '<eos>'. Por isso a delimitacao
    e feita sobre a SEQUENCIA DE IDS, nao sobre texto re-decodificado.
    """
    prefix = f"<bos>{jb}\n"
    for q, a in turns[:-1]:
        prefix += f"Q: {q}\nA: {a}\n"
    last_q, last_a = turns[-1]
    head = prefix + f"Q: {last_q}\nA: "
    body = prefix + f"Q: {last_q}\nA: {last_a}"
    ids = tokenizer.encode(body + "\n<eos>")
    lo = len(tokenizer.encode(head))
    hi = len(tokenizer.encode(body))
    # sanity: o slice [lo:hi] deve ser exatamente a resposta
    if tokenizer.decode(ids[lo:hi]) != last_a:
        # fallback: busca a substring da resposta nos ids
        ans_ids = tokenizer.encode(last_a)
        for i in range(lo, len(ids) - len(ans_ids) + 1):
            if ids[i:i + len(ans_ids)] == ans_ids:
                lo, hi = i, i + len(ans_ids)
                break
    return ids, lo, hi


def build_weights(ids, lo, hi, answer_weight, stop_weight, prompt_weight):
    """Peso por posicao (a posicao t prediz o token t+1).

    A resposta ocupa ids[lo:hi]; a posicao que a PREDIZ e t = (lo-1 .. hi-2),
    porque ids[hi-1] e a ultima letra da resposta (predita em t=hi-2).
    O token ids[hi] e o '\\n' (stop_weight) e ids[hi+1] e '<eos>'.
    O prompt (JSON+pergunta) e prompt_weight (0 = sem supervisao).
    """
    w = [0.0] * (len(ids) - 1)
    for t in range(len(ids) - 1):
        nxt = t + 1
        if lo <= nxt <= hi - 1:
            w[t] = answer_weight
        elif nxt >= hi + 1:
            w[t] = stop_weight
        else:
            w[t] = prompt_weight
    # o '\\n' logo apos a resposta (nxt == hi) tambem e stop
    if 0 <= hi - 1 < len(w):
        w[hi - 1] = stop_weight
    return w


def build_curated_tensors(args, tasks):
    """Monta os tensores a partir do dataset CURADO (input/dialogos.jsonl).

    Cada entrada vira um exemplo no formato de treino:
        <bos>{"alfabeto":"..."}\nQ: pergunta\nA: resposta<eos>
    (entradas multi-turno continuam com "\nQ: ...\nA: ...")

    O cotidiano usa o mesmo cabecalho JSON (A-Z) para que o template do
    Ollama possa ser fixo e o modelo veja sempre a mesma estrutura.
    """
    from minigpt.data import EXTRA_CHARS
    from minigpt.tokenizer import ByteTokenizer

    entries = load_curated(args.dataset)
    default_alpha = load_base_alphabet(args.input)
    default_alpha = "".join(default_alpha)

    exemplos = []          # (json_block, q_primeiro, a_primeiro, categoria)
    itens = []             # (jb, turns) alinhado com exemplos
    raw = ""
    for e in entries:
        cat = e.get("task", "?")
        if tasks and cat not in tasks and cat not in ("general", "followup"):
            continue
        alpha = e.get("alphabet") or default_alpha
        jb = build_json_block(list(alpha))
        turns = list(zip(e["user"], e["assistant"]))
        exemplos.append((jb, turns[0][0], turns[0][1], cat))
        itens.append((jb, turns))
        text = f"<bos>{jb}\n"
        for q, a in turns:
            text += f"Q: {q}\nA: {a}\n"
        raw += text + "<eos>"

    if getattr(args, "tokenizer", "item") == "byte":
        tokenizer = ByteTokenizer()
    else:
        tokenizer = ItemChTokenizer.from_corpus(raw + default_alpha + EXTRA_CHARS,
                                                items=list(default_alpha))

    n = len(exemplos)
    x = torch.full((n, args.block_size), tokenizer.pad_id, dtype=torch.long)
    y = torch.full((n, args.block_size), -100, dtype=torch.long)
    w = torch.zeros((n, args.block_size))
    for r, (jb, turns) in enumerate(itens):
        ids, lo, hi = encode_with_answer_span(tokenizer, jb, turns)
        if len(ids) > args.block_size:
            raise SystemExit(
                f"Exemplo com {len(ids)} tokens excede block_size={args.block_size}. "
                "Aumente --block-size.")
        x[r, :len(ids)] = torch.tensor(ids)
        ww = build_weights(ids, lo, hi, args.answer_weight, args.stop_weight,
                           args.prompt_weight)
        for t in range(len(ids) - 1):
            y[r, t] = ids[t + 1]
            w[r, t] = ww[t]
    cats = [c for (_jb, _q, _a, c) in exemplos]
    return tokenizer, x, y, w, cats, exemplos


def build_especialista_tensors(args, tasks):
    """Monta (base, alfabetos, tokenizer, x, y, w, cats, exemplos) do especialista."""
    from minigpt.data import EXTRA_CHARS
    from minigpt.tokenizer import ByteTokenizer

    base = load_base_alphabet(args.input)
    alphabets = specialist_alphabets(base, args.n_alphabets, args.min_length, args.max_length, args.seed)

    exemplos = []  # (jb, pergunta, resposta, categoria)
    raw = ""
    for a in alphabets:
        jb = build_json_block(a)
        for cat, q, ans in make_examples(a, tasks):
            exemplos.append((jb, q, ans, cat))
            raw += format_prefix(jb, q) + ans + "<eos>"
    if getattr(args, "tokenizer", "item") == "byte":
        # byte-level: nenhum char e desconhecido (acentos, emojis, etc.) e a
        # exportacao para GGUF/Ollama e 1:1 (nomes ja no byte-encode GPT-2)
        tokenizer = ByteTokenizer()
    else:
        tokenizer = ItemChTokenizer.from_corpus(raw + "".join(base) + EXTRA_CHARS, items=base)

    n = len(exemplos)
    x = torch.full((n, args.block_size), tokenizer.pad_id, dtype=torch.long)
    y = torch.full((n, args.block_size), -100, dtype=torch.long)
    w = torch.zeros((n, args.block_size))
    for r, (jb, q, ans, _cat) in enumerate(exemplos):
        ids, mask = encode_example(tokenizer, jb, q, ans)
        if len(ids) > args.block_size:
            raise SystemExit(f"Exemplo com {len(ids)} tokens excede block_size={args.block_size}.")
        x[r, :len(ids)] = torch.tensor(ids)
        for t in range(len(ids) - 1):
            is_ans = bool(mask[t + 1])
            y[r, t] = ids[t + 1]
            w[r, t] = args.answer_weight if is_ans else 1.0
    cats = [c for (_, _, _, c) in exemplos]
    return base, alphabets, tokenizer, x, y, w, cats, exemplos


def especialista_config(args, tasks, tokenizer, alphabets):
    return {
        "block_size": args.block_size,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
        "dropout": 0.0,
        "tie_embeddings": True,
        "pos_encoding": "learned",
        "copy_head": False,
        "normalize_embeddings": False,
        "vocab_size": tokenizer.vocab_size,
        "json_style": "compact",
        "min_length": args.min_length,
        "max_length": args.max_length,
        "tasks": tasks,
        "specialist": True,
        "train_alphabets": ["".join(a) for a in alphabets],
    }


def build_intents_tensors(args, tasks, seed=None, exclude=None):
    """Dataset v2: intents curados x alfabetos (respostas verificadas).

    `seed` permite regenerar os alfabetos a cada epoca (--fresh-alphabets);
    `exclude` = alfabetos proibidos no treino (disjuncao com o split de eval).
    """
    from minigpt.curated_dataset import (build_training_examples, load_intents,
                                         make_alphabets)
    from minigpt.data import build_json_block
    from minigpt.tokenizer import ByteTokenizer

    seed = args.seed if seed is None else seed
    intents = load_intents(args.intents)
    examples = build_training_examples(
        intents, n_alphabets=args.n_alphabets, seed=seed,
        daily_variants=not args.no_daily_variants,
        daily_repeat=args.n_daily_repeat,
        exclude_alphabets=exclude)

    alphabets = [a for a in make_alphabets(args.n_alphabets, seed)
                 if a not in set(exclude or ())]
    tokenizer = ByteTokenizer() if getattr(args, "tokenizer", "byte") == "byte" else None
    n = len(examples)
    x = torch.full((n, args.block_size), tokenizer.pad_id, dtype=torch.long)
    y = torch.full((n, args.block_size), -100, dtype=torch.long)
    w = torch.zeros((n, args.block_size))
    for r, (jb, turns, _cat, _kind) in enumerate(examples):
        ids, lo, hi = encode_with_answer_span(tokenizer, jb, turns)
        if len(ids) > args.block_size:
            raise SystemExit(f"Exemplo com {len(ids)} tokens excede block_size={args.block_size}. "
                             "Aumente --block-size.")
        x[r, :len(ids)] = torch.tensor(ids)
        ww = build_weights(ids, lo, hi, args.answer_weight, args.stop_weight,
                           args.prompt_weight)
        for t in range(len(ids) - 1):
            y[r, t] = ids[t + 1]
            w[r, t] = ww[t]
    cats = [c for (_jb, _t, c, _k) in examples]
    return tokenizer, x, y, w, cats, examples, alphabets


def prepare_especialista_data(args, tasks, exclude=None):
    """Dados do especialista: intents (v2) > dataset curado (v1) > templates."""
    if getattr(args, "intents", None):
        tokenizer, x, y, w, cats, exemplos, alphs = build_intents_tensors(
            args, tasks, exclude=exclude)
        base = load_base_alphabet(args.input)
        return base, alphs, tokenizer, x, y, w, cats, exemplos
    if getattr(args, "dataset", None):
        tokenizer, x, y, w, cats, exemplos = build_curated_tensors(args, tasks)
        base = load_base_alphabet(args.input)
        alph_set = sorted({e.get("alphabet") for e in load_curated(args.dataset)
                           if e.get("alphabet")})
        alphabets = alph_set or [base]
        return base, alphabets, tokenizer, x, y, w, cats, exemplos
    return build_especialista_tensors(args, tasks)


def run_especialista(args):
    if getattr(args, "num_threads", 0):
        torch.set_num_threads(args.num_threads)
    if getattr(args, "epochs", 0):
        run_especialista_epochs(args)
        return
    tasks = args.tasks.split(",")
    base, alphabets, tokenizer, x, y, w, cats, exemplos = prepare_especialista_data(args, tasks)
    n = len(exemplos)

    print(f"Letras-base: {len(base)} | alfabetos fixos: {len(alphabets)} | "
          f"exemplos: {n} | vocab: {tokenizer.vocab_size}")
    print(f"tarefas: {tasks}")

    torch.manual_seed(args.seed)
    model = TinyGPT(
        tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        tie_embeddings=True,
    )
    print(f"Parametros: {model.num_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(args.steps):
        ix = torch.randint(0, n, (args.batch_size,), generator=gen)
        _, _, loss = model(x[ix], targets=y[ix], weights=w[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 4) == 0:
            print(f"  step {step:4d}/{args.steps}  loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")

    config = especialista_config(args, tasks, tokenizer, alphabets)
    save_checkpoint(args.out, model, tokenizer, config)
    print(f"\nEspecialista salvo em: {args.out}  ({time.time()-t0:.0f}s)")
    print(f"Alfabetos conhecidos: {config['train_alphabets']}")
    export_onnx_after(args.out, args.no_onnx, args.onnx_int8)


def make_length_batches(lengths, batch_size, seed, epoch):
    """Lotes com exemplos de comprimento parecido (bucket), em ordem aleatoria.

    O tensor e padded at�� block_size, mas o comprimento real mediano e ~84 de
    256 -> ~60% do compute ia para o padding. Agrupando por comprimento, cada
    lote pode ser cortado no maximo do lote: ~2.5x mais passos no mesmo tempo
    (passos e o que faz o circuito de inducao emergir).
    """
    rng = random.Random(seed + epoch)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    idx.sort(key=lambda i: lengths[i])
    batches = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    rng.shuffle(batches)
    return batches


def run_especialista_epochs(args):
    """Treino por epocas: DataLoader (workers=0, tensor em RAM), tqdm, eval
    de respostas exatas e historico CSV (curva de aprendizado)."""
    import os
    from minigpt.utils import clean_answer

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    tasks = args.tasks.split(",")
    use_intents = bool(getattr(args, "intents", None))
    fresh = use_intents and getattr(args, "fresh_alphabets", True)

    # split de EVAL (alfabetos NAO vistos): FIXO, e o probe do treino.
    # Os mesmos alfabetos sao excluidos do treino -> sem leakage.
    probe_pool = None
    probe_tag = "treino"
    eval_alphabets = set()
    if use_intents:
        from minigpt.curated_dataset import (build_eval_examples, load_intents,
                                             make_unseen_alphabets)
        n_eval_alpha = max(4, args.n_alphabets // 2)
        ev = build_eval_examples(load_intents(args.intents),
                                 n_alphabets=n_eval_alpha,
                                 seed=args.seed + 777, per_task=args.probe_per_task)
        if ev:
            probe_pool = ev
            probe_tag = "alfabetos NAO vistos"
            eval_alphabets = set(make_unseen_alphabets(n_eval_alpha, args.seed + 777))

    def build_data(epoch):
        """Dados de uma epoca. Com --fresh-alphabets cada epoca usa OUTRO
        conjunto de alfabetos: nenhum par (pergunta,resposta) se repete, logo
        memorizar deixa de ser uma opcao viavel."""
        if not fresh:
            return prepare_especialista_data(args, tasks, exclude=eval_alphabets)
        a2 = argparse.Namespace(**vars(args))
        a2.seed = args.seed + 1000 * (epoch - 1)
        return prepare_especialista_data(a2, tasks, exclude=eval_alphabets)

    start_epoch = 1
    base, alphabets, tokenizer, x, y, w, cats, exemplos = build_data(start_epoch)
    n = len(exemplos)
    steps_per_epoch = (n + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    print(f"Letras-base: {len(base)} | alfabetos: {len(alphabets)} | "
          f"exemplos: {n} | vocab: {tokenizer.vocab_size}")
    if use_intents:
        print(f"dataset v2: intents {args.intents} x {args.n_alphabets} alfabetos "
              f"(respostas verificadas) | probe em alfabetos NAO vistos")
        print(f"  perda: prompt_weight={args.prompt_weight} answer_weight={args.answer_weight} "
              f"stop_weight={args.stop_weight} | "
              f"alfabetos frescos por epoca: {'sim' if fresh else 'nao'}")
    elif getattr(args, "dataset", None):
        print(f"dataset curado: {args.dataset} (alfabeto + cotidiano)")
    print(f"epocas: {args.epochs} | steps/epoca: {steps_per_epoch} | total steps: {total_steps}")
    print(f"tarefas: {tasks}")

    torch.manual_seed(args.seed)
    model = TinyGPT(
        tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        tie_embeddings=True,
    )
    print(f"Parametros: {model.num_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = make_scheduler(opt, total_steps, min(200, max(10, total_steps // 10)))

    # --resume: retoma modelo + optimizer + scheduler + epoca do checkpoint
    if getattr(args, "resume", None):
        path = args.out if args.resume == "auto" else args.resume
        if os.path.exists(path):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            if ck.get("optimizer"):
                opt.load_state_dict(ck["optimizer"])
            if ck.get("scheduler"):
                sched.load_state_dict(ck["scheduler"])
            start_epoch = int(ck.get("epoch", 0)) + 1
            print(f"RESUME: retomando de {path} na epoca {start_epoch}/{args.epochs} "
                  f"(optimizer={'sim' if ck.get('optimizer') else 'nao'})")
        else:
            print(f"aviso: --resume pedido mas {path} nao existe; comecando do zero")

    # workers=0: dataset ja esta em RAM (fetch <2ms); no Windows, workers>0
    # exige spawn e so adiciona overhead. Paralelismo real = threads do torch.
    lengths = (x != tokenizer.pad_id).sum(dim=1).tolist()

    # probe estratificado por tarefa (saida esperada das respostas)
    # se houver intents, o probe usa o conjunto de EVAL (alfabetos NAO vistos)
    probe_pool = exemplos
    probe_tag = "treino"
    if getattr(args, "intents", None):
        from minigpt.curated_dataset import build_eval_examples, load_intents
        ev = build_eval_examples(load_intents(args.intents),
                                 n_alphabets=max(4, args.n_alphabets // 2),
                                 seed=args.seed + 777, per_task=args.probe_per_task)
        if ev:
            probe_pool = ev
            probe_tag = "alfabetos NAO vistos"
    rng = random.Random(args.seed)
    if probe_pool is None:
        probe_pool = exemplos
        probe_tag = "treino"
    by_task = {}
    for i, c in enumerate([e[2] for e in probe_pool]):
        by_task.setdefault(c, []).append(i)
    probe = []
    for c in tasks:
        idxs = by_task.get(c, [])
        rng.shuffle(idxs)
        probe += idxs[:args.probe_per_task]
    if not probe:  # sem tarefas de alfabeto no probe -> usa geral
        probe = list(range(min(len(probe_pool), args.probe_per_task * 3)))

    def eval_probe():
        model.eval()
        ok = 0
        with torch.no_grad():
            for i in probe:
                jb, turns, _c, _k = probe_pool[i]
                q, ans = turns[0]
                prompt = f"<bos>{jb}\nQ: {q}\nA: "
                ids = tokenizer.encode(prompt)
                idx = torch.tensor([ids], dtype=torch.long)
                cap = max(1, min(120, model.block_size - len(ids)))
                out = model.generate(idx, max_new_tokens=cap, temperature=0,
                                     eos_id=tokenizer.eos_id)
                pred = clean_answer(tokenizer.decode(out[0, len(ids):].tolist()))
                ok += int(" ".join(pred.split()) == " ".join(ans.split()))
        model.train()
        return ok, len(probe)

    stem, _ext = os.path.splitext(args.out)
    hist_path = stem + "_history.csv"
    # ao retomar, o CSV e acrescido (nao sobrescrito) para manter a curva
    keep_hist = start_epoch > 1 and os.path.exists(hist_path)
    hist = open(hist_path, "a" if keep_hist else "w", encoding="utf-8")
    if not keep_hist:
        hist.write("epoch,loss,lr,acc_probe,n_probe,secs\n")

    use_tqdm = tqdm is not None and sys.stdout.isatty()
    t0 = time.time()
    step = 0
    print(f"  inicio: {n} exemplos, {steps_per_epoch} batches/epoca, "
          f"probe a cada {args.eval_every} epocas", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        if fresh:
            # alfabetos novos nesta epoca (cada um visto uma unica vez)
            base, alphabets, tokenizer, x, y, w, cats, exemplos = build_data(epoch)
            lengths = (x != tokenizer.pad_id).sum(dim=1).tolist()
        model.train()
        loss_sum, nb = 0.0, 0
        ep_t0 = time.time()
        # sem tty (log/background): heartbeat a cada ~20% da epoca p/ provar vida
        beat_every = max(1, steps_per_epoch // 5)
        batch_list = make_length_batches(lengths, args.batch_size, args.seed, epoch)
        batches = tqdm(batch_list, desc=f"epoca {epoch}/{args.epochs}",
                       leave=False, unit="batch") if use_tqdm else batch_list
        for b, bidx in enumerate(batches, start=1):
            # corta o bloco no comprimento maximo do lote (evita calcular o padding)
            T = max(lengths[i] for i in bidx)
            bx = x[bidx][:, :T]
            by = y[bidx][:, :T]
            bw = w[bidx][:, :T]
            _, _, loss = model(bx, targets=by, weights=bw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            loss_sum += loss.item()
            nb += 1
            if use_tqdm:
                batches.set_postfix(loss=f"{loss.item():.4f}")
            elif b % beat_every == 0 or b == steps_per_epoch:
                el = time.time() - ep_t0
                bps = b / max(el, 1e-6)
                eta_ep = (steps_per_epoch - b) / max(bps, 1e-6)
                left_ep = args.epochs - epoch
                eta_tot = eta_ep + left_ep * (el / b * steps_per_epoch)
                print(f"    epoca {epoch}/{args.epochs} batch {b}/{steps_per_epoch} "
                      f"loss {loss.item():.4f} {bps:.1f} batch/s "
                      f"eta_ep {eta_ep:.0f}s eta_total {eta_tot/60:.0f}min", flush=True)
        avg = loss_sum / max(1, nb)
        line = f"epoca {epoch:3d}/{args.epochs} loss {avg:.4f} lr {sched.get_last_lr()[0]:.2e}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ok, tot = eval_probe()
            acc = ok / tot
            line += f" | probe({probe_tag}) {ok}/{tot} = {acc:.1%}"
            hist.write(f"{epoch},{avg:.4f},{sched.get_last_lr()[0]:.2e},{acc:.4f},{tot},{time.time()-t0:.0f}\n")
            hist.flush()
            config = especialista_config(args, tasks, tokenizer, alphabets)
            save_checkpoint(args.out, model, tokenizer, config,
                            extra=_resume_state(epoch, opt, sched))
        else:
            hist.write(f"{epoch},{avg:.4f},{sched.get_last_lr()[0]:.2e},,,{time.time()-t0:.0f}\n")
        print(f"  {line}  ({time.time()-t0:.0f}s)", flush=True)

    hist.close()
    config = especialista_config(args, tasks, tokenizer, alphabets)
    save_checkpoint(args.out, model, tokenizer, config,
                    extra=_resume_state(args.epochs, opt, sched))
    print(f"\nEspecialista salvo em: {args.out}  ({time.time()-t0:.0f}s)")
    print(f"Historico: {hist_path}")
    export_onnx_after(args.out, args.no_onnx, args.onnx_int8)


def get_batch(x, y, w, s, batch_size, generator):
    ix = torch.randint(0, x.size(0), (batch_size,), generator=generator)
    return x[ix], y[ix], w[ix], s[ix]


@torch.no_grad()
def estimate_loss(model, x, y, w, s, batch_size, iters, generator):
    model.eval()
    losses = []
    for _ in range(iters):
        bx, by, bw, _bs = get_batch(x, y, w, s, batch_size, generator)
        _, _, loss = model(bx, targets=by, weights=bw)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def render_bar(step, total, loss, lr, width=30):
    frac = (step + 1) / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {step + 1:>6}/{total}  loss {loss:.4f}  lr {lr:.2e}")
    sys.stdout.flush()


def make_scheduler(optimizer, total, warmup):
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _resume_state(epoch, opt, sched):
    """Estado extra gravado no checkpoint para permitir `--resume`."""
    return {
        "epoch": int(epoch),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
    }


def run_generalista(args):
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device("cpu")

    print(f"Dispositivo: {device} (CUDA disponivel: {torch.cuda.is_available()})")

    categories = args.tasks.split(",") if args.tasks else None
    base = load_base_alphabet(args.input)
    print(f"Letras-base ({args.input}): {len(base)} -> {''.join(base[:12])}...")
    data = build_dataset(
        args.block_size,
        n_alphabets=args.n_alphabets,
        n_val_alphabets=args.n_val_alphabets,
        seed=args.seed,
        json_style=args.json_style,
        min_length=args.min_length,
        max_length=args.max_length,
        categories=categories,
        base=base,
        loss_mode=args.loss,
        answer_weight=args.answer_weight,
    )
    tokenizer = data["tokenizer"]
    train_x, train_y, train_w, train_s = data["train"]
    val_x, val_y, val_w, val_s = data["val"]

    max_len = int((train_x != tokenizer.pad_id).sum(1).max())
    print(
        f"Sequencia: maior exemplo = {max_len} tokens | block_size = {args.block_size} "
        f"({max_len / args.block_size:.0%} de ocupacao)"
    )

    mode = "on-the-fly (alfabetos novos por passo)" if args.on_the_fly else "dataset fixo"
    print(
        f"Vocab: {tokenizer.vocab_size} | modo: {mode} | loss: {args.loss} "
        f"(answer_weight={args.answer_weight}) | "
        f"val: {args.n_val_alphabets} alfabetos novos | tarefas: {categories or 'todas'}"
    )

    model = TinyGPT(
        tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        tie_embeddings=args.tie_embeddings,
        pos_encoding=args.pos_encoding,
        copy_head=args.copy_head,
        normalize_embeddings=args.normalize_embeddings,
    ).to(device)
    print(f"Parametros: {model.num_params:,} | tie_embeddings: {args.tie_embeddings} | "
          f"pos: {args.pos_encoding} | copy_head: {args.copy_head} | "
          f"norm_emb: {args.normalize_embeddings}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(optimizer, args.steps, args.warmup)

    config = {
        "block_size": args.block_size,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
        "dropout": args.dropout,
        "tie_embeddings": args.tie_embeddings,
        "pos_encoding": args.pos_encoding,
        "copy_head": args.copy_head,
        "normalize_embeddings": args.normalize_embeddings,
        "vocab_size": tokenizer.vocab_size,
        "json_style": args.json_style,
        "alphabet_length": args.max_length,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "n_alphabets": args.n_alphabets,
        "n_val_alphabets": args.n_val_alphabets,
        "on_the_fly": args.on_the_fly,
        "loss_mode": args.loss,
        "answer_weight": args.answer_weight,
        "aux_weight": args.aux_weight,
        "input": args.input,
        "seed": args.seed,
    }

    best_val = float("inf")
    py_rng = random.Random(args.seed + 1)
    t0 = time.time()
    for step in range(args.steps):
        if args.on_the_fly:
            x, y, w, s = sample_batch(
                tokenizer, args.batch_size, args.block_size,
                args.min_length, args.max_length, categories, py_rng,
                base=base, loss_mode=args.loss, answer_weight=args.answer_weight,
            )
        else:
            x, y, w, s = get_batch(train_x, train_y, train_w, train_s, args.batch_size, generator)
        _, _, loss = model(x, targets=y, weights=w, aux_src=s, aux_weight=args.aux_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.eval_interval == 0 or step == args.steps - 1:
            sys.stdout.write("\n")
            train_loss = estimate_loss(model, train_x, train_y, train_w, train_s, args.batch_size, args.eval_iters, generator)
            val_loss = estimate_loss(model, val_x, val_y, val_w, val_s, args.batch_size, args.eval_iters, generator)
            print(f"  [eval] treino {train_loss:.4f} | val (alfabetos novos, so tarefa) {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(args.out, model, tokenizer, config)

        render_bar(step, args.steps, loss.item(), scheduler.get_last_lr()[0])

    sys.stdout.write("\n")

    print(f"Tempo de treino: {time.time() - t0:.1f}s | melhor val loss: {best_val:.4f}")
    print(f"Melhor checkpoint salvo em: {args.out}")
    export_onnx_after(args.out, args.no_onnx, args.onnx_int8)

    results = evaluate_exact_match(
        model, tokenizer, data["val_prompts"], data["val_answers"], data["val_cats"], greedy=True
    )
    correct, total, per = summarize(results)
    print(f"\nAcuracia exata (greedy) em alfabetos NAO vistos: {correct}/{total} = {correct / total:.1%}")
    for group in TASK_GROUPS:
        c, n = per.get(group, (0, 0))
        if n:
            print(f"  {group:9s}: {c:4d}/{n:<4d} = {c / n:5.1%}")
    print("\nExemplos:")
    for prompt, gold, pred, ok, cat in results[:6]:
        m = re.search(r"Q:\s*(.*?)\s*\nA:", prompt, re.S)
        q = m.group(1) if m else prompt
        print(f"  [{'OK' if ok else 'XX'}] ({cat}) {q} -> {pred!r} (esperado {gold!r})")

    print("\nGeneralizacao por comprimento de alfabeto (exact match, greedy):")
    probe_lengths = sorted(
        set([args.min_length, args.max_length, args.max_length + 1, args.max_length + 2, args.max_length + 5])
    )
    for L in probe_lengths:
        if L > len(base):
            continue
        pr, ans, cs = val_prompts_for(
            args.n_alphabets, args.n_val_alphabets, args.seed, args.json_style,
            L, L, categories, base,
        )
        res = evaluate_exact_match(model, tokenizer, pr, ans, cs, greedy=True)
        c2, t2, _ = summarize(res)
        tag = "visto" if args.min_length <= L <= args.max_length else "NAO visto"
        print(f"  {L:2d} letras ({tag:9s}): {c2:4d}/{t2:<4d} = {c2 / t2:5.1%}")


def run_slm(args):
    import json as _json
    import os as _os

    system = load_config(args.config).get("slm", {}).get(
        "system", "Voce e um assistente especialista no alfabeto.")
    entries = []
    for path in args.datasets:
        entries += load_curated_alpha_everyday(path)
    print(f"Dataset curado: {len(entries)} dialogos de {args.datasets}")

    rng = random.Random(args.seed)
    rng.shuffle(entries)
    records = [{"messages": to_messages(e, system), "task": e.get("task", "?")} for e in entries]
    n_val = max(1, int(len(records) * args.val_frac))
    val, train = records[:n_val], records[n_val:]
    for recs, path in ((train, args.train_out), (val, args.val_out)):
        _dir = _os.path.dirname(path)
        if _dir:
            _os.makedirs(_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(_json.dumps(r, ensure_ascii=False) + "\n")
    print(f"SLM dataset: {len(train)} treino + {len(val)} val")
    print(f"salvo em: {args.train_out} e {args.val_out}")

    if args.build_dataset_only:
        print("\nSomente dataset (sem treino). Para treinar:")
        print("  pip install -r requirements-slm.txt")
        print(f"  python train.py --mode slm --base {args.base}")
        return

    try:
        import torch as _torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from transformers import DataCollatorForLanguageModeling, Trainer
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise SystemExit(
            "Dependencias de treino ausentes. Instale com:\n"
            "  pip install -r requirements-slm.txt\n"
            f"Detalhe: {exc}"
        )

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_dataset("json", data_files={"train": args.train_out, "validation": args.val_out})

    def map_fn(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        ids = tok(text, truncation=True, max_length=1024)
        ids["labels"] = list(ids["input_ids"])
        return ids

    ds_tok = ds.map(map_fn, remove_columns=ds["train"].column_names)
    model = AutoModelForCausalLM.from_pretrained(args.base, trust_remote_code=True)
    lora = LoraConfig(r=args.lora_r, lora_alpha=32,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    targs = TrainingArguments(output_dir=args.out, num_train_epochs=args.epochs,
                              per_device_train_batch_size=args.batch_size,
                              learning_rate=args.lr, logging_steps=20,
                              save_steps=200, fp16=_torch.cuda.is_available(),
                              report_to="none")
    trainer = Trainer(model=model, args=targs, train_dataset=ds_tok["train"],
                      eval_dataset=ds_tok["validation"],
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"\nSLM salvo em: {args.out}")
    print("Proximo passo:")
    print(f"  python export_gguf.py --from-hf {args.out} --out modelos/slm.gguf")


def load_curated_alpha_everyday(path):
    from minigpt.dialogues import load_jsonl
    return load_jsonl(path)


def main():
    args = parse_args()
    if args.mode == "especialista":
        run_especialista(args)
    elif args.mode == "slm":
        run_slm(args)
    else:
        run_generalista(args)


if __name__ == "__main__":
    main()
