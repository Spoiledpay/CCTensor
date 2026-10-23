"""Dataset v2: intents curados x alfabetos (respostas SEMPRE verificadas).

Conceito:
  - A CURADORIA esta nos intents (input/intents.jsonl): as perguntas sao
    escritas a mao, com placeholders {L} (letra) e {k} (numero).
  - O que e varia sao os ALFABETOS: o modelo e obrigado a LER o JSON do
    alfabeto para responder, em vez de decorar pares (pergunta, resposta).
    Cada instancia tem a resposta RECOMPUTADA e conferida -> zero ruido.

Splits:
  - treino: alfabetos "vistos" (A-Z, invertido, e N permutacoes com seed S)
  - eval:   alfabetos "NAO vistos" (permutacoes com outra seed) + perguntas
            ineditas -> mede generalizacao de verdade.
"""

import json
import os
import random

from .dialogues import expected_answer

ALPHA_AZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
INTENTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "input", "intents.jsonl")


def load_intents(path=INTENTS_FILE):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def make_alphabets(n, seed, min_len=10, max_len=26, include_canonical=True):
    """Alfabetos de treino: canonico + inverso + permutacoes + subconjuntos."""
    rng = random.Random(seed)
    out = []
    if include_canonical:
        out.append(ALPHA_AZ)
        out.append(ALPHA_AZ[::-1])
    while len(out) < n + 2:
        size = rng.randint(min_len, max_len)
        letters = rng.sample(ALPHA_AZ, size)
        rng.shuffle(letters)
        out.append("".join(letters))
    seen, uniq = set(), []
    for a in out:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


def make_unseen_alphabets(n, seed, min_len=10, max_len=26):
    """Alfabetos de EVAL: gerados com outra seed (nao vistos no treino)."""
    return make_alphabets(n, seed, min_len, max_len, include_canonical=False)


def _instantiate(intent, alphabet, rng):
    """(pergunta, resposta) de um intent de alfabeto sobre um alfabeto.
    A resposta e recomputada e conferida; None se nao for valida."""
    task = intent["task"]
    n = len(alphabet)
    letter = None
    k = None
    variant = None

    if task in ("next", "prev"):
        letter = rng.choice(alphabet[:-1] if task == "next" else alphabet[1:])
    elif task == "position":
        letter = rng.choice(alphabet)
    elif task == "contains":
        letter = rng.choice(alphabet) if rng.random() < 0.5 else \
            rng.choice([c for c in ALPHA_AZ if c not in alphabet] or alphabet)
    elif task in ("first_n", "last_n"):
        k = rng.randint(1, min(10, n))
    elif task == "vowels":
        variant = "count" if "quantas" in intent["templates"][0].lower() else "list"
    elif task == "consonants":
        variant = "count" if "quantas" in intent["templates"][0].lower() else "list"

    for tpl in intent["templates"]:
        q = tpl
        if "{L}" in q:
            if letter is None:
                continue
            q = q.replace("{L}", letter)
        if "{k}" in q:
            if k is None:
                continue
            q = q.replace("{k}", str(k))
        try:
            ans = expected_answer(task, alphabet, letter, k, variant)
        except Exception:
            continue
        return q, ans
    return None


def _daily_variants(templates, rng):
    """Variantes de superficie para o cotidiano: caixa e pontuacao.

    A curadoria define a INTENCAO; aqui so variamos a forma como o utilizador
    a escreve (minusculas, sem '!'), para o modelo nao depender da pontuacao.
    """
    out = []
    for t in templates:
        out.append(t)
        low = t[:1].lower() + t[1:]
        if low != t:
            out.append(low)
        if t.endswith("!"):
            out.append(t[:-1])
        if t.endswith("?"):
            out.append(t[:-1])
    seen, uniq = set(), []
    for t in out:
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def build_training_examples(intents, n_alphabets=40, seed=1337, daily_variants=True,
                            daily_repeat=1, exclude_alphabets=None):
    """-> lista de (json_block, [(pergunta, resposta), ...], task, kind).

    exclude_alphabets: alfabetos que NAO podem aparecer no treino (disjuncao
    com o split de eval -> sem leakage). Sao as permutacoes de outra seed.
    """
    from .data import build_json_block

    rng = random.Random(seed)
    alphabets = make_alphabets(n_alphabets, seed)
    excluded = set(exclude_alphabets or ())
    examples = []

    for intent in intents:
        if intent["kind"] == "computed":
            for alpha in alphabets:
                if alpha in excluded:
                    continue
                inst = _instantiate(intent, alpha, rng)
                if inst is None:
                    continue
                q, a = inst
                jb = build_json_block(list(alpha))
                examples.append((jb, [(q, a)], intent["task"], "alpha"))
        else:
            variants = (_daily_variants(intent["templates"], rng) if daily_variants
                        else intent["templates"])
            for _ in range(daily_repeat):
                for v in variants:
                    jb = build_json_block(list(ALPHA_AZ))
                    examples.append((jb, [(v, intent["answer"])], "general", "daily"))
    return examples


def build_eval_examples(intents, n_alphabets=12, seed=99991, per_task=2,
                         unseen_phrasings=True):
    """Avaliacao: alfabetos NAO vistos + (opcional) perguntas ineditas."""
    from .data import build_json_block

    rng = random.Random(seed)
    alphabets = make_unseen_alphabets(n_alphabets, seed)
    examples = []
    for intent in intents:
        if intent["kind"] != "computed":
            continue
        for alpha in alphabets:
            for _ in range(per_task):
                inst = _instantiate(intent, alpha, rng)
                if inst is None:
                    continue
                q, a = inst
                jb = build_json_block(list(alpha))
                examples.append((jb, [(q, a)], intent["task"], "unseen_alphabet"))
    return examples
