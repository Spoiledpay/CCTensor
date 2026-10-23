"""Geracao do corpus e das tarefas sobre um alfabeto definido em JSON.

Formato curto de cada exemplo (treino causal):

    <bos>{"alfabeto":"MNYBI"}
    Q: Depois de M?
    A: N<eos>

A loss pode ser:
  - "answer": apenas nos tokens da resposta (Recomendado didaticamente);
  - "full":   next-token em toda a sequencia (LM denso).

As letras-base vem de um arquivo JSON (default: input/alfabeto.json).

Para forcar o modelo a LER o alfabeto (em vez de decorar), o treino gera
alfabetos aleatorios novos a cada passo (dados infinitos) e valida em um
conjunto fixo de alfabetos nao vistos.
"""

import json
import os
import random

import torch

from .tokenizer import CharTokenizer, ItemChTokenizer

DEFAULT_INPUT = os.path.join("input", "alfabeto.json")
FALLBACK_ALPHABET = [chr(ord("A") + i) for i in range(26)]
# Caracteres que podem aparecer na resposta mas nao nas letras-base:
# digitos de `count`/`position` e separadores de `last_n`.
EXTRA_CHARS = "0123456789, "

# Presets usados pelo chat.
A_TO_Z = FALLBACK_ALPHABET
ALPHABETS = {
    "canonical": A_TO_Z,
    "reversed": list(reversed(A_TO_Z)),
    "qwerty": list("QWERTYUIOPASDFGHJKLZXCVBNM"),
    "first10": A_TO_Z[:10],
}

TASK_GROUPS = ("count", "first", "last", "next", "prev", "position", "last_n", "show", "first_n")


def load_base_alphabet(path=DEFAULT_INPUT):
    """Le as letras-base de um JSON ({"alfabeto": [...]} ou lista)."""
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        letters = obj.get("alfabeto") if isinstance(obj, dict) else obj
        letters = [str(x) for x in letters]
        if letters:
            return letters
    return list(FALLBACK_ALPHABET)


def build_json_block(alphabet, style="compact"):
    alphabet = list(alphabet)
    if style == "list":
        return "JSON: " + json.dumps({"alfabeto": alphabet}, ensure_ascii=False)
    return '{"alfabeto":"' + "".join(alphabet) + '"}'


def _case_variants(q):
    """Variantes de caixa e pontuacao de uma pergunta.

    Gera: original, 1ª letra minuscula, letras soltas minusculas
    ("Depois de C?" -> "Depois de c?") e ambas — cada uma com e sem o "?"
    final ("quantas letras tem" vale como "Quantas letras tem?").
    O modelo aprende que caixa/ponto nao mudam a intencao.
    """
    import re

    out = [q]
    f = q[:1].lower() + q[1:]
    if f != q:
        out.append(f)
    g = re.sub(r"\b([A-Z])\?", lambda m: m.group(1).lower() + "?", q)
    if g not in out:
        out.append(g)
    h = g[:1].lower() + g[1:]
    if h not in out:
        out.append(h)
    for v in list(out):
        if v.endswith("?") and v[:-1] not in out:
            out.append(v[:-1])
    return out


def make_examples(alphabet, categories=None, variants=True):
    """Retorna (categoria, pergunta, resposta) com perguntas curtas.

    FASE 1: cada intencao tem a forma canonica (primeira da lista, usada pelo
    `canonicalize_question`) + parafrases em PT sem acento. O modelo treina em
    todas e aprende a entender frase livre no dominio do alfabeto, sem regex.
    Com `variants=True` (default, treino) inclui variantes de caixa;
    `variants=False` retorna so as formas canonicas+parafrases (p/ /exemplos).
    """
    letters = list(alphabet)
    n = len(letters)
    out = []
    for q in ("Total?", "Quantas letras tem?", "Quantas letras sao?",
              "Quantas letras tem o alfabeto?", "Quantas letras sao no alfabeto?",
              "Quantas letras ha no alfabeto?", "Qual o total de letras?",
              "Qual o tamanho do alfabeto?"):
        out.append(("count", q, str(n)))
    for q in ("Primeira?", "Qual a primeira letra?", "Mostre a primeira letra?"):
        out.append(("first", q, letters[0]))
    for q in ("Ultima?", "Qual a ultima letra?", "Mostre a ultima letra?"):
        out.append(("last", q, letters[-1]))
    for q in ("Mostre o alfabeto?", "Qual e o alfabeto?", "Liste o alfabeto?",
              "Me mostre o alfabeto?"):
        out.append(("show", q, ", ".join(letters)))
    for k in range(1, min(10, n) + 1):
        out.append(("first_n", f"Mostre as {k} primeiras?", ", ".join(letters[:k])))
        out.append(("first_n", f"Quais sao as {k} primeiras letras?", ", ".join(letters[:k])))
    for i in range(n - 1):
        out.append(("next", f"Depois de {letters[i]}?", letters[i + 1]))
        out.append(("next", f"Qual letra vem depois de {letters[i]}?", letters[i + 1]))
        out.append(("next", f"Qual vem depois do {letters[i]}?", letters[i + 1]))
    for i in range(1, n):
        out.append(("prev", f"Antes de {letters[i]}?", letters[i - 1]))
        out.append(("prev", f"Qual letra vem antes de {letters[i]}?", letters[i - 1]))
        out.append(("prev", f"Qual vem antes do {letters[i]}?", letters[i - 1]))
    for i, ch in enumerate(letters):
        out.append(("position", f"Posicao de {ch}?", str(i + 1)))
        out.append(("position", f"Qual a posicao da letra {ch}?", str(i + 1)))
    for k in range(1, min(10, n) + 1):
        out.append(("last_n", f"Ultimas {k}?", ", ".join(letters[-k:])))
        out.append(("last_n", f"Quais sao as {k} ultimas letras?", ", ".join(letters[-k:])))
        out.append(("last_n", f"Me mostre as {k} ultimas?", ", ".join(letters[-k:])))
    # variantes COM ACENTOS (o usuario digita em portugues acentuado; com o
    # tokenizador byte-level todo char e valido e nada e descartado)
    for q in ("Qual a última letra?", "Mostre a última letra?"):
        out.append(("last", q, letters[-1]))
    for i, ch in enumerate(letters):
        out.append(("position", f"Posição de {ch}?", str(i + 1)))
        out.append(("position", f"Qual a posição da letra {ch}?", str(i + 1)))
    for k in range(1, min(10, n) + 1):
        out.append(("last_n", f"Últimas {k}?", ", ".join(letters[-k:])))
        out.append(("last_n", f"Quais são as {k} últimas letras?", ", ".join(letters[-k:])))
    for i in range(n - 1):
        out.append(("next", f"Qual letra vem depois de {letters[i]}?", letters[i + 1]))
        out.append(("prev", f"Qual letra vem antes de {letters[i]}?", letters[i - 1]))
    for ch in letters:
        out.append(("contains", f"O alfabeto contém a letra {ch}?", "Sim."))
    out.append(("vowels", "Quais são as vogais?", ", ".join(c for c in letters if c in "AEIOU")))
    out.append(("consonants", "Quantas consoantes?", str(sum(1 for c in letters if c not in "AEIOU"))))
    out.append(("reverse", "Mostre ao contrário?", ", ".join(reversed(letters))))
    if variants:
        expanded = []
        for cat, q, a in out:
            for v in _case_variants(q):
                expanded.append((cat, v, a))
        out = expanded
    if categories:
        allowed = set(categories)
        out = [ex for ex in out if ex[0] in allowed]
    return out


def format_prefix(json_block, question):
    return f"<bos>{json_block}\nQ: {question}\nA: "


def canonicalize_question(raw):
    """Traduz frase livre em PT para a pergunta canonica do treino.

    O modelo so entende as formas curtas de `make_examples`
    ("Total?", "Depois de C?"...). Frase livre ("quantas letras tem?",
    "mostre a ultima letra") vai literal ao prompt e quebra (char fora do
    vocab) ou alucina. Esta funcao faz o papel de NLU por regras/regex:
    retorna a pergunta canonica ou None (nao entendido).

    Exemplos:
        "quantas letras sao no alfabeto?" -> "Total?"
        "mostre a ultima letra"            -> "Ultima?"
        "qual vem depois do d?"            -> "Depois de D?"
        "ultimas 4"                        -> "Ultimas 4?"
    """
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", raw or "").encode("ascii", "ignore").decode()
    text = text.lower().strip()

    def letter_after(pattern):
        m = re.search(pattern, text)
        if m:
            for g in m.groups():
                if g and len(g) == 1 and g.isalpha():
                    return g.upper()
        return None

    # 1) letra especifica: depois / antes / posicao (mais especifico primeiro)
    ch = letter_after(r"depois\s+d[eo]\s+([a-z])")
    if ch:
        return f"Depois de {ch}?"
    ch = letter_after(r"antes\s+d[eo]\s+([a-z])")
    if ch:
        return f"Antes de {ch}?"
    # "qual a posicao da letra C?" -> pega a letra apos 'letra' (antes do
    # padrao generico, que capturaria o 'l' de "letra" em "posicao da letra")
    if "posicao" in text:
        m = re.search(r"letra\s+([a-z])", text)
        if m:
            return f"Posicao de {m.group(1).upper()}?"
    ch = letter_after(r"posicao\s+d[eaos]*\s*([a-z])")
    if ch:
        return f"Posicao de {ch}?"

    # 2) N primeiras / N ultimas (com numero)
    m = re.search(r"(\d{1,2})\s*primeiras?", text)
    if m:
        k = max(1, min(10, int(m.group(1))))
        return f"Mostre as {k} primeiras?"
    m = re.search(r"ultimas?\s*(\d{1,2})", text) or re.search(r"(\d{1,2})\s*ultimas?", text)
    if m:
        k = max(1, min(10, int(m.group(1))))
        return f"Ultimas {k}?"

    # 3) primeira / ultima (singular)
    if re.search(r"\bprimeira\b", text):
        return "Primeira?"
    if re.search(r"\bultima\b", text) and "ultimas" not in text:
        return "Ultima?"

    # 4) total / contagem
    if re.search(r"\b(quant|total|qtd|tamanho|comprimento|todas?\s+as\s+letras\s+(tem|ha|existem|sao))", text):
        return "Total?"

    # 5) mostrar o alfabeto inteiro
    if re.search(r"(mostr|list|exib|tod[oa]s?\s+as\s+letras|alfabeto\s+completo|qual\s+e\s+o\s+alfabeto)", text):
        return "Mostre o alfabeto?"

    # 6) "ultimas" sem numero: assume 3 e avisa (o treino exige N)
    if re.search(r"\bultimas\b", text):
        return "Ultimas 3?"

    # 7) ja esta na forma canonica? (ex.: "Total?", "Depois de C?")
    for _cat, q, _a in make_examples(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), variants=False):
        if text == q.lower():
            return q

    return None


def encode_example(tokenizer, json_block, question, answer):
    prefix = format_prefix(json_block, question)
    full = prefix + answer + "<eos>"
    ids = tokenizer.encode(full)
    prefix_len = len(tokenizer.encode(prefix))
    mask = [0] * prefix_len + [1] * (len(ids) - prefix_len)
    return ids, mask


def source_position(json_block, answer):
    """Indice, na sequencia codificada, do token-fonte que deve ser COPIADO.

    O bloco JSON comeca logo apos <bos> (1 token), entao o offset do caractere
    no bloco e o proprio indice na sequencia (+1). Vale para respostas de 1
    letra (next/prev/first/last). Retorna -1 quando nao se aplica.
    """
    if len(answer) != 1 or not answer.isalpha():
        return -1
    i = json_block.find(answer)
    return -1 if i < 0 else 1 + i


def build_tensors(tokenizer, examples, block_size, loss_mode="answer", answer_weight=1.0,
                  aux=False):
    """examples: lista de (json_block, pergunta, resposta).

    Retorna (inputs, targets, weights, aux_src):
      - targets: id seguinte a prever; -100 onde nao ha supervisao;
      - weights: peso por token supervisionado. Tokens da RESPOSTA recebem
        `answer_weight` (forca a resposta); o resto do texto recebe 1.0
        (curriculo denso). Com loss_mode="answer" so a resposta e supervisionada;
      - aux_src: posicao-fonte a copiar em cada posicao (pointer loss); -1 ignora.
    """
    n = len(examples)
    inputs = torch.full((n, block_size), tokenizer.pad_id, dtype=torch.long)
    targets = torch.full((n, block_size), -100, dtype=torch.long)
    weights = torch.zeros((n, block_size), dtype=torch.float)
    aux_src = torch.full((n, block_size), -1, dtype=torch.long)
    for row, (json_block, question, answer) in enumerate(examples):
        ids, mask = encode_example(tokenizer, json_block, question, answer)
        if len(ids) > block_size:
            raise ValueError(
                f"Exemplo com {len(ids)} tokens excede block_size={block_size}. "
                "Aumente --block-size."
            )
        inputs[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        for t in range(len(ids) - 1):
            is_answer = bool(mask[t + 1])
            if loss_mode == "full" or is_answer:
                targets[row, t] = ids[t + 1]
                weights[row, t] = answer_weight if is_answer else 1.0
        if aux and answer in tokenizer.stoi:
            ans_id = tokenizer.stoi[answer]
            prefix_len = len(mask) - sum(mask)
            for p in range(1, prefix_len):
                if ids[p] == ans_id:
                    aux_src[row, prefix_len - 1] = p
                    break
    return inputs, targets, weights, aux_src


def random_alphabets(count, seed, min_length=6, max_length=10, base=None):
    """`count` alfabetos distintos: subconjuntos aleatorios de `base`, embaralhados."""
    base = list(base) if base else list(FALLBACK_ALPHABET)
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < count:
        size = rng.randint(min_length, max_length)
        letters = rng.sample(base, size)
        rng.shuffle(letters)
        key = tuple(letters)
        if key in seen:
            continue
        seen.add(key)
        out.append(letters)
    return out


def make_alphabets(n_train, n_val, seed, min_length=6, max_length=10, base=None):
    alphas = random_alphabets(n_train + n_val, seed, min_length, max_length, base)
    return alphas[n_val:], alphas[:n_val]


def build_dataset(
    block_size,
    n_alphabets=100,
    n_val_alphabets=15,
    seed=1337,
    json_style="compact",
    min_length=6,
    max_length=10,
    categories=None,
    base=None,
    loss_mode="answer",
    answer_weight=1.0,
):
    train_alphas, val_alphas = make_alphabets(
        n_alphabets, n_val_alphabets, seed, min_length, max_length, base
    )

    train_ex, raw_text = [], ""
    for alphabet in train_alphas:
        jb = build_json_block(alphabet, json_style)
        for _cat, question, answer in make_examples(alphabet, categories):
            train_ex.append((jb, question, answer))
            raw_text += format_prefix(jb, question) + answer + "<eos>"

    tokenizer = ItemChTokenizer.from_corpus(
        raw_text + "".join(base or FALLBACK_ALPHABET) + EXTRA_CHARS, items=base or FALLBACK_ALPHABET
    )
    train_inputs, train_targets, train_weights, train_aux = build_tensors(
        tokenizer, train_ex, block_size, loss_mode, answer_weight, aux=True
    )

    val_ex = []
    for alphabet in val_alphas:
        jb = build_json_block(alphabet, json_style)
        for _cat, question, answer in make_examples(alphabet, categories):
            val_ex.append((jb, question, answer))
    val_inputs, val_targets, val_weights, val_aux = build_tensors(
        tokenizer, val_ex, block_size, loss_mode, answer_weight, aux=True
    )

    val_prompts, val_answers, val_cats = val_prompts_for(
        n_alphabets, n_val_alphabets, seed, json_style, min_length, max_length, categories, base
    )

    return {
        "tokenizer": tokenizer,
        "block_size": block_size,
        "json_style": json_style,
        "min_length": min_length,
        "max_length": max_length,
        "train": (train_inputs, train_targets, train_weights, train_aux),
        "val": (val_inputs, val_targets, val_weights, val_aux),
        "val_prompts": val_prompts,
        "val_answers": val_answers,
        "val_cats": val_cats,
        "n_alphabets": n_alphabets,
        "n_val_alphabets": n_val_alphabets,
        "seed": seed,
    }


def val_prompts_for(n_train, n_val, seed, json_style, min_length=6, max_length=10, categories=None, base=None):
    _train_alphas, val_alphas = make_alphabets(n_train, n_val, seed, min_length, max_length, base)
    prompts, answers, cats = [], [], []
    for alphabet in val_alphas:
        jb = build_json_block(alphabet, json_style)
        for cat, question, answer in make_examples(alphabet, categories):
            prompts.append(format_prefix(jb, question))
            answers.append(answer)
            cats.append(cat)
    return prompts, answers, cats


def prompts_for_question(alphabet, question, json_style="compact"):
    return format_prefix(build_json_block(alphabet, json_style), question)


def build_tokenizer(n_alphabets=20, seed=0, categories=None, json_style="compact",
                    min_length=6, max_length=10, base=None):
    alphas, _ = make_alphabets(n_alphabets, 0, seed, min_length, max_length, base)
    text = ""
    for alphabet in alphas:
        jb = build_json_block(alphabet, json_style)
        for _cat, question, answer in make_examples(alphabet, categories):
            text += format_prefix(jb, question) + answer + "<eos>"
    return ItemChTokenizer.from_corpus(
        text + "".join(base or FALLBACK_ALPHABET) + EXTRA_CHARS, items=base or FALLBACK_ALPHABET
    )


def sample_batch(tokenizer, batch_size, block_size, min_length, max_length, categories, py_rng,
                 base=None, loss_mode="answer", answer_weight=1.0, aux=True):
    """Gera um batch com alfabetos ALEATORIOS novos a cada chamada (dados infinitos)."""
    base = list(base) if base else list(FALLBACK_ALPHABET)
    examples = []
    for _ in range(batch_size):
        size = py_rng.randint(min_length, max_length)
        letters = py_rng.sample(base, size)
        py_rng.shuffle(letters)
        jb = build_json_block(letters)
        _cat, question, answer = py_rng.choice(make_examples(letters, categories))
        examples.append((jb, question, answer))
    return build_tensors(tokenizer, examples, block_size, loss_mode, answer_weight, aux=aux)
