"""Dataset de conversa em ingles do cotidiano (sem alfabeto).

Fonte unica de verdade: input/chat_en.jsonl, uma conversa por linha:

    {"id": "greet_01", "topic": "greetings",
     "turns": [{"user": ["Hi!", "Hello!"], "assistant": ["Hi! How are you?"]}]}

Cada turno tem LISTAS de variantes de usuario e de assistente. O builder:

  - sorteia uma variante de cada lado (combinacoes diferentes por exemplo);
  - aplica variantes de superficie SO no texto do usuario (caixa/pontuacao),
    para o modelo ficar robusto a como a pessoa escreve;
  - renderiza o template de chat e supervisiona SO as respostas;
  - separa treino/validacao por ID de intencao (validacao = intencoes NUNCA
    vistas -> mede generalizacao, nao memorizacao).

Template (mesmo no treino e na inferencia):

    <bos>User: {u1}\\nAssistant: {a1}\\nUser: {u2}\\nAssistant: {a2}\\n<eos>
"""

import json
import os
import random

from .tokenizer import ByteTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAT_FILE = os.path.join(ROOT, "input", "chat_en.jsonl")

USER_TAG = "User: "
ASSISTANT_TAG = "Assistant: "


# ------------------------------------------------------------------ carga
def load_catalog(path=CHAT_FILE):
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_line"] = ln
            out.append(obj)
    return out


def _check_entry(e):
    where = f"{e.get('id', '?')} (linha {e.get('_line', '?')})"
    for key in ("id", "topic", "turns"):
        if key not in e:
            raise SystemExit(f"entrada sem '{key}': {where}")
    if not isinstance(e["turns"], list) or not e["turns"]:
        raise SystemExit(f"'turns' vazio/invalido: {where}")
    for i, t in enumerate(e["turns"]):
        if "user" not in t or "assistant" not in t:
            raise SystemExit(f"turno {i} sem user/assistant: {where}")
        for side in ("user", "assistant"):
            v = t[side]
            if not isinstance(v, list) or not v:
                raise SystemExit(f"turno {i} '{side}' deve ser lista nao vazia: {where}")
            for s in v:
                if not isinstance(s, str) or not s.strip():
                    raise SystemExit(f"turno {i} '{side}' tem texto vazio: {where}")


def validate(path=CHAT_FILE):
    """Valida esquema e unicidade de IDs. Devolve (n_entradas, n_turnos, topicos)."""
    cat = load_catalog(path)
    if not cat:
        raise SystemExit(f"dataset vazio: {path}")
    seen = set()
    for e in cat:
        _check_entry(e)
        if e["id"] in seen:
            raise SystemExit(f"id duplicado: {e['id']}")
        seen.add(e["id"])
    topics = sorted({e["topic"] for e in cat})
    n_turns = sum(len(e["turns"]) for e in cat)
    n_ut = sum(len(t["user"]) for e in cat for t in e["turns"])
    n_at = sum(len(t["assistant"]) for e in cat for t in e["turns"])
    return len(cat), n_turns, topics, n_ut, n_at


# ------------------------------------------------------------------ variantes
def _user_variants(text):
    """Variantes de superficie do lado do USUARIO (o assistente fica canonico)."""
    out = [text]
    low = text[:1].lower() + text[1:]
    if low not in out:
        out.append(low)
    if text and text[-1] in "?!":
        out.append(text[:-1])
        if low and low[-1] in "?!":
            out.append(low[:-1])
    return list(dict.fromkeys(out))


def _pick_variants(text, rng, surface):
    pool = _user_variants(text) if surface else [text]
    return rng.choice(pool)


# ------------------------------------------------------------------ render
def render(turns):
    """Conversa -> texto do template de chat."""
    text = "<bos>"
    for t in turns:
        text += f"{USER_TAG}{t['user']}\n{ASSISTANT_TAG}{t['assistant']}\n"
    return text + "<eos>"


def encode_with_spans(tokenizer, turns):
    """-> (ids, spans). spans = [(lo, hi)] do CONTEUDO de cada resposta.

    ByteTokenizer e consistente por prefixo, logo len(encode(prefixo)) e o
    indice do token seguinte.
    """
    cur = "<bos>"
    spans = []
    for t in turns:
        cur += f"{USER_TAG}{t['user']}\n{ASSISTANT_TAG}"
        lo = len(tokenizer.encode(cur))
        cur += t["assistant"]
        hi = len(tokenizer.encode(cur))
        spans.append((lo, hi))
        cur += "\n"
    ids = tokenizer.encode(cur + "<eos>")
    return ids, spans


def build_weights(ids, spans, answer_weight, stop_weight, prompt_weight):
    """Peso por posicao (a posicao t prediz o token t+1).

    resposta (conteudo) -> answer_weight
    '\\n' que fecha cada resposta e o '<eos>' final -> stop_weight
    resto (User:, Assistant:, turnos anteriores) -> prompt_weight
    """
    w = [0.0] * (len(ids) - 1)
    for lo, hi in spans:
        for t in range(max(0, lo - 1), min(len(w), hi)):
            w[t] = answer_weight
    for lo, hi in spans:
        if 0 <= hi - 1 < len(w):
            w[hi - 1] = stop_weight          # prediz o '\n' que fecha a resposta
    if len(w) >= 2:
        w[-1] = stop_weight                  # prediz '<eos>'
        w[-2] = stop_weight                  # prediz o '\n' antes de <eos>
    return w


# ------------------------------------------------------------------ splits
def split_entries(catalog, val_frac=0.12, seed=1337):
    """Divide por ID (disjunto). Validacao = intencoes nunca vistas."""
    ids = sorted(e["id"] for e in catalog)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    val_ids = set(ids[:n_val])
    train = [e for e in catalog if e["id"] not in val_ids]
    val = [e for e in catalog if e["id"] in val_ids]
    return train, val


def _make_turns(entry, rng, surface=True):
    turns = []
    for t in entry["turns"]:
        turns.append({
            "user": _pick_variants(rng.choice(t["user"]), rng, surface),
            "assistant": rng.choice(t["assistant"]),
        })
    return turns


def build_split(entries, tokenizer, block_size, n_per_entry=20, seed=1337,
                answer_weight=20.0, stop_weight=1.0, prompt_weight=0.0,
                surface=True):
    """-> (x, y, w, stats) com `n_per_entry` conversas por intencao."""
    rng = random.Random(seed)
    rows = []
    n_skip = 0
    for e in entries:
        for _ in range(n_per_entry):
            turns = _make_turns(e, rng, surface)
            ids, spans = encode_with_spans(tokenizer, turns)
            if len(ids) > block_size:
                n_skip += 1
                continue
            ww = build_weights(ids, spans, answer_weight, stop_weight, prompt_weight)
            rows.append((ids, ww))

    n = len(rows)
    import torch
    x = torch.full((n, block_size), tokenizer.pad_id, dtype=torch.long)
    y = torch.full((n, block_size), -100, dtype=torch.long)
    w = torch.zeros((n, block_size))
    lens = []
    for r, (ids, ww) in enumerate(rows):
        x[r, :len(ids)] = torch.tensor(ids)
        for t in range(len(ids) - 1):
            y[r, t] = ids[t + 1]
            w[r, t] = ww[t]
        lens.append(len(ids))
    stats = {
        "n": n,
        "skipped_too_long": n_skip,
        "max_len": max(lens) if lens else 0,
        "mean_len": (sum(lens) / len(lens)) if lens else 0,
        "n_entries": len(entries),
    }
    return x, y, w, stats


def build_datasets(tokenizer=None, block_size=256, n_per_entry=20,
                   val_frac=0.12, seed=1337, answer_weight=20.0,
                   stop_weight=1.0, prompt_weight=0.0, path=CHAT_FILE):
    """-> dict com train/val tensores, catalogos e estatisticas."""
    tokenizer = tokenizer or ByteTokenizer()
    catalog = load_catalog(path)
    train_e, val_e = split_entries(catalog, val_frac, seed)
    x, y, w, st = build_split(train_e, tokenizer, block_size, n_per_entry, seed,
                              answer_weight, stop_weight, prompt_weight)
    xv, yv, wv, sv = build_split(val_e, tokenizer, block_size,
                                 max(1, n_per_entry // 5), seed + 1,
                                 answer_weight, stop_weight, prompt_weight)
    return {
        "tokenizer": tokenizer,
        "x": x, "y": y, "w": w, "train_stats": st,
        "xv": xv, "yv": yv, "wv": wv, "val_stats": sv,
        "train_entries": [e["id"] for e in train_e],
        "val_entries": [e["id"] for e in val_e],
        "topics": sorted({e["topic"] for e in catalog}),
    }


# ------------------------------------------------------------------ probe
def probe_items(val_entries, tokenizer, per_entry=2, seed=7, surface=False,
                max_prompt_chars=10**9):
    """-> [(prompt_text, set(respostas_aceitaveis), topic, entry_id)].

    Prompt = template ate 'Assistant: ' do ULTIMO turno (multi-turno: inclui os
    turnos anteriores). A resposta esperada e a lista de variantes daquele turno.
    """
    rng = random.Random(seed)
    items = []
    for e in val_entries:
        base = e["turns"]
        for _ in range(per_entry):
            # conversa inteira menos a ultima resposta
            prefix_turns = []
            for t in base[:-1]:
                prefix_turns.append({
                    "user": _pick_variants(rng.choice(t["user"]), rng, surface),
                    "assistant": rng.choice(t["assistant"]),
                })
            last = base[-1]
            u = _pick_variants(rng.choice(last["user"]), rng, surface)
            prompt = "<bos>"
            for t in prefix_turns:
                prompt += f"{USER_TAG}{t['user']}\n{ASSISTANT_TAG}{t['assistant']}\n"
            prompt += f"{USER_TAG}{u}\n{ASSISTANT_TAG}"
            if len(prompt) > max_prompt_chars:
                continue
            items.append((prompt, set(last["assistant"]), e["topic"], e["id"]))
    return items


def norm(text):
    return " ".join(text.strip().split())


def evaluate_probe(model, tokenizer, items, max_new_tokens=48):
    """Exact-match multi-referencia (aceita qualquer variante da resposta)."""
    import torch

    hits, per_topic = 0, {}
    model.eval()
    with torch.no_grad():
        for prompt, refs, topic, _eid in items:
            ids = tokenizer.encode(prompt)
            idx = torch.tensor([ids], dtype=torch.long)
            cap = max(1, min(max_new_tokens, model.block_size - len(ids)))
            out = model.generate(idx, max_new_tokens=cap, temperature=0,
                                 eos_id=tokenizer.eos_id)
            pred = out[0, len(ids):].tolist()
            text = tokenizer.decode(pred)
            for stop in ("<eos>", "<bos>", "<pad>", "\n"):
                i = text.find(stop)
                if i != -1:
                    text = text[:i]
            ok = norm(text) in {norm(r) for r in refs}
            hits += int(ok)
            c, n = per_topic.get(topic, (0, 0))
            per_topic[topic] = (c + int(ok), n + 1)
    model.train()
    return hits, len(items), per_topic
