"""Dataset curado: loader, validador e estatisticas. ZERO geracao mecanica.

O conteudo mora em input/dialogos.jsonl, escrito/revisado a mao:
  dialogos sobre o alfabeto (com `verify`) + conversa geral PT-BR.

Schema (alfabeto):
  {"id","task","alphabet","user":[...],"assistant":[...],
   "verify":[{"turn":i,"task":...,"letter":...,"k":...,"variant":...}]}
  `turn` = indice em `assistant` cuja resposta e recomputada e comparada.

Schema (cotidiano):
  {"id","task":"general","user":[...],"assistant":[...]}

Como curar mais dialogos: acrescente linhas a mao nos JSONL e rode
`python -c "from minigpt.dialogues import validate; validate()"`.
Regras: portugues natural (acentos ok), 1 ideia por dialogo, resposta
curta no alfabeto, sem combinatoria (nada de gerar todas as letras x
todos os alfabetos por script).
"""

import json
import os
from collections import Counter

VOGAIS = "AEIOU"
CURATED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")
DIALOGUES_FILE = os.path.join(CURATED_DIR, "dialogos.jsonl")


def expected_answer(task, alphabet, letter=None, k=None, variant=None):
    """Recomputa a resposta a partir do alfabeto. Base da verificacao."""
    alpha = list(alphabet)
    n = len(alpha)
    if task == "count":
        return str(n)
    if task == "first":
        return alpha[0]
    if task == "last":
        return alpha[-1]
    if task == "show":
        return ", ".join(alpha)
    if task == "first_n":
        return ", ".join(alpha[:k])
    if task == "last_n":
        return ", ".join(alpha[-k:])
    if task == "next":
        i = alpha.index(letter)
        return alpha[i + 1] if i + 1 < n else alpha[-1]
    if task == "prev":
        i = alpha.index(letter)
        return alpha[i - 1] if i - 1 >= 0 else alpha[0]
    if task == "position":
        return str(alpha.index(letter) + 1)
    if task == "contains":
        return "Sim." if letter in alpha else "Não."
    if task == "vowels":
        present = [c for c in alpha if c in VOGAIS]
        if variant == "count":
            return str(len(present))
        return ", ".join(present) if present else "Nenhuma."
    if task == "consonants":
        present = [c for c in alpha if c not in VOGAIS]
        if variant == "count":
            return str(len(present))
        return ", ".join(present) if present else "Nenhuma."
    if task == "reverse":
        return ", ".join(reversed(alpha))
    if task == "middle":
        if n % 2 == 1:
            return alpha[n // 2]
        return f"{alpha[n // 2 - 1]}, {alpha[n // 2]}"
    raise ValueError(f"task desconhecida: {task}")


def load_jsonl(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: JSON invalido: {exc}")
    return entries


def load_curated(path=DIALOGUES_FILE):
    """Carrega o dataset curado (alfabeto com verify + cotidiano)."""
    return load_jsonl(path)


def validate(path=DIALOGUES_FILE):
    """Valida tudo: schema, ids unicos, respostas recomputadas, duplicatas."""
    entries = load_jsonl(path)
    alpha_entries = [e for e in entries if e.get("alphabet")]
    every_entries = [e for e in entries if not e.get("alphabet")]
    errors = []

    def check_turns(e, where):
        u, a = e.get("user"), e.get("assistant")
        if not isinstance(u, list) or not u or not all(isinstance(x, str) for x in u):
            errors.append(f"{where}: 'user' deve ser lista nao vazia de strings")
        if not isinstance(a, list) or len(a) != len(u) or not all(isinstance(x, str) for x in a):
            errors.append(f"{where}: 'assistant' deve ter 1 resposta por turno")

    seen_ids = set()
    for e in alpha_entries + every_entries:
        eid = e.get("id")
        if not eid or eid in seen_ids:
            errors.append(f"id ausente ou duplicado: {eid!r}")
        seen_ids.add(eid)

    for e in alpha_entries:
        where = f"alfabeto:{e.get('id')}"
        check_turns(e, where)
        alpha = e.get("alphabet", "")
        if not alpha or len(set(alpha)) != len(alpha):
            errors.append(f"{where}: 'alphabet' ausente ou com letras repetidas")
        for v in e.get("verify", []):
            t = v.get("turn")
            try:
                exp = expected_answer(v["task"], alpha, v.get("letter"),
                                      v.get("k"), v.get("variant"))
            except (ValueError, KeyError) as exc:
                errors.append(f"{where}: verify invalido ({exc})")
                continue
            got = (e.get("assistant") or [None] * (t + 1))[t] if isinstance(t, int) else None
            if got is None or got.strip() != exp:
                errors.append(f"{where} turno {t}: esperado {exp!r}, escrito {got!r}")

    for e in every_entries:
        check_turns(e, f"cotidiano:{e.get('id')}")

    # duplicatas exatas de dialogo completo (sinal de geracao mecanica)
    seen_dialogues = set()
    for e in alpha_entries:
        key = (e.get("alphabet"), tuple(e.get("user", [])), tuple(e.get("assistant", [])))
        if key in seen_dialogues:
            errors.append(f"dialogo duplicado: {e.get('id')}")
        seen_dialogues.add(key)

    if errors:
        print(f"VALIDACAO FALHOU ({len(errors)} erros):")
        for err in errors[:20]:
            print(f"  - {err}")
        raise SystemExit(f"{len(errors)} erros de curadoria")
    print(f"OK: {len(alpha_entries)} alfabeto + {len(every_entries)} cotidiano, "
          f"{len(seen_ids)} ids unicos, respostas conferidas.")
    return alpha_entries, every_entries


def stats(path=DIALOGUES_FILE):
    entries = load_jsonl(path)
    alpha_entries = [e for e in entries if e.get("alphabet")]
    every_entries = [e for e in entries if not e.get("alphabet")]
    by_task = Counter(e.get("task", "?") for e in alpha_entries)
    multi = sum(1 for e in alpha_entries if len(e.get("user", [])) > 1)
    multi_every = sum(1 for e in every_entries if len(e.get("user", [])) > 1)
    total = len(alpha_entries) + len(every_entries)
    print(f"dialogos curados: {total} = {len(alpha_entries)} alfabeto + {len(every_entries)} cotidiano")
    print(f"por tarefa (alfabeto): {dict(by_task)}")
    print(f"multi-turno: {multi} alfabeto + {multi_every} cotidiano")
    return {"total": total, "alfabeto": len(alpha_entries),
            "cotidiano": len(every_entries), "by_task": dict(by_task)}


def to_messages(entry, system):
    """Monta messages[] com system injetado. 1o turno do alfabeto ganha o bloco JSON."""
    from .data import build_json_block

    messages = [{"role": "system", "content": system}]
    users, assistants = entry["user"], entry["assistant"]
    for i, (u, a) in enumerate(zip(users, assistants)):
        if i == 0 and entry.get("alphabet"):
            u = f"Alfabeto em JSON:\n{build_json_block(entry['alphabet'])}\n\nPergunta: {u}"
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    return messages
