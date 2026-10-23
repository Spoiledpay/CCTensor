"""Tokenizador caractere-a-caractere (didatico).

Cada caractere vira um id. Tokens especiais (<pad>, <bos>, <eos>) ocupam os
primeiros ids do vocabulario, seguidos pelos caracteres em ordem ordenada.
"""

import json

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>")


class CharTokenizer:
    def __init__(self, chars, special_tokens=SPECIAL_TOKENS):
        self.special_tokens = list(special_tokens)
        self.chars = sorted(set(chars))

        self.stoi = {}
        self.itos = {}
        for i, tok in enumerate(self.special_tokens):
            self.stoi[tok] = i
            self.itos[i] = tok
        offset = len(self.special_tokens)
        for i, ch in enumerate(self.chars):
            self.stoi[ch] = offset + i
            self.itos[offset + i] = ch

        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        ids = []
        i = 0
        n = len(text)
        while i < n:
            for tok in self.special_tokens:
                if text.startswith(tok, i):
                    ids.append(self.stoi[tok])
                    i += len(tok)
                    break
            else:
                ch = text[i]
                if ch not in self.stoi:
                    raise KeyError(f"Caractere fora do vocabulario: {ch!r}")
                ids.append(self.stoi[ch])
                i += 1
        return ids

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def state_dict(self):
        return {"special_tokens": self.special_tokens, "chars": self.chars}

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["chars"], d["special_tokens"])

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_state_dict(json.load(f))

    @classmethod
    def from_corpus(cls, text, special_tokens=SPECIAL_TOKENS):
        stripped = text
        for tok in special_tokens:
            stripped = stripped.replace(tok, "")
        return cls(stripped, special_tokens)


def gpt2_byte_encoder():
    """bytes_to_unicode() do GPT-2.

    O tokenizer BPE do llama.cpp aplica esta representacao antes de dividir
    o texto: espaco -> 'Ġ' (U+0120), newline -> 'Ċ' (U+010A), printaveis
    ASCII ficam iguais. Usar os MESMOS nomes no vocab garante que o
    llama.cpp reproduza exatamente a segmentacao do treino (zero UNK).
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


class ByteTokenizer:
    """Tokenizador byte-level: 1 byte = 1 token (259 tokens no total).

    Vantagens:
      - nenhum caractere e "desconhecido": qualquer texto (com acentos,
        emojis, etc.) e representavel -> nada e descartado na inferencia;
      - o nome de cada token e exatamente o byte-encode do GPT-2, o mesmo
        que o llama.cpp usa -> a exportacao para GGUF/Ollama e 1:1, sem
        remapeamento e sem [UNK_BYTE_..].
    """

    def __init__(self, special_tokens=SPECIAL_TOKENS):
        self.special_tokens = list(special_tokens)
        self.byte_level = True
        enc = gpt2_byte_encoder()
        self.itos = {}
        self.stoi = {}
        for i, tok in enumerate(self.special_tokens):
            self.stoi[tok] = i
            self.itos[i] = tok
        nid = len(self.special_tokens)
        self.id_to_byte = {}
        for b in range(256):
            s = enc[b]
            self.stoi[s] = nid
            self.itos[nid] = s
            self.id_to_byte[nid] = b
            nid += 1
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        ids = []
        i, n = 0, len(text)
        while i < n:
            matched = False
            for tok in self.special_tokens:
                if text.startswith(tok, i):
                    ids.append(self.stoi[tok])
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue
            for b in text[i].encode("utf-8"):
                ids.append(3 + b)
            i += 1
        return ids

    def encode_bytes(self, data: bytes):
        """Version byte-a-byte (UTF-8), sem passar por str."""
        return [3 + b for b in data]

    def decode(self, ids):
        pieces = []
        raw = bytearray()
        for i in ids:
            i = int(i)
            if i in self.id_to_byte:
                raw.append(self.id_to_byte[i])
            else:
                if raw:
                    pieces.append(raw.decode("utf-8", errors="replace"))
                    raw = bytearray()
                pieces.append(self.itos.get(i, ""))
        if raw:
            pieces.append(raw.decode("utf-8", errors="replace"))
        return "".join(pieces)

    def state_dict(self):
        return {"special_tokens": self.special_tokens, "byte_level": True}

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["special_tokens"])

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_state_dict(json.load(f))

    @classmethod
    def from_corpus(cls, text, special_tokens=SPECIAL_TOKENS):
        return cls(special_tokens)


class ItemChTokenizer:
    """Tokenizador de especialista: cada ITEM do alfabeto e um token atomico.

    Funciona tanto para letras (A, B, ...) quanto para palavras ("carro",
    "Roda"): o texto e codificado por casamento mais longo, preferindo os
    itens inteiros; qualquer caractere restante (template, digitos, pontuacao)
    cai no vocabulario de caracteres.

    Estado salvo: {"chars", "items", "special_tokens"}.
    """

    def __init__(self, chars, items=(), special_tokens=SPECIAL_TOKENS):
        self.special_tokens = list(special_tokens)
        self.chars = sorted(set(chars))
        self.items = sorted({str(it) for it in items if it})

        self.stoi = {}
        self.itos = {}
        for i, tok in enumerate(self.special_tokens):
            self.stoi[tok] = i
            self.itos[i] = tok
        nid = len(self.special_tokens)
        for ch in self.chars:
            if ch in self.stoi:
                continue
            self.stoi[ch] = nid
            self.itos[nid] = ch
            nid += 1
        # itens ocupam ids proprios (multi-caractere viram um unico id)
        self.items_by_len = sorted(self.items, key=len, reverse=True)
        for item in self.items_by_len:
            if item in self.stoi:
                continue
            self.stoi[item] = nid
            self.itos[nid] = item
            nid += 1

        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        ids = []
        i, n = 0, len(text)
        while i < n:
            consumed = False
            for tok in self.special_tokens:
                if text.startswith(tok, i):
                    ids.append(self.stoi[tok])
                    i += len(tok)
                    consumed = True
                    break
            if consumed:
                continue
            for item in self.items_by_len:
                if text.startswith(item, i):
                    ids.append(self.stoi[item])
                    i += len(item)
                    consumed = True
                    break
            if consumed:
                continue
            ch = text[i]
            if ch not in self.stoi:
                raise KeyError(f"Caractere fora do vocabulario: {ch!r}")
            ids.append(self.stoi[ch])
            i += 1
        return ids

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def state_dict(self):
        return {
            "special_tokens": self.special_tokens,
            "chars": self.chars,
            "items": self.items,
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["chars"], d.get("items", []), d["special_tokens"])

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_state_dict(json.load(f))

    @classmethod
    def from_corpus(cls, text, items=(), special_tokens=SPECIAL_TOKENS):
        stripped = text
        for tok in special_tokens:
            stripped = stripped.replace(tok, "")
        return cls(stripped, items, special_tokens)
