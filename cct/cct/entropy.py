"""Huffman canonico (numpy) — estagio de entropia dos codecs L1/L2.

Implementacao propria, sem dependencia externa. Trabalha sobre
np.ndarray[uint8] e devolve bytes.

Formato do payload (v2):
    MAGIC "HF2"
    [2 bytes] n_simbolos           (0 = tabela vazia)
    por simbolo (7 bytes): len(2) | code(4) | sym(1)
    bits empacotados (MSB first)

O formato v1 (1 byte n-1 + 5 bytes/simbolo + 2 bytes de code) ainda e'
aceito na leitura, porque era o usado pelos checksums/benchmarks anteriores.

Notas de correcao (v2):

  * A construcao de comprimentos original somava +1 em TODOS os simbolos
    (o merge final era contado duas vezes). Isso deixava a soma de Kraft em
    0.5 — prefixo-livre, mas gastando 1 bit extra por simbolo.
  * O campo `code` tinha 2 bytes. Com distribuicoes enviesadas (ex.: plano
    de expoente BF16, 24 simbolos) o comprimento maximo passava de 16 bits e
    o codigo era truncado em silencio -> dois simbolos colidiam e o decode
    falhava. Agora o campo tem 4 bytes e o comprimento e' limitado a 31 bits
    (fallback de codigo plano se a arvore for mais profunda que isso).
"""

import numpy as np

MAGIC = b"HF2"
MAX_CODE_BITS = 31


class BitWriter:
    __slots__ = ("buf", "acc", "nbits")

    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.nbits = 0

    def write(self, code: int, length: int):
        self.acc = (self.acc << length) | (code & ((1 << length) - 1))
        self.nbits += length
        while self.nbits >= 8:
            self.nbits -= 8
            self.buf.append((self.acc >> self.nbits) & 0xFF)
        self.acc &= (1 << self.nbits) - 1

    def flush(self):
        if self.nbits:
            self.buf.append((self.acc << (8 - self.nbits)) & 0xFF)
            self.acc = 0
            self.nbits = 0
        return bytes(self.buf)


class BitReader:
    __slots__ = ("data", "pos", "nbits", "acc")

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.nbits = 0
        self.acc = 0

    def read_bit(self) -> int:
        if self.nbits == 0:
            if self.pos >= len(self.data):
                raise ValueError("stream huffman truncado")
            self.acc = self.data[self.pos]
            self.pos += 1
            self.nbits = 8
        self.nbits -= 1
        return (self.acc >> self.nbits) & 1


def _huffman_lengths(freqs):
    """Comprimentos otimos de Huffman -> {sym: len}.

    Arvore construida com ids disjuntos (folhas 0..255, internos >= 256) e
    profundidade calculada por travessia, sem dupla contagem.
    """
    import heapq

    syms = [s for s, f in freqs.items() if f > 0]
    if not syms:
        return {}
    if len(syms) == 1:
        return {syms[0]: 1}

    heap = [(freqs[s], 256 + i, s) for i, s in enumerate(syms)]
    heapq.heapify(heap)
    left = {}
    right = {}
    nxt = 256 + len(syms)
    while len(heap) > 1:
        f1, _o1, a = heapq.heappop(heap)
        f2, _o2, b = heapq.heappop(heap)
        node = nxt
        nxt += 1
        left[node] = a
        right[node] = b
        heapq.heappush(heap, (f1 + f2, node, node))

    lengths = {}
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if node in left:
            stack.append((left[node], depth + 1))
            stack.append((right[node], depth + 1))
        else:
            lengths[node] = max(1, depth)

    # Arvore patologicamente profunda -> codigo plano (correto, menos eficiente).
    if max(lengths.values()) > MAX_CODE_BITS:
        width = max(1, (len(lengths) - 1).bit_length())
        lengths = {s: width for s in lengths}
    return lengths


def _canonical_codes(lengths):
    """{sym: len} -> {sym: (code, len)} em ordem canonica."""
    if not lengths:
        return {}
    max_len = max(lengths.values())
    bl_count = [0] * (max_len + 1)
    for ln in lengths.values():
        bl_count[ln] += 1
    code = 0
    next_code = [0] * (max_len + 2)
    for bits in range(1, max_len + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code
    codes = {}
    for sym in sorted(lengths, key=lambda s: (lengths[s], s)):
        ln = lengths[sym]
        codes[sym] = (next_code[ln], ln)
        next_code[ln] += 1
    return codes


def huffman_compress(data: np.ndarray) -> bytes:
    """Comprime np.ndarray[uint8] (1-D) com Huffman canonico."""
    data = np.ascontiguousarray(data, dtype=np.uint8)
    if data.size == 0:
        return MAGIC + (0).to_bytes(2, "little")

    counts = np.bincount(data, minlength=256)
    freqs = {i: int(c) for i, c in enumerate(counts) if c > 0}
    lengths = _huffman_lengths(freqs)
    codes = _canonical_codes(lengths)
    syms = sorted(lengths, key=lambda s: (lengths[s], s))

    code_of = np.zeros(256, dtype=np.uint32)
    len_of = np.zeros(256, dtype=np.uint8)
    for s, (c, ln) in codes.items():
        code_of[s] = c
        len_of[s] = ln

    bw = BitWriter()
    # laco local (evita lookup de atributo por simbolo)
    write = bw.write
    for sym, ln in zip(code_of[data].tolist(), len_of[data].tolist()):
        write(sym, ln)
    body = bw.flush()

    out = bytearray(MAGIC)
    out += len(syms).to_bytes(2, "little")
    for s in syms:
        ln, c = lengths[s], codes[s][0]
        out += ln.to_bytes(2, "little") + c.to_bytes(4, "little") + bytes([s])
    return bytes(out) + body


def _parse_legacy(payload: bytes):
    """Formato v1: 1 byte (n-1) + 5 bytes/simbolo (len2|code2|sym)."""
    n_syms = payload[0] + 1
    off = 1
    table = {}
    for _ in range(n_syms):
        ln = (payload[off] << 8) | payload[off + 1]
        code = (payload[off + 2] << 8) | payload[off + 3]
        sym = payload[off + 4]
        table[(int(code), int(ln))] = int(sym)
        off += 5
    return table, payload[off:]


def huffman_decompress(payload: bytes, size: int) -> np.ndarray:
    """Descomprime payload de huffman_compress para np.ndarray[uint8].

    O decoder e' sequencial (Huffman de comprimento variavel). Ele e' usado
    por L1/L2; o L3 (o codec competitivo) nao depende dele.
    """
    if size == 0:
        return np.empty(0, dtype=np.uint8)

    if payload[:3] == MAGIC:
        n_syms = int.from_bytes(payload[3:5], "little")
        off = 5
        table = {}
        for _ in range(n_syms):
            ln = int.from_bytes(payload[off:off + 2], "little")
            code = int.from_bytes(payload[off + 2:off + 6], "little")
            sym = payload[off + 6]
            table[(code, ln)] = sym
            off += 7
        body = payload[off:]
    else:
        table, body = _parse_legacy(payload)

    if not table:
        raise ValueError("tabela huffman vazia")

    out = np.empty(size, dtype=np.uint8)
    get = table.get
    ndata = len(body)
    idx = 0
    acc = 0
    nacc = 0
    cur = 0
    nbits = 0
    for i in range(size):
        while True:
            if nacc == 0:
                if idx >= ndata:
                    raise ValueError("stream huffman truncado")
                acc = body[idx]
                idx += 1
                nacc = 8
            nacc -= 1
            cur = (cur << 1) | ((acc >> nacc) & 1)
            nbits += 1
            sym = get((cur, nbits))
            if sym is not None:
                out[i] = sym
                cur = 0
                nbits = 0
                break
            if nbits > MAX_CODE_BITS:
                raise ValueError("stream huffman invalido")
    return out
