"""TinyGPT: Transformer decoder-only minimo, CPU-friendly.

Componentes: embeddings de token/posicao, N blocos (LayerNorm pre-norm,
atencao causal multi-cabeca, MLP com GELU), projecao final para o vocabulario.

A atencao aceita um KV cache opcional para geracao token-a-token (inferencia),
o que evita recomputar o contexto a cada passo.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x, positions, base=10000.0):
    """Aplica Rotary Position Embedding em x de shape (B, H, T, D).

    `positions` tem shape (T,) com a posicao absoluta de cada token. Assim
    funciona tambem na geracao token-a-token (T=1, pos_offset correto).
    """
    D = x.shape[-1]
    half = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def masked_cross_entropy(logits, targets, weights=None):
    """Cross-entropy com mascara (-100) e peso opcional por token.

    `weights` (B, T) permite dar peso extra aos tokens da resposta, mantendo
    o restante da sequencia como curriculo (peso 1). Isso "forca a resposta"
    sem perder o sinal denso que ajuda o modelo a aprender a ler o contexto.
    """
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        ignore_index=-100, reduction="none",
    ).reshape_as(targets)
    valid = (targets != -100).to(ce.dtype)
    if weights is None:
        return (ce * valid).sum() / valid.sum().clamp_min(1.0)
    w = weights.to(ce.dtype) * valid
    return (ce * w).sum() / w.sum().clamp_min(1e-8)


def attention_copy_loss(att, src):
    """Pointer loss: forca a atencao (media das cabecas) em cada posicao com
    `src>=0` a apontar para o token-fonte `src`.

    att: (B, H, T, Tk) pesos de atencao da ultima camada;
    src: (B, T) indice alvo por posicao (-1 ignora).
    """
    B, H, T, Tk = att.shape
    p = att.mean(dim=1)  # (B, T, Tk); media sobre as cabecas
    valid = src >= 0
    if not bool(valid.any()):
        return att.new_zeros(())
    p = p.clamp_min(1e-9).log().reshape(-1, Tk)
    tgt = src.clamp_min(0).reshape(-1, 1)
    picked = p.gather(1, tgt).squeeze(1)[valid.reshape(-1)]
    return -picked.mean()


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0, rope=False):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd deve ser divisivel por n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        self.rope = rope
        if rope:
            assert self.head_dim % 2 == 0, "RoPE exige head_dim par"
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x, past_kv=None, positions=None, return_attn=False):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.rope:
            q = apply_rope(q, positions)
            k = apply_rope(k, positions)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        present_kv = (k, v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if past_kv is None:
            scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        else:
            assert T == 1, "KV cache so e suportado para geracao token a token"
        att = F.softmax(scores, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(y))
        return out, present_kv, (att if return_attn else None)


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0, rope=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, rope)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x, past_kv=None, positions=None, return_attn=False):
        h, present_kv, att = self.attn(self.ln1(x), past_kv, positions, return_attn)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, present_kv, att


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=384,
        n_embd=128,
        n_head=4,
        n_layer=4,
        dropout=0.0,
        tie_embeddings=False,
        pos_encoding="learned",
        copy_head=False,
        normalize_embeddings=False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings
        self.pos_encoding = pos_encoding
        self.copy_head = copy_head
        self.normalize_embeddings = normalize_embeddings
        assert pos_encoding in ("learned", "rope"), "pos_encoding deve ser 'learned' ou 'rope'"

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd) if pos_encoding == "learned" else None
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                Block(n_embd, n_head, block_size, dropout, rope=(pos_encoding == "rope"))
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        if copy_head:
            self.copy_q = nn.Linear(n_embd, n_embd, bias=False)
            self.copy_k = nn.Linear(n_embd, n_embd, bias=False)
            self.copy_v = nn.Linear(n_embd, n_embd, bias=False)
            self.copy_gate = nn.Linear(n_embd, 1)
            self.register_buffer(
                "copy_mask",
                torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
                persistent=False,
            )
        self.apply(self._init_weights)
        if tie_embeddings:
            self.head.weight = self.tok_emb.weight
        if normalize_embeddings:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def _project(self, h):
        """Readout. Com `normalize_embeddings`, vira retrieval por cosseno
        (embeddings L2-normalizados + temperatura aprendivel) — copiar passa a
        ser vizinho mais proximo e some o vies de norma do embedding."""
        if not self.normalize_embeddings:
            return self.head(h)
        w = F.normalize(self.head.weight, dim=-1)
        return F.normalize(h, dim=-1) @ w.t() * self.logit_scale.exp()

    def _copy_readout(self, x, logits):
        """Readout Ponteiro+Copia (attention head dedicada + gate).

        Um head de atencao causal produz um vetor "copiado" das posicoes do
        input; esse vetor passa pela MESMA cabeca de saida (tied) e vira uma
        distribuicao sobre o vocabulario. A prob. final mistura o softmax
        gerativo com a copia via um gate aprendido:

            p = gate * softmax(logits) + (1-gate) * softmax(head(pointer @ V))

        Como a unica forma barata de acertar next/prev e apontar para a letra
        certa do JSON e copia-la, o gradiente da loss FORCA o circuito de
        match+copia (inducao) a se formar -> generaliza por construcao.
        Retorna (log-probs, pointer).
        """
        B, T, C = x.shape
        q = self.copy_q(x)
        k = self.copy_k(x)
        v = self.copy_v(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(C)
        scores = scores.masked_fill(~self.copy_mask[:T, :T], float("-inf"))
        pointer = F.softmax(scores, dim=-1)
        copy_logits = self._project(pointer @ v)
        gate = torch.sigmoid(self.copy_gate(x))
        probs = gate * F.softmax(logits, dim=-1) + (1.0 - gate) * F.softmax(copy_logits, dim=-1)
        return torch.log(probs.clamp_min(1e-9)), pointer

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_kvs=None, pos_offset=0, weights=None,
                aux_src=None, aux_weight=0.0, return_attn=False):
        """Forward.

        `aux_src` (B, T) com o indice-fonte (na sequencia) que o modelo DEVE
        atender para copiar a resposta em cada posicao; -1 ignora. Supervisiona
        a atencao da ULTIMA camada com `aux_weight` (pointer loss), forçando a
        formacao do circuito de inducao.
        """
        B, T = idx.shape
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        last = len(self.blocks) - 1
        want_attn = return_attn or (aux_src is not None and aux_weight > 0)
        last_att = None
        presents = []
        for i, block in enumerate(self.blocks):
            past_kv = None if past_kvs is None else past_kvs[i]
            want = want_attn and (i == last or return_attn)
            x, present_kv, att = block(x, past_kv, pos, return_attn=want)
            if want and i == last:
                last_att = att
            presents.append(present_kv)

        x = self.ln_f(x)
        logits = self._project(x)

        pointer = None
        if self.copy_head:
            logits, pointer = self._copy_readout(x, logits)

        loss = None
        if targets is not None:
            loss = masked_cross_entropy(logits, targets, weights)
        if aux_src is not None and aux_weight > 0:
            src_probs = pointer if self.copy_head else last_att
            if src_probs is not None:
                if src_probs.dim() == 3:
                    src_probs = src_probs.unsqueeze(1)
                aux = attention_copy_loss(src_probs, aux_src)
                loss = aux * aux_weight if loss is None else loss + aux_weight * aux
        if return_attn:
            return logits, presents, loss, last_att
        return logits, presents, loss

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        self.eval()
        past_kvs = None
        for _ in range(max_new_tokens):
            if past_kvs is None:
                idx_cond = idx[:, -self.block_size:]
            else:
                idx_cond = idx[:, -1:]
            pos_offset = 0 if past_kvs is None else idx.size(1) - 1
            logits, past_kvs, _ = self(idx_cond, past_kvs=past_kvs, pos_offset=pos_offset)
            logits = logits[:, -1, :]

            if temperature == 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    topv, _ = torch.topk(logits, k)
                    logits = logits.masked_fill(logits < topv[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
        return idx
