<p align="center">
  <img src="Logotipo_CCT_%20Tensor_Compactado.png" alt="Projeto" width="420">
</p>

<h1 align="center">SLM de conversa em inglês</h1>

<p align="center">
  Um <strong>Small Language Model</strong> decoder-only, byte-level, treinado do zero em CPU,<br>
  especializado em <strong>diálogos curtos do cotidiano em inglês</strong>.
</p>

---

## Sumário

1. [O que é](#1-o-que-é)
2. [Ficha técnica](#2-ficha-técnica)
3. [Arquitetura do modelo](#3-arquitetura-do-modelo)
4. [Arquitetura dos dados](#4-arquitetura-dos-dados)
5. [Arquitetura do treino](#5-arquitetura-do-treino)
6. [Inferência](#6-inferência)
7. [Sistema de pastas](#7-sistema-de-pastas)
8. [Como usar](#8-como-usar)
9. [Resultados medidos](#9-resultados-medidos)
10. [Diagnóstico e limitações](#10-diagnóstico-e-limitações)
11. [Roadmap](#11-roadmap)
12. [Documentação](#12-documentação)

---

## 1. O que é

Um SLM (Small Language Model) **especialista em conversa cotidiana em inglês**:
saudações, small talk, tempo/data, direções, compras, restaurante, saúde, trabalho,
estudos, hobbies, sentimentos, convites, emergências — **39 tópicos**.

O objetivo do projeto é ser um **laboratório completo e reproduzível** de treino de
linguagem em CPU: do dataset curado ao checkpoint, da loss mascarada ao ONNX, e
do modelo ao formato comprimido (CCT). Não é um assistente de uso geral — é um
SLM pequeno o suficiente para ser entendido e auditado por inteiro.

```
Objetivo (ainda não atingido — ver §10):

Usuário:  What time is it?
Modelo :  It's three o'clock.
```

> **Nota histórica:** a primeira fase treinava um "especialista em alfabetos"
> (ler um JSON de A–Z). Esse dataset e os modelos foram **removidos**; o foco
> passou a ser conversa. `train.py`, `chat.py`, `infer.py`, `export_gguf.py` e
> `GUIA.md` ficam como registro daquela fase.

## 2. Ficha técnica

| item | valor |
|---|---|
| tipo | Transformer decoder-only (GPT), causal |
| tokenizador | byte-level (256 bytes + 3 especiais = **259 tokens**), sem `<UNK>` |
| vocabulário | 259 |
| contexto (`block_size`) | 192 tokens (≈ 192 bytes) |
| comprimento real das conversas | média **74** · máximo **128** |
| **run medido (1.9M)** | `n_embd/n_head/n_layer = 192/6/4` · **1.866.432** params · MLP 192→768→192 · `head_dim` 32 |
| **default do `config.json` (6.1M)** | `288/8/6` · **6.143.328** params · MLP 288→1152→288 · `head_dim` 36 |
| embeddings | token + posição aprendida, **weight tying** (head = `tok_emb`) |
| precisão de treino | float32 (CPU) |
| dataset | `input/chat_en.jsonl` — 206 intenções · 39 tópicos |
| artefatos esperados | `modelos/chat_en.pt`, `modelos/chat_en.onnx`, `chat_en_history.csv` |

> **Atenção:** `config.json` (default) descreve o modelo de **6.1M**. O run de
> 1.9M documentado em §9 foi treinado passando `--n-embd 192 --n-head 6
> --n-layer 4`. **Artefatos gerados em 2026-10-22:** `modelos/chat_en.pt`,
> `modelos/chat_en.onnx`, `modelos/chat_en.gguf` e `ollama/Modelfile.chat_en`.

## 3. Arquitetura do modelo

`minigpt/model.py` → `TinyGPT`

```
idx (B, T) ──► tok_emb ──┐
                         ├─(+)──► 4 × Bloco ──► LN_f ──► head (tied) ──► logits (B,T,259)
              pos_emb ───┘

Bloco (pré-norm):
    x ──► LN1 ──► Atenção causal multi-head ──►(+)──┐
     │                      6 heads, head_dim 32    │
     └──────────────────────────────────────────────┤
    x ──► LN2 ──► Linear 192→768 ──► GELU ──► Linear 768→192 ──►(+)──► saída
```

Detalhes de cada componente:

| componente | implementação |
|---|---|
| Atenção | causal multi-head com máscara triangular; QKV separados; `dropout=0`; suporta **KV cache** para geração token-a-token |
| Normalização | `LayerNorm` (pré-norm), uma antes da atenção e uma antes do MLP |
| MLP | `Linear(192→768)` + `GELU` + `Linear(768→192)` (4x) |
| Saída | `Linear(192→259, bias=False)` com **weight tying** (`head.weight = tok_emb.weight`) |
| Inicialização | normal(0, 0.02); bias zero |
| Opcionais (não usados no chat) | RoPE, copy-head, readout por cosseno |

Contagem de parâmetros:

| bloco | params |
|---|---|
| `tok_emb` | 49.728 |
| `pos_emb` | 36.864 |
| 4 × bloco (LN 384 + attn 148.224 + LN 384 + MLP 295.872) | 4 × 444.864 = 1.779.456 |
| `ln_f` | 384 |
| `head` | 49.728 (**tied** → não soma) |
| **Total** | **1.866.432** |

## 4. Arquitetura dos dados

`input/chat_en.jsonl` é a **fonte única de verdade** — uma conversa por linha:

```json
{"id": "greet_01", "topic": "greetings",
 "turns": [{"user": ["Hi!", "Hello!", "Hey!"],
            "assistant": ["Hi! How are you?", "Hello! Nice to see you."]}]}
```

* `turns` é uma **lista** → diálogos multi-turno são suportados nativamente.
* `user` e `assistant` são **listas de variantes**. O builder sorteia uma de
  cada lado, gerando combinações diferentes por exemplo.
* As variantes de usuário recebem **variantes de superfície** (caixa e
  pontuação), para o modelo não depender de `?`/`!` nem de maiúscula inicial.

### Template de chat

```
<bos>User: {u1}\nAssistant: {a1}\nUser: {u2}\nAssistant: {a2}\n<eos>
```

O mesmo template é usado no treino e na inferência (é o contrato do modelo).

### Volume atual

| medida | valor |
|---|---|
| intenções (`id`) | **206** |
| turnos | 207 |
| tópicos | **39** |
| variantes de usuário / assistente | 416 / 417 |
| exemplos com `n_per_entry=20` | 3.620 treino (181 intenções) + 100 val (25 intenções inéditas) |
| exemplos com `n_per_entry=60` | 10.860 treino + 300 val |
| split | **por `id`**: as intenções de validação nunca aparecem no treino |

## 5. Arquitetura do treino

`train_chat.py` + `minigpt/chat_dataset.py`

### 5.1 Máscara de loss (o ponto central)

Só as **respostas** são supervisionadas. Prompt, rótulos (`User:`/`Assistant:`) e
turnos anteriores têm peso **zero** — o modelo não gasta capacidade copiando o
que já está no contexto.

```
<bos> User: Hi! \n Assistant: Hi! How are you? \n <eos>
      └──────── peso 0 ────────┘ └── peso 20 ──┘ └─ peso 1 ─┘
                                   (conteúdo)     (\n e <eos>: ensina a parar)
```

| parte | peso |
|---|---|
| conteúdo da resposta | `--answer-weight` (20) |
| `\n` que fecha a resposta e `<eos>` | `--stop-weight` (1) |
| JSON/rótulos/perguntas/turnos anteriores | `--prompt-weight` (0) |

A loss é `CrossEntropy` ponderada por token, normalizada pela soma dos pesos, com
`-100` nas posições de padding (`masked_cross_entropy`, `minigpt/model.py`).

### 5.2 Loop de treino

| item | escolha |
|---|---|
| otimizador | AdamW (`wd=0.01`) |
| LR | **warmup + cosine**, pico `3e-3` (config), warmup = 10% dos passos |
| clipping | norma de gradiente em 1.0 |
| batch | 64 |
| **length bucketing** | lotes com comprimentos parecidos; o bloco é cortado em `T = max(len do lote)` → ~2,5x menos compute no padding |
| épocas | 10 |
| checkpoint | salvo **a cada melhora de `val_loss`** (não a cada época) |
| retomada | `--resume` (modelo + optimizer + scheduler + época) |
| histórico | `modelos/chat_en_history.csv` |
| export | `.onnx` automático ao final (`--no-onnx` para pular) |

### 5.3 Métricas por época

| métrica | o que mede |
|---|---|
| `loss` | CE ponderada nas respostas (treino) |
| `val_loss` | a mesma CE em **intenções nunca vistas** |
| `val_ppl` | `exp(val_loss)` |
| `probe` | **exact-match multi-referência**: gera a resposta greedy e aceita qualquer variante da lista |

## 6. Inferência

`minigpt/utils.py` → `load_checkpoint` + `TinyGPT.generate`

```
prompt = "<bos>User: {texto}\nAssistant: "
ids    = ByteTokenizer.encode(prompt)
loop   = forward → argmax (temperature=0) → anexa → para em <eos>
```

* **greedy** (`temperature=0`) para respostas determinísticas.
* `temperature`/`top_k` opcionais para amostragem.
* **KV cache** na geração (cada passo processa só o token novo).
* Limpeza da saída: corta em `<eos>`, `<bos>`, `<pad>` e no primeiro `\n`.
* **ONNX**: grafo de forward completo (sem KV cache), para deploy/portabilidade.

## 7. Sistema de pastas

```
.
├── Logotipo_CCT_ Tensor_Compactado.png
├── README.md                   <- este arquivo
├── config.json                 <- defaults de modelo + chat (+ slm)
├── Compress.md                 <- especificação do CCT
├── GUIA.md                     <- histórico da fase alfabeto
│
├── input/
│   └── chat_en.jsonl           <- DATASET (fonte de verdade)
│
├── minigpt/                    <- biblioteca do modelo
│   ├── __init__.py
│   ├── chat_dataset.py         <- loader, validador, template, spans, pesos, splits, probe
│   ├── model.py                <- TinyGPT (atenção causal, MLP, KV cache, loss mascarada)
│   ├── tokenizer.py            <- ByteTokenizer (259) / CharTokenizer / ItemChTokenizer
│   ├── utils.py                <- save/load checkpoint, geração, limpeza de resposta
│   ├── config.py               <- carrega o config.json
│   ├── data.py                 <- (legado) geração de tarefas de alfabeto
│   ├── dialogues.py            <- (legado) dataset curado da fase alfabeto
│   └── curated_dataset.py      <- (legado) dataset v2 da fase alfabeto
│
├── train_chat.py               <- TREINO do SLM de conversa
├── train.py                    <- (legado) treino multi-modo da fase alfabeto
├── export_onnx.py              <- PyTorch -> ONNX
├── export_gguf.py              <- (legado) PyTorch -> GGUF/Ollama
│
├── cct/                        <- Compressed Compute Tensor (ver cct/README.md)
│   ├── README.md
│   ├── cct/                    <- tensor, índice, codecs L0–L3, ops, runtime, training
│   │   ├── native/             <- decoder C do L3 (l3_native.c) + loader ctypes
│   │   └── integrate.py        <- cctify_model / from_dense / refresh_all
│   ├── benchmarks/             <- baselines, codec_table, generation, probes
│   └── tests/                  <- run_all + suites unitárias
│
├── modelos/                    <- (gerado) chat_en.pt, chat_en.onnx, *_history.csv
└── logs/                       <- (gerado) logs de treino
```

> Os scripts `tools/` citados em versões anteriores deste README não existem
> nesta cópia; a geração de respostas/inspeção do checkpoint fica nos
> benchmarks do `cct/` e no `infer.py`.

## 8. Como usar

```powershell
# ambiente
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install onnx onnxruntime zstandard
```

```powershell
# treinar com os defaults do config.json -> modelo de 6.1M (288/8/6)
.\.venv\Scripts\python.exe train_chat.py

# reproduzir o run de 1.9M documentado em §9
.\.venv\Scripts\python.exe train_chat.py --epochs 10 --n-embd 192 --n-layer 4 `
  --n-head 6 --n-per-entry 60 --lr 2e-3 --out modelos/chat_en.pt

# retomar
.\.venv\Scripts\python.exe train_chat.py --resume --epochs 20

# validar o dataset após editar
.\.venv\Scripts\python.exe -c "from minigpt import chat_dataset as cd; print(cd.validate())"

# CCT: baselines, tabela CR x tempo e geração autorregressiva
python cct\benchmarks\codec_table.py
python cct\benchmarks\generation.py --layers 4
```

Flags úteis do `train_chat.py`: `--n-per-entry` (volume), `--val-frac`,
`--prompt-weight` / `--answer-weight` / `--stop-weight` (máscara de loss),
`--eval-probe`, `--num-threads`, `--no-onnx`, `--resume`.

## 9. Resultados medidos

### Dois testes de 10 épocas

| run | modelo | passos | train loss | val loss | ppl | probe | leitura |
|---|---|---|---|---|---|---|---|
| 1 | 6.1M | 570 | 1.52 | 2.28 | 9.75 | 0/50 | só aprendeu a **marginal** (top-5 idêntico para qualquer entrada) |
| 2 | 1.9M | 1700 | 0.049 (ep. 10) | **2.166** (ep. 3) | **8.72** | 0/50 | **memoriza** o treino; não generaliza para intenções inéditas |

Curva do run 2 (o checkpoint em `modelos/chat_en.pt` é o da época 3, melhor val):

| época | loss | val_loss | ppl |
|---|---|---|---|
| 1 | 2.93 | 2.42 | 11.2 |
| 2 | 2.04 | 2.24 | 9.4 |
| **3** | **1.66** | **2.17** | **8.7** |
| 4 | 1.28 | 2.31 | 10.0 |
| 6 | 0.53 | 2.90 | 18.1 |
| 10 | 0.05 | 3.99 | 54.0 |

### O que está comprovado

* O **pipeline** funciona de ponta a ponta: dataset → split → tensores → treino →
  validação → checkpoint → ONNX.
* A **supervisão está correta**: o texto supervisionado é exatamente a resposta
  (verificado linha a linha).
* A **máscara de loss** funciona: só a resposta contribui.
* O **length bucketing** dá ~2,5x de velocidade.

### O CCT nos pesos (medido em 2026-10-22)

No **checkpoint real** `modelos/chat_en.pt` (1.9M, melhor época 3), em BF16:

| codec | CR | decode |
|---|---:|---:|
| zstd-19 | 1.311x | 0.26 GB/s |
| blosc zstd+shuffle | **1.442x** | 0.64 GB/s |
| **CCT-L3** (zstd 19 + decoder C) | **1.503x** | 0.56 GB/s |

Ou seja: o CCT **ganha +4,2% do padrão de fato** (blosc) em compressão e perde
**12% em velocidade de decode**. Em geração autorregressiva (batch=1), o CCT
fica **4,5–7,8x mais lento por token**. A tese defensável é que ele ganha
**memória** (~31%), não tempo, em CPU pequena. Números e ressalvas completos em
[`cct/README.md`](cct/README.md) §5–§9.

O valor de 4.44x citado em versões anteriores vinha de um checkpoint pouco
treinado (pesos concentrados perto de zero) e **não** se reproduz neste
checkpoint: o ganho honesto é 1.44–1.50x.

## 10. Diagnóstico e limitações

**Por que a conversa ainda não sai boa** (medido, não suposto):

1. **Ambiguidade de alvo.** Cada intenção tem 2–3 respostas válidas; o modelo
   aprende a *mistura* e o `argmax` cai entre respostas → texto incoerente.
   Evidência: teacher-forcing do 1º token da resposta = **22,6%**.
2. **Poucos passos.** 570 passos só bastaram para a distribuição marginal; 1.700
   passos já memorizam o treino (loss 0,05) mas não generalizam.
3. **Validação por intenção inédita é dura demais** para 1,9M de parâmetros:
   o modelo não tem como generalizar o *conteúdo* de tópicos que nunca viu.
4. **Escala.** ~200 intenções e 1,9M de parâmetros é pouco para conversa aberta.

Outras limitações:

* **Sem `chat.py`** para o novo formato (o REPL existente é da fase alfabeto).
* **Byte-level** limita o contexto útil a 192 bytes.
* **Qualidade inicial baixa (medida):** o run de 1.9M (10 épocas, melhor val
  2.1696) **memoriza** o treino e não generaliza. Exemplos reais:
  `Hi!` → “Gooo God morning metiod you good”;
  `What time is it?` → “It can well the check. That for”.
  É o diagnóstico de §10.1–§10.4 (ambiguidade de alvo + poucos parâmetros).
* O CCT **tem** decoder nativo em C desde 2026-10-22. Ele bate o `blosc` em CR
  (+4,2%) e fica 12% atrás em decode, mas ainda perde para o denso em geração
  (4,5–7,8x/token). O CCT ganha **memória**, não tempo, no regime de CPU
  pequena.
* **O CR está no teto de entropia** deste checkpoint (~1,52x). Mais codec
  rende 1–2%; melhorias maiores exigem `CCT(INT4(...))` (fora do escopo
  lossless) ou um modelo com distribuição de pesos diferente.

## 11. Roadmap

1. **Alvo determinístico**: uma resposta canônica por exemplo, alternando a
   variante entre épocas → remove a ambiguidade que quebra o greedy.
2. **Validação por paráfrase** (mesmas intenções, formulações novas) além de
   intenções inéditas.
3. Mais dados + modelo **16M** e `block_size` 512.
4. `chat.py` conversacional com histórico e template.
5. **CCT**:
   * ✅ decoder nativo em C (`cct/native/l3_native.c`);
   * ✅ baselines externos (blosc/bitshuffle/lz4/zstd) e tabela CR × tempo;
   * ✅ ECD corrigido (`ECD_mem` vs `ECD_time`);
   * ⏳ provar a tese **memory-bound** com o modelo 16M (benchmark A–F do
     `Compress.md` §23) — só isso justifica CUDA/NPU.

## 12. Documentação

| arquivo | conteúdo |
|---|---|
| [`cct/README.md`](cct/README.md) | descrição completa do CCT (arquitetura, codecs, baselines, geração, ECD, conformidade, INT4) |
| [`Compress.md`](Compress.md) | especificação original do CCT (30 seções) |
| [`Projeto.md`](Projeto.md) | escopo do CCT: compressor, pipeline de utilização e framework |
| [`GUIA.md`](GUIA.md) | guia da fase alfabeto (histórico) |

---

<div align="center">

*~1,9M parâmetros, 259 tokens, 39 tópicos — treinado em CPU, sem CUDA.*

</div>
