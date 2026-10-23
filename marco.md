# Marco do projeto — Mini-GPT especialista em alfabetos

Data: 2026-10-15
Status: **funcional de ponta a ponta (CPU), com um item de modelagem em aberto
(indução "depois/antes/posição").**

---

## 1. Objetivo

Treinar um Transformer decoder-only minúsculo que **lê um alfabeto em JSON** e
responde perguntas sobre ele, servindo de base de estudo para treino e inferência
em CPU e posterior exportação para NPU (sem CUDA por enquanto).

---

## 2. O que foi realizado

### Infraestrutura
- Ambiente `.venv` com **PyTorch 2.14.1+cpu**, `onnx`, `onnxruntime`, `numpy`.
- Contorno de proxy corporativo (SSL) via `--trusted-host` documentado no `GUIA.md`.

### Dados
- Pasta `input/` com `alfabeto.json` (letras-base). O treino lê esse arquivo.
- `minigpt/data.py`:
  - gera alfabetos como **subconjuntos aleatórios embaralhados** de A-Z (tamanho 6-10);
  - tarefas por categoria: `count, first, last, next, prev, position, last_n`;
  - prompts **curtos** (`Q:`/`A:`) para reduzir o contexto;
  - **batches on-the-fly**: alfabetos novos a cada passo (dados "infinitos"),
    impedindo memorização;
  - conjunto de validação com alfabetos **disjuntos** do treino.

### Modelo (`minigpt/model.py`)
- Transformer pré-norm com atenção causal multi-cabeça, MLP+GELU, LayerNorm.
- **Weight tying** (embedding de token = cabeça de saída).
- **KV cache** opcional para geração token-a-token (usado na inferência).

### Treino (`train.py`)
- AdamW + **warmup + cosine**, gradient clipping.
- **Barra de progresso** e avaliação periódica em alfabetos novos.
- Salvamento automático do **melhor checkpoint** (menor val loss).
- `--loss full` (next-token denso) **ou** `--loss answer` (só a resposta).
- `--on-the-fly` e `--tasks` para experimentos/currículo.
- Avaliação final **por categoria**.

### Inferência e demais scripts
- `infer.py`: `--eval` (por categoria, alfabetos novos), `--prompt`, `--letters`.
- `chat.py`: chat do **modelo especialista** (`/alfabeto`, `/exemplos`, `/help`).
- `export_onnx.py`: export **ONNX** (+ `--int8`). Paridade PyTorch×onnxruntime
  validada (diff ~7.6e-6) e eixos dinâmicos OK.
- `modelos/especialista.pt` (462.720 parâmetros) e `modelos/especialista.onnx`.

### Descobertas experimentais (importantes)
1. **Memorização × generalização:** com poucos alfabetos fixos o modelo **decora**
   (treino→0, val→alto). Com dados on-the-fly a memorização é impossível.
2. **A loss importa muito:** com supervisão **esparsa** (`--loss answer`) o modelo
   **não formava circuitos de cópia** (first/last ~0-7%); com **`--loss full`**
   passou a generalizar: `count/first/last` = 100%, `last_n` = 93%.
3. **Contexto curto ajuda:** encurtar prompts reduziu a sequência de ~245 → ~74
   tokens (mais rápido e mais fácil de aprender).

---

## 3. Resultado atual (alfabetos não vistos)

Modelo `modelos/especialista.pt` (loss=full, 6-10 letras):

| categoria | acurácia |
|-----------|----------|
| count     | 100.0% |
| first     | 100.0% |
| last      | 100.0% |
| last_n    | 93.3%  |
| position  | 13.4%  |
| next      | 13.5%  |
| prev      | 12.5%  |
| **geral** | **40.5%** |

Leitura: **cópia/contagem generalizam**; **indução "depois/antes/posição" não**
(≈ nível de chute). Teste isolado (só `next`/`prev`, 3000 passos) confirmou ~9%.

---

## 4. O que falta implementar (amanhã)

### Prioridade alta — fazer o modelo aprender indução (`next`/`prev`/`position`)
Hipóteses e experimentos a rodar:
1. **Mais computação/treino:** 5k-10k passos; batch maior; medida da val loss por
   categoria ao longo do tempo (não só a global).
2. **Capacidade/cabeças:** `--n-embd 128/192`, `--n-layer 6`, mais cabeças
   (induction heads costumam exigir composição de 2+ camadas).
3. **Posicional:** testar **RoPE** em vez de embedding posicional absoluto
   (ajuda generalização de comprimento e circuitos de atenção).
4. **Currículo:** pré-treinar em `count/first/last/last_n` e depois focar em
   `next/prev`; ou pesos por tarefa no loss.
5. **Dados:** formatos que favoreçam a "cabeça de token anterior" (ex.: expor
   pares vizinhos de forma mais explícita), e checar se `position` ajuda `next/prev`.
6. **Otimização:** varrer LR (1e-3, 3e-3, 6e-3), warmup, `weight_decay`,
   `label_smoothing`, e grad accumulation para batch efetivo maior.

### Prioridade média — engenharia/entrega
- **ONNX com KV cache** (grafo de inferência incremental) e **INT8**.
- Testes automatizados (smoke de `build_dataset`, `sample_batch`, geração com/sem
  cache, paridade ONNX).
- `chat.py`: multi-turno, histórico, `--list-alfabetos`, melhor UX.
- Suporte a **tokenizador BPE** (`tokenizers`) como alternativa ao char-level.
- (Opcional) rodar treino em **GPU** se disponível (habitual: treino em GPU,
  inferência em NPU).

### Prioridade baixa
- `README.md` na raiz; empacotar em CLI único (`minigpt train|chat|eval`).
- Mais presets de alfabetos e tarefas (ex.: "letra na posição N", "soma de posições").

---

## 5. Como reproduzir hoje

```powershell
# treinar (padrão: on-the-fly + loss full, 6-10 letras)
.\.venv\Scripts\python.exe train.py

# avaliar em alfabetos não vistos
.\.venv\Scripts\python.exe infer.py --eval

# conversar
.\.venv\Scripts\python.exe chat.py

# exportar ONNX
.\.venv\Scripts\python.exe export_onnx.py
```

Referência de uso detalhada: `GUIA.md`.

---

## 6. Atualização — 2026-10-16 (especialista + tokenização + diagnósticos)

### O que foi implementado
- **`config.json` central** (+ `minigpt/config.py`): todos os parâmetros de
  modelo/dados/treino num único arquivo; `train.py` e `train_specialist.py` o usam
  como defaults e a CLI sobrepõe. Escolha com `--config`.
- **Tokenizador por item** (`minigpt/tokenizer.py`, `ItemChTokenizer`): cada item
  do alfabeto é um **token atômico** (funciona para letra *e* palavra). Fronteiras
  limpas: `JSON` / `Q:` / `A:` / resposta coincidem com fronteiras de token.
- **Tarefas novas**: `show` ("Mostre o alfabeto?") e `first_n` ("Mostre as N
  primeiras?") — cópia de lista/prefixo (`minigpt/data.py`).
- **`answer_weight`** (peso maior nos tokens da resposta) e **`aux` pointer loss**,
  **`copy_head`**, **RoPE** e **readout por cosseno** — flags em `train.py`.
- **`train_specialist.py`**: treino de um **especialista** com dataset fixo
  (alfabetos do `input/`), salvo em `modelos/especialista.pt`.
- `chat.py`: perguntas `show`/`first_n`, comando `/alfabetos`, aviso de alfabeto
  fora do conjunto conhecido.

### Verificações de tokenização (todas OK)
1. round-trip: cada letra/dígito = 1 id; `decode(encode(x))==x`.
2. mesmo id no input (JSON) e no output (resposta).
3. `aux_s` alinhado: `x[src] == target` em 100% das posições de resposta.
4. fronteira do separador (`A: `) limpa; item multi-caractere vira 1 token.

### Resultados (testes de 200–300 passos, CPU)
- **Cópia/extracção generaliza**: `show`, `first_n`, `last_n`, `count`, `first`,
  `last` = ~100%.
- **Especialista (300 passos)**: responde corretamente, para alfabetos conhecidos,
  `Mostre o alfabeto`, `Mostre as N primeiras`, `Depois/Antes de X`, `Total`,
  `Ultima`, `Ultimas N`.
- **`next/prev` memoriza (100%) mas NÃO generaliza** para alfabetos novos.

### Diagnósticos que descartamos
- **RoPE**: não ajuda (fica abaixo do posicional aprendido).
- **Readout por cosseno**: piora.
- **Pointer loss / copy-head**: pioram (o copy-head chegou a quebrar `last_n`).
- **Toy de indução pura** `[a1..a8][a_i]→a_{i+1}`: a arquitetura aprende (85%),
  mas **lento** (~5000 passos) — os induction heads emergem devagar, não em 200.
- **Toy `position`** exige contar/emitir número (não é cópia); não memoriza em 300.

### Conclusão / próximo passo
O gargalo é **generalização de `Depois/Antes de X`** (match+copy), que emerge
lentamente. Cópia e especialização funcionam; o especialista atende o uso atual.

### Como reproduzir (especialista)
```powershell
.\.venv\Scripts\python.exe train_specialist.py
.\.venv\Scripts\python.exe chat.py
# ou:
.\.venv\Scripts\python.exe infer.py --letters ABCDEFGHIJ --prompt "Mostre o alfabeto?"
```

---

## 7. Atualização — 2026-10-19 (chat via ONNX + NLU por regras)

### O que foi implementado
- **Backend ONNX no `chat.py`/`infer.py`** (`--backend pt|onnx`, `--onnx-path`):
  `minigpt/utils.py` ganhou `load_tokenizer_config`, `load_onnx_session`,
  `resolve_onnx_path`, `generate_continuation_onnx` (loop token-a-token via
  onnxruntime) e `evaluate_exact_match_onnx`. Paridade pt×onnx ~5e-6,
  respostas idênticas (0.11s vs 0.13s/resp no A-Z).
- **Auto-export ONNX**: `export_onnx.py` expõe `export_checkpoint_to_onnx()`;
  `train.py` e `train_specialist.py` reexportam o `.onnx` ao fim de todo
  treino (`--no-onnx` pula, `--onnx-int8` quantiza).
- **Alfabeto do dataset por padrão**: chat/infer usam `train_alphabets[0]`
  sozinhos (`/usar N` troca); `--max-new-tokens` virou `auto (block_size)`.
- **NLU por regras** (`canonicalize_question` em `minigpt/data.py`): traduz
  frase livre → pergunta canônica (`quantas letras tem?` → `Total?`).
  Forma desconhecida não quebra mais (sugere as válidas).
- `GUIA.md` reescrito (foco: treinar + conversar).

### Decisão registrada: ONNX segue sem KV-cache
Grafo = forward de sequência completa (recompute por token). Custo irrelevante
nesta escala; cache incremental ficaria para modelo maior (8+ tensores
past/present, `pos_offset`, suporte irregular nas NPUs).

---

## 8. Rumo ao chat cotidiano — plano e FASE 1 (2026-10-19)

### Diagnóstico (arquitetura atual vs. conversa livre)
| camada | hoje | cotidiano exige |
|---|---|---|
| Tokenizer (item/char, vocab 61) | ótimo p/ A-Z + templates | BPE/subword 8k–32k p/ texto livre |
| Modelo (~470k, block ≤192) | decora 26 letras | 10M–100M+ p/ coerência mínima |
| Dados (sintético, 1 turno c/ JSON) | tarefa fechada | milhares de diálogos multi-turno |
| Treino CPU/minutos | ok p/ tiny | GPU vira requisito |
| Chat (1 pergunta isolada, greedy) | quiz do alfabeto | histórico + system prompt + sampling |
| Eval (exact-match) | tarefa fechada | perplexidade + avaliação humana |

### Fases
- **FASE 1 (agora)**: tirar o regex *dentro do domínio do alfabeto* —
  expandir `make_examples` com paráfrases por intenção (sem acento, para não
  explodir o vocab), retreinar os especialistas, conversar em modo `--raw`
  (texto cru ao modelo, sem `canonicalize`). CPU, dias não — horas.
- **FASE 2 (futuro)**: chat cotidiano pequeno de verdade — BPE + corpus
  dialógico PT + modelo 10–50M + block 512–1024 + GPU + chat com histórico.
- **FASE 3 (recomendação se a meta for qualidade)**: fine-tuning de pequeno
  pré-treinado em PT em vez de treinar do zero; infra de chat/ONNX reaproveitada.

### Critério de sucesso da FASE 1
Perguntas parafraseadas **em modo `--raw`** (sem regex) respondidas
corretamente nos alfabetos conhecidos — ex.: `quantas letras tem?` → `26`,
`mostre a ultima letra` → `Z`, `qual vem depois do d?` → `E`.

### Andamento FASE 1 — fim do dia 2026-10-19 (continua amanhã)
Implementado:
- `make_examples(..., variants=True)`: paráfrases por intenção (count 8 formas
  incl. `Quantas letras sao no alfabeto?`; first/last/show 3–4; first_n/last_n
  2–3; next/prev 3; position 2) + `_case_variants` (caixa alta/baixa, letra
  solta minúscula e forma sem `?`). `variants=False` p/ `/exemplos`.
- Modo `--raw` em `chat.py`/`infer.py` (texto cru ao modelo) + guarda
  `KeyError` amigável nos dois.
- Retreinos: `especialista.pt` (500 steps, ~142s) e `especialista_full.pt`
  (800 steps, ~544s, **com `position` incluída via `--tasks`**); `.onnx`
  reexportado sozinho nos dois.
- Achados no caminho: minúscula `q` fora do vocab (`KeyError`) → resolvido
  com variantes de caixa (vocab 61→73); `posição da letra C` capturava o `L`
  de "letra" → ordem das regras corrigida; `first_n` com `k=1` devolvia 6
  letras (sub-memorização rara).

Medido (modelo full anterior, 400 steps, sem `position` no treino):
`726/938 = 77.4%` em todas as paráfrases; **99.5% nas tarefas treinadas**
(count/first/last/show/last_n/next/prev 100%, first_n 90%); `position` 0%
— esperado, não estava no treino desse checkpoint.

Pendente amanhã:
1. Rodar a bateria `--raw` nos modelos finais (frases do critério + `eval_para.py`)
   e anotar o placar aqui.
2. Confirmar `position` no full (foi incluída no retreino de 800 steps).
3. Se `first_n k=1` persistir, investigar (possível confusão `1`×quantidade).
4. Atualizar `GUIA.md` com o modo `--raw` e o placar final.

---

## 9. Atualizacao — 2026-10-21 (dataset curado + GGUF gpt2 + Ollama a funcionar)

### O que mudou
- **Dataset curado unico**: `input/dialogos.jsonl` (606 dialogos: 165 de
  alfabeto + 441 de cotidiano, ~200 intencoes). Fontes unicas de verdade;
  `input/alfabeto.json` fica so com as letras-base. Validado por
  `minigpt.dialogues.validate()` (respostas recomputadas, sem duplicatas).
- **Treino unico** `train.py --mode especialista|generalista|slm`; o modo
  especialista passa a consumir o dataset curado (`--dataset`).
- **TinyGPT de 6.1M** (n_embd 288, 6 camadas, 8 heads, ctx 256), 50 epocas:
  probe **28/28 = 100%**.
- **Tokenizador byte-level** (`minigpt/tokenizer.py:ByteTokenizer`, 259 tokens):
  nenhum char desconhecido, acentos incluidos, e export GGUF 1:1.

### Exportacao GGUF -> Ollama (o que foi discoveries)
1. O TinyGPT exporta como **`gpt2`** (arquitetura real do llama.cpp).
   Como "tinygpt" o Ollama recusa: `unknown model architecture: 'tinygpt'`.
2. `attn_qkv` fundido, `feed_forward_length` explicito (4x n_embd),
   LayerNorm (nao RMSNorm) — o llama.cpp gpt2 usa LayerNorm.
3. Os nomes dos tokens vao no **byte-encode do GPT-2** (espaco -> `G`):
   o tokenizer BPE do llama.cpp aplica essa transformacao antes de dividir o
   texto. Sem ela, o espaco cai no fallback e vira `[UNK_BYTE_0x20]`.
4. **CRLF no Modelfile = resposta corrompida** (Windows): o `\r` entra no
   prompt. O exportador grava com `newline="\n"`.
5. O **template** monta o prompt de treino; sem ele o modelo devolve o eco
   do JSON. Com o template escreve-se so a pergunta.

### Resultado no Ollama
```
ollama run tinygpt "Depois de C?"                  -> D
ollama run tinygpt "Qual a última letra do Alfabeto?" -> Z
ollama run tinygpt "Total?"                          -> 26
ollama run tinygpt "Ola"  -> "Olá! Que bom te ver. Quer me fazer uma pergunta sobre o alfabeto?"
bateria (28 perguntas, 14 tarefas, A-Z): 28/28 = 100%
```

### CCT (Compress.md) — inicio
`cct/` com CCTTensor, TileIndex, codecs L0/L1/L2 + hybrid, Huffman proprio,
cache/prefetch e **CCT-GEMM** (decode on-the-fly). Testes: **35/35**,
bit-exact em todos os 27 tensores 2-D do checkpoint. Conclusao honesta: a
matematica esta provada; o codec ainda perde para o zstd (CR 1.08x) e o
decoder em Python e o gargalo (0.33 GB/s). Detalhes em `cct/README.md`.

---

## 10. Atualizacao — 2026-10-21 (fecho do dia: dataset v2, GGUF gpt2 no Ollama, CCT)

### Feito hoje
1. **Exportador unico** `export_gguf.py` (substitui `export_tinygpt_gguf.py`):
   deteta o tipo de entrada — `.pt` do TinyGPT -> GGUF **gpt2** (llama.cpp),
   ou pasta HF -> GGUF via `convert_hf_to_gguf.py` (+ `--quant`). Gera o
   `Modelfile` oficial em `ollama/` e faz `ollama create` com `--create`.
2. **TinyGPT de 6.1M** (n_embd 288, 6 camadas, 8 heads, ctx 256, 6.143.328
   parametros) treinado 50 epocas sobre o dataset curado: probe **28/28 = 100%**.
3. **GGUF gpt2 funciona no Ollama.** Aprendizados (todos em `GUIA.md`):
   - exportar como "tinygpt" e recusado (`unknown model architecture`);
   - precisa `attn_qkv` fundido, `feed_forward_length` explicito, LayerNorm;
   - os nomes dos tokens vao no **byte-encode do GPT-2** (espaco -> `G`);
     sem isso o espaco cai no fallback e vira `[UNK_BYTE_0x20]`;
   - o `Modelfile` tem de ser escrito com **LF** (CRLF no Windows mete um
     `\r` no prompt e corrompe a resposta);
   - o **template** monta o prompt de treino; com ele escreve-se so a pergunta.
   Resultado: `ollama run tinygpt "Depois de C?"` -> `D`, `Total?` -> `26`,
   bateria 28/28 = 100%, e a conversa do dia a dia responde ("Ola" -> "Ola! Que
   bom te ver...").
4. **Tokenizador byte-level** (`minigpt/tokenizer.py`): 259 tokens, nenhum char
   desconhecido, acentos incluidos, export GGUF 1:1.
5. **Dataset curado** `input/dialogos.jsonl` (606 dialogos: 165 alfabeto +
   441 cotidiano / ~200 intencoes), com `minigpt/dialogues.validate()` a
   conferir cada resposta e a recusar duplicatas.
6. **Dataset v2** `input/intents.jsonl` (310 intents: 105 templates de alfabeto
   + 205 de cotidiano) + `minigpt/curated_dataset.py`: os templates curados
   sao instanciados sobre **varios alfabetos** (respostas RECOMPUTADAS e
   verificadas) -> o modelo tem de LER o JSON, nao decorar pares. Treino:
   3560 exemplos; **eval com alfabetos nunca vistos** (2940 exemplos).
   Probe do treino passa a medir o eval real.
7. **CCT (Compress.md)** iniciado em `cct/`: CCTTensor, TileIndex, codecs
   L0/L1/L2 + hybrid (best-of por tile), Huffman canonico proprio, cache LRU
   com prefetch, **CCT-GEMM** (decode sob demanda, sem materializar W).
   Testes **35/35**: bit-exact em todos os 27 tensores 2-D do checkpoint,
   GEMM com rel_err 3.7e-07, gradiente correto. Matriz de conformidade com o
   `Compress.md` em `cct/README.md`.

### Medido (honesto)
- Dataset v1 (pares fixos): 100% no que viu, ~38% em formulas ineditas.
- CCT: matematica provada; o codec ainda perde para o zstd (CR 1.08x vs
  1.45x da referencia NWC) e o decoder em Python e o gargalo (0.33 GB/s).
- Hardware: esta CPU **nao tem AMX** (bf16 = 0.5 GFLOP/s vs 112.6 do fp32) e
  nao ha CUDA -> as fases de Tensor Core do `Compress.md` exigem GPU.

### Para amanha
**A) Modelo (prioridade)**
1. Terminar as 15 epocas do dataset v2 e medir a **generalizacao real**
   (alfabetos ineditos): e a experiencia que falhou no v1. Se `next`/`prev`
   ficarem abaixo de ~50%, o gargalo e o circuito de inducao (precisa de
   mais camadas/cabeças ou mais epocas).
2. Se generalizar: exportar v2 -> GGUF -> Ollama e reavaliar a bateria
   (alfabeto A-Z + variantes ineditas + cotidiano).
3. Robustez do cotidiano: `<eos>` colado a resposta (o modelo as vezes nao
   para) e mais variantes de superficie (caixa/pontuacao) nos intents.
4. Multi-turno no Ollama: template que renderize `.Messages` em pares Q/A
   (hoje e single-turn, por isso a "memoria" do dataset nao e usada).

**B) CCT (caminho critico)**
5. **CCT-L3**: predictor adaptativo + entropia por contexto, para ultrapassar
   o zstd (meta CR >= 1.3x) — hoje L1/L2 expandem os dados.
6. **Decoder nativo** (C/Cython/SIMD): o gargalo real e o decode em Python.
7. Medir o trade-off tipo LCT: treino com encode-on-update (Fase 4) —
   memoria vs tempo por step.

**C) Publicacao (HF)**
8. Empacotamento: model card, license, dataset card, tags, `config.json`,
   `safetensors` + GGUF + Modelfile, e um script de upload.
9. Publicar tambem os pesos **comprimidos pelo CCT** (a contribuicao nova):
   o mesmo modelo, dois representacoes, com o mesmo resultado numerico.

**D) Infra**
10. `README.md` na raiz (falta) e CLI unica `minigpt train|chat|eval|export`.
11. Testes smoke automaticos (build do dataset, geracao com/sem cache,
    paridade ONNX, paridade GGUF<->PyTorch).

### Pausa do dia (2026-10-21 17:25)
Treino do dataset v2 interrompido na **epoca 2/15** (loss 2.81 -> 1.75,
probe em alfabetos ineditos 0% -> 3.6%, ainda a subir). Checkpoint da epoca 2
em `modelos/tinygpt_v2.pt` + `modelos/tinygpt_v2_history.csv`.
**Para retomar** (mesmos argumentos, ~6 min/epoca, ETA ~80 min):

```powershell
.\.venv\Scripts\python.exe train.py --mode especialista --n-alphabets 20 `
  --epochs 15 --batch-size 64 --eval-every 1 --probe-per-task 2 `
  --num-threads 8 --out modelos/tinygpt_v2.pt
```

Nota: o probe mede sempre **alfabetos nao vistos** (gerados com seed diferente
da do treino) — e o que separa "memorizar" de "generalizar".

---

## 11. Atualizacao — 2026-10-22 (CCT: diretrizes do `Projeto.md` aplicadas)

O `Projeto.md` listou sete acoes prioritarias; este e o registro do que foi
feito, com numeros. Antes desta atualizacao o `marco.md` nao mencionava
baselines externos, INT4, ECD_mem/ECD_time nem a tese memory-bound.

### 11.1 Bug real corrigido no Huffman (`cct/cct/entropy.py`)

Dois defeitos encontrados ao rodar o L1 em 4 MB de BF16:

1. **Kraft = 0.5**: a construcao somava +1 em todos os comprimentos (o merge
   final era contado duas vezes) -> 1 bit desperdicado por simbolo.
2. **Campo `code` de 2 bytes**: o plano de expoente BF16 gera codigos > 16
   bits; eles eram truncados em silencio e o **decode falhava**
   (`ValueError: stream huffman invalido`).

Formato v2 (magic `HF2`, `len` 2B + `code` 4B + `sym` 1B, `Kraft = 1`, leitura
do v1 preservada). Efeito medido: **L1 foi de CR 1.25x para 1.44–1.50x** e
passou a ser bit-exact em todos os tamanhos.

### 11.2 Baselines externos (a comparacao obrigatoria do §2.3)

`cct/benchmarks/baselines.py` mede zstd (3/12/19), zstd+dict held-out, lz4,
blosc2 (`lz4`/`zstd` x `shuffle`/`bitshuffle`/sem filtro) e fpzip (opcional).
Tensor BF16 2048x1024 (4.19 MB), pesos `N(0,0.02)`:

| metodo | CR | decode GB/s |
|---|---:|---:|
| zstd-19 | 1.299x | 0.31 |
| blosc lz4+bitshuffle | 1.320x | 1.81 |
| **blosc zstd+shuffle** | **1.443x** | **0.83** |
| blosc zstd+bitshuffle | 1.363x | 1.06 |
| zstd-19+dict (held-out) | 1.284x | 0.41 |
| **CCT-L3 (zstd 19)** | **1.509x** | 0.56* |

\* decode do L3 medido depois, sem carga (mediana de 9); o blosc no mesmo
buffer faz 0,64 GB/s. A primeira medicao (0,38 vs 0,83) foi feita com a
maquina carregada.

No **checkpoint real** (3,81 MB): L3 **1,503x** vs blosc **1,442x**
(**+4,2%**) e zstd-19 1,311x.

**Conclusao:** o L3 **bate o blosc em CR** e fica **12% atras** em decode —
nao e' mais 2x. `bitshuffle` nao bate `shuffle` em BF16. O diferencial
estrutural continua sendo o runtime (tiles + random access + CCT-GEMM); o
codec agora esta' a ~0,7% do teto de entropia (§11.10).

### 11.3 ECD corrigido (§2.2 do `Projeto.md`)

`ECD = ops/bytes` era so' `1/CR` disfarçado (o antigo "4.0 -> 6.1" tinha
6.1/4.0 = 1.525 ~ CR 1.568). Agora existem:

- `ecd(flops, bytes)` -> **ECD_mem** (ops/byte);
- `ecd_time(flops, s)` -> **ECD_time** (ops/s);
- `ecd_report(...)` -> os dois + overhead + `cct_slower`.

Na amostra de 4.19 MB: ECD_mem do L3 = 1.44 ops/B (denso 0.50), mas ECD_time =
**379 Mops/s** contra **76.678 Mops/s** do denso. A metrica que decide mostra
a perda, nao um ganho.

### 11.4 Decoder nativo em C (P6)

`cct/native/l3_native.c` (build com gcc/clang; loader ctypes com autoteste
contra numpy e fallback automatico). Remonta BF16 a partir de sinal
empacotado + expoente + mantissa de 7 bits, com leituras de 64 bits.

| decode L3 (4.19 MB) | tempo | GB/s |
|---|---:|---:|
| numpy | 173 ms | 0.024 |
| C nativo | 7.5 ms | 0.56 |
| blosc zstd+shuffle | 6.6 ms | 0.64 |

~23x sobre numpy, bit-exact nos dois caminhos. Fica **12% atras** do blosc em
decode (nao 2x).

### 11.5 Geracao autorregressiva (§3.2) — o numero que decide

`cct/benchmarks/generation.py` (TinyGPT 1.87M, batch=1, KV cache):

| variante | ms/token | vs denso |
|---|---:|---:|
| denso fp32 | 3.07 | 1.00x |
| CCT-L3, tile 64 | 23.89 | **7.79x mais lento** |
| CCT-L3, tile 192 | 16.30 | **4.45x mais lento** |

Tokens gerados identicos ao modelo com pesos arredondados para BF16 (perda
zero). Trade-off novo documentado: `tile` maior reduz tiles/chamadas e o tempo
(-32%), sem perda de CR (1.503x -> 1.507x). Decode puro = 39% (tile 64) a 44%
(tile 192) do tempo CCT; o resto e' o laco Python + GEMM por tile — proximo
alvo de otimizacao. **Em CPU pequena o CCT perde tempo; ganha ~31% de
memoria.**

### 11.6 Posicionamento INT4 (§2.4) e bit-exact (§3.1)

Documentado em `cct/README.md` §11–§12: (a) a descompressao e' bit-exact, o
CCT-GEMM **nao** e' (rel_err 3.7e-07 por ordem de acumulacao); (b) CCT e'
**ortogonal e cumulativo** com INT4 (`CCT(INT4(BF16(W)))`), nao concorrente.

### 11.7 Integracao e tabela unica

- `CCTLinear.from_dense(...)`, `cctify_model`, `refresh_all`, `cct_stats`,
  `uncctify_model` (`cct/cct/integrate.py`) — o exemplo de adocao que faltava.
- `cct/benchmarks/codec_table.py`: CR x encode x decode x ECD por codec, e
  tensores pequenos/impares (L3 1.43–1.50x; zstd 1.22–1.28x).
- Paths hardcoded `C:\Dev\gpt` removidos de 9 scripts; `l3_shuffle` incluido
  no `compression.py`.

### 11.8 Estado dos artefatos (nao confundir com o historico)

O alfabeto e' **fase antiga**; o alvo agora e' o SLM de conversa em ingles
(depois um dataset de cotidiano em BR). O foco de engenharia e' o compressor.

**Gerado em 2026-10-22 (run de 1.9M, 10 epocas, melhor val 2.1696):**

- `modelos/chat_en.pt` — 1.866.432 params (192/6/4), checkpoint da epoca 3;
- `modelos/chat_en.onnx` — forward completo, exportado pelo `train_chat.py`;
- `modelos/chat_en.gguf` — 7,67 MB, arch `gpt2`, vocab 259;
- `ollama/Modelfile.chat_en` + modelo `chat-en` criado no Ollama;
- `logs/train_chat_2026-10-22.log`.

**Qualidade (honesta):** o modelo **memoriza** o treino e nao generaliza
(probe 0/50 em intencoes ineditas). Exemplos reais (PyTorch e Ollama, iguais):
`Hi!` -> "Gooo God morning metiod you good";
`What time is it?` -> "It can well the check. That for".
Nao e' um assistente utilizavel — e' o artefato inicial do pipeline.

**Correcoes de infraestrutura feitas no caminho:**

- `train_chat.py` nao criava `modelos/` antes de abrir o CSV de historico;
- `CCTLinear.forward` so aceitava entrada 2-D (quebrava na geracao com KV
  cache);
- `export_gguf.py` ganhou `--chat` (template `<bos>User: ...\nAssistant: `)
  para nao depender de edicao manual do Modelfile;
- decoder nativo C do L3 + `CCTLinear.from_dense`/`cctify_model`.

**CCT:** `format.py` continua ausente; o encoder segue em numpy (~240 ms/MB no
nivel 19); falta o modelo de 16M (P0) para o benchmark A–F do `Compress.md`
§23.

### 11.10 O teto de entropia (por que o CR esta acabando)

Medido no checkpoint real (1.905.792 valores BF16):

| campo | entropia (bits/valor) |
|---|---:|
| sinal | 1,000 |
| expoente | 2,567 (cond. ao de cima: 2,554) |
| mantissa | 6,971 (de 7) |
| mantissa MSB cond. expoente | 0,946 (de 1) |
| **total ordem-0** | **10,538 (CR max. 1,518x)** |

O L3 entrega **10,64 bits/valor (CR 1,503x)**: folga de 0,1 bit (~0,7%).
Variantes testadas: delta vertical do expoente 1,390x (pior), tile transposto
1,484x (neutro), mantissa com zstd 1,482x (neutro), fluxo unico por tensor
1,442x (pior). Nivel zstd 12 -> 19: **+1,5%** (1,483 -> 1,503) sem mudar o
decode; o compressor zstd passou a ser cacheado por thread (o encode gastava
tempo recriando-o por tile).

**Conclusao:** mais codec lossless rende 1–2% neste modelo. Para ir alem:
`CCT(INT4(BF16(W)))` (ortogonal, fora do escopo atual) ou atacar tempo de
runtime — que e' onde a geracao ainda perde 4,5–7,8x.

### 11.9 O que foi feito (ordem acordada)

1. READMEs (`cct/README.md` e raiz) — **feito**: tabelas medidas, baselines,
   geração, ECD, INT4, bit-exact, teto de entropia.
2. `marco.md` — **esta secao**.
3. Testes — **feito**: `test_entropy.py`, `test_native_l3.py`,
   `test_integrate.py`, `run_all.py`; 5 suites / 80+ assercoes passando.
4. Treino + artefatos — **feito**: `.pt`, `.onnx`, `.gguf`, `chat-en` no
   Ollama (qualidade inicial baixa, documentada).

### 11.11 Proximo passo (o que realmente sobra)

Com o CR no teto, o gargalo virou o **runtime**:
`cct_matmul` decodifica e multiplica **tile a tile**, pagando um GEMM pequeno
por tile. Na geracao de 1.9M, o decode e' 39–44% do tempo CCT; os outros
56–61% sao laco Python + GEMMs pequenos. Candidatos, em ordem de ganho
esperado:

1. **Fundir tiles por camada**: decodificar um grupo de tiles para um buffer
   transitorio e fazer **um** GEMM por camada (memoria = uma camada, nao o
   modelo). Deve levar a geracao de 7,8x para ~3x.
2. **Cache de tiles decodificados (LRU)**: a partir do 2o token o decode some,
   ao custo de materializar os pesos — mede o trade-off memoria x tempo.
3. **Batch > 1**: amortiza o decode (o caso de treino/avaliacao).
4. So' depois: 16M (P0) e o benchmark A–F; e o dataset de cotidiano em BR.

Nada disso muda o CR; muda o `T_decode` — que e' a metrica que ainda separa o
CCT de ser util.
