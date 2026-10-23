# CCT — Escopo, Pipeline de Utilização e Framework

**Status:** documento de escopo (vivo). Última atualização: 2026-10-23.
**Especificação original:** [`Compress.md`](Compress.md).
**Estado medido:** [`cct/README.md`](cct/README.md).
**Alvo do produto:** o compressor e o runtime do CCT. O SLM de conversa é o
caso de uso e o benchmark — não o produto.

---

## 1. Escopo

### 1.1 O que o projeto é

Um **compressor lossless de tensores numéricos** (BF16/FP16/FP32/INT8/UINT8)
com um **runtime que computa a partir do estado comprimido**:

```
W (BF16) → encode lossless → tiles comprimidos + índice
                                    │
                                    ▼
           GEMM consome o tile → decode → multiplica → acumula
           (nunca materializa a matriz inteira)
```

O princípio chama-se **CFCS — Compute From Compressed State**. O produto tem
três partes:

| parte | o que entrega |
|---|---|
| **codec** | compressão bit-exact por tile, com acesso aleatório |
| **runtime** | cache, prefetch, decode sob demanda, CCT-GEMM |
| **framework** | API de integração (`CCTLinear`, `cctify_model`), métricas (ECD), testes |

### 1.2 O que o projeto NÃO é

- **Não é quantização.** INT8/INT4/FP8/FP4 alteram a matemática; o CCT não.
- **Não é um formato de modelo.** É uma representação de tensor; o grafo
  (ONNX/GGUF/PyTorch) continua sendo do modelo.
- **Não é o SLM.** O modelo de conversa em inglês (e depois o de cotidiano em
  BR) é o *caso de uso* que dá escala e distribuição de pesos realistas.
- **Não é um plugin do llama.cpp (ainda).** Inferência CCT hoje é
  PyTorch/CPU. GGUF/ONNX são exportações densas.

### 1.3 Regras congeladas

| permitido | proibido |
|---|---|
| compressão lossless bit-exact | INT8, INT4, FP8, FP4 |
| tiles + acesso aleatório | pruning, esparsidade forçada |
| predição + entropia própria | qualquer redução de precisão |
| decode sob demanda | "compressão" que perde informação |

Invariante que fecha qualquer discussão:

```python
torch.equal(W_original, W_reconstruido)   # True para TODO tensor
```

### 1.4 Relação com INT4 (posicionamento)

CCT é **ortogonal e cumulativo**: `CCT(INT4(BF16(W)))` é uma linha de pesquisa
válida e provavelmente a de maior ganho futuro. O escopo atual isola a
variável "compressão pura" para medir a hipótese lossless. Quantização não
entra neste marco.

---

## 2. Critérios de sucesso e métricas

### 2.1 Portas de entrada (Compress.md §29)

| porta | critério | estado atual |
|---|---|---|
| **1 — Matemática** | `decode(encode(W)) == W` bit-exact em todos os tensores | ✅ |
| **2 — Eficiência** | `CR > baseline` e `T_decode < T_mem_saved` | CR ✅; tempo ❌ (só em regime memory-bound) |
| **3 — Treinamento** | `loss(CCT) ≈ loss(BF16)` e overhead aceitável | ⚠️ harness; +230–300% com encoder Python |

### 2.2 Métricas (definições fechadas)

| métrica | definição | uso |
|---|---|---|
| **CR** | `bytes_crus / bytes_comprimidos` | tamanho |
| **T_encode** | ms por MB comprimido | custo offline / encode-on-update |
| **T_decode** | ms por MB descomprimido (mediana, após warm-up) | gargalo de inferência |
| **ECD_mem** | `ops_úteis / bytes_movidos` | densidade de memória (≈ 1/CR) |
| **ECD_time** | `ops_úteis / (T_decode + T_compute)` | **métrica de decisão** |
| **ECD_eff** | `ECD_time / ECD_mem` | sanity: deve cair quando o decode domina |

`ECD_mem` sozinho é proporcional a `1/CR`; reportá-lo como "ganho de compute"
é circular. Toda comparação deve trazer `ECD_time` junto.

### 2.3 Condições de medição

- Sempre informar: CPU, threads, `tile_rows`, nível zstd, tamanho da amostra.
- Decode: mediana de ≥5 execuções após warm-up; registrar se a máquina estava
  carregada (timings variam 2x).
- CR é estável e pode ser comparado entre execuções; tempo absoluto não.
- Comparar contra **blosc + shuffle/bitshuffle + lz4/zstd**, não só zstd puro.

---

## 3. Modelo de dados e formato

### 3.1 Estruturas

```
CCTTensor
 ├── shape, dtype
 ├── tiles: list[bytes]          # payloads independentes
 ├── index: TileIndex            # offset, compressed_size, raw_size, codec
 └── codec_name

TileIndex (serializado: magic CCTI)
 ├── tile_size, shape, dtype
 ├── row_tiles, tile_cols        # modo "bloco de linhas" para o GEMM
 └── entries[]: (tile_id, offset, compressed, raw, codec)
```

### 3.2 Tiling

- **2-D + `tile_rows`**: cada tile é um bloco de linhas `(tile_rows, cols)`.
  É o único modo que permite `CCT-GEMM` tile a tile.
- **1-D/outros**: tiles planos de `tile_size` elementos.
- Não há padding: o último tile é menor. Em tensores pequenos o cabeçalho por
  tile pesa mais (CR do L3 cai de ~1,50x para ~1,43x).

### 3.3 Codecs

| nome | pipeline | uso |
|---|---|---|
| `l0_zstd` | zstd no BF16 interleaved | baseline |
| `l1_entropy` | split alto/baixo + Huffman próprio | entropia própria (decode lento) |
| `l2_bitplane` | bit-planes + RLE + Huffman | não competitivo |
| **`l3_shuffle`** | sinal 1 bit + expoente + mantissa 7 bits, zstd por fluxo | **codec principal** |
| `hybrid` | melhor codec por tile | garante CR ≥ 1 |
| `l3_structured` | alias de `l3_shuffle` | compatibilidade |

Contratos:

- `l3_shuffle` exige `itemsize == 2`; caso contrário o `CCTTensor` cai para
  `hybrid` automaticamente.
- Decompressão é **bit-exact**; `cct_matmul` **não** é (ordem de acumulação,
  `rel_err ≈ 3.7e-07`).
- O decoder nativo em C é opcional; sem ele, o caminho numpy é equivalente em
  resultado (e ~23x mais lento).

### 3.4 Formato em disco

Hoje: `[TileIndex][tiles concatenados]` (`CCTTensor.save`).

Pendência declarada: `format.py` com **versão, checksum e flags** no
cabeçalho. Enquanto não existir, arquivos `.cct` não devem ser tratados como
formato estável.

---

## 4. Pipeline de utilização

### 4.1 Offline — comprimir

```python
from cct import CCTTensor

cct = CCTTensor.from_tensor(W, codec="l3_shuffle", tile_rows=192)
cct.save("W.cct")
print(cct.compression_ratio)
```

Requisitos: tensor contíguo; BF16/FP16 para o L3. Encode é offline (não
otimizado: ~240 ms/MB no zstd 19).

### 4.2 Inferência — CFCS

Duas formas, mesma matemática:

```python
# A) tensor isolado
from cct import cct_matmul
y = cct_matmul(cct, x)          # x 2-D: (in_features, batch)

# B) modelo inteiro
from cct.integrate import cctify_model, cct_stats
info = cctify_model(model, codec="l3_shuffle", tile_rows=192)
model.eval()
y = model(x)                    # nn.Linear -> CCTLinear -> cct_matmul
```

Não materializa `W`: itera tiles, `load_tile → matmul → acumula`.

### 4.3 Treino — encode-on-update

```python
for x, y in loader:
    opt.zero_grad(); loss = model(x, y); loss.backward(); opt.step()
    refresh_all(model)          # recompressão após o update
```

Limite honesto: em treino o peso **precisa** existir descomprimido para o
AdamW. A economia de memória real fica em aberto; o que está entregue é a
harness de medição (`measure_encode_on_update`).

### 4.4 Exportação e publicação

| artefato | contém CCT? | observação |
|---|---|---|
| `.cct` | ✅ | tensor comprimido + índice |
| `.pt` (PyTorch) | ❌ | pesos densos + config |
| `.onnx` | ❌ | grafo denso |
| `.gguf` (Ollama) | ❌ | grafo denso gpt2 |

Para publicar "o mesmo modelo em duas representações": publicar o `.pt`/`.gguf`
e um `.cct` por tensor (ou um pacote com o índice global). Ainda **não** há
runtime CCT para llama.cpp/Ollama.

### 4.5 Quando usar CCT (guia de decisão)

```
O workload é memory-bound?
  (modelo grande, banda de memória saturada, GPU/NPU com FLOPs sobrando)
        │ sim                         │ não (CPU pequena, compute-bound)
        ▼                             ▼
  CCT pode ganhar tempo        CCT perde tempo; ganha memória (~31%)
  (decode se sobrepõe)         (use denso, ou só para armazenar/transportar)
```

Medido em CPU pequena (1.9M, batch=1): **4,5–7,8x mais lento por token**.
Memória: ~31% menor. Essa é a tese defensável; prometer speedup em CPU é
desonesto.

### 4.6 Comandos do pipeline

```powershell
# build do decoder nativo (opcional)
python -c "import cct.native as n; n.build()"

# testes
python cct\tests\run_all.py

# medir compressor e baselines
python cct\benchmarks\codec_table.py
python cct\benchmarks\baselines.py --checkpoint modelos\chat_en.pt
python cct\benchmarks\l3_check.py modelos\chat_en.pt

# medir geração (dense vs CCT)
python cct\benchmarks\generation.py --layers 4 --tile-rows 192

# SLM (caso de uso)
python train_chat.py --epochs 10 --n-embd 192 --n-layer 4 --n-head 6 `
  --n-per-entry 60 --lr 2e-3 --out modelos/chat_en.pt
python export_gguf.py modelos/chat_en.pt --chat --out modelos/chat_en.gguf `
  --ollama-name chat-en
```

---

## 5. Framework

### 5.1 Estrutura de módulos

```
cct/
├── cct/
│   ├── tensor.py          CCTTensor (from_tensor, load_tile, decompress, save/load)
│   ├── index.py           TileIndex + TileEntry (serialização)
│   ├── entropy.py         Huffman canônico próprio (formato HF2)
│   ├── codec/
│   │   ├── __init__.py    L0/L1/L2, hybrid, get_codec, CODECS
│   │   └── l3_structured.py  L3 (shuffle + zstd por fluxo)
│   ├── native/
│   │   ├── __init__.py    loader ctypes + build + autoteste + fallback
│   │   └── l3_native.c    decoder C (remontagem BF16)
│   ├── ops.py             cct_matmul, cct_matmul_streaming
│   ├── runtime/cpu.py     TileCache (LRU), prefetch, medida de GB/s
│   ├── training/          CCTLinear, encode-on-update, ECD
│   └── integrate.py       cctify_model, refresh_all, cct_stats, uncctify_model
├── benchmarks/            baselines, codec_table, generation, probes
└── tests/                 run_all + 5 suites
```

### 5.2 API pública

| símbolo | contrato |
|---|---|
| `CCTTensor.from_tensor(t, codec, tile_rows)` | comprime; `tile_rows` só para 2-D |
| `CCTTensor.load_tile(i)` / `load_tile_tensor(i)` | acesso aleatório; bytes/tensor do tile |
| `CCTTensor.decompress()` | materializa (verificação/teste; não é o caminho de inferência) |
| `CCTTensor.save/load` | serialização `[índice][tiles]` |
| `cct_matmul(cct, x)` | `Y = W @ x`, `x` 2-D; autograd em `x` |
| `cct_matmul_streaming(cct, x, cache)` | variante com `TileCache` |
| `TileCache(max_tiles)` | LRU + `prefetch` + `hit_rate` |
| `CCTLinear.from_dense(linear, codec, tile_rows)` | substitui `nn.Linear`; já comprime |
| `cctify_model(model, ...)` | troca projeções; ignora embeddings/head |
| `refresh_all(model)` | recompressão pós-`optimizer.step()` |
| `ecd/ecd_time/ecd_report` | métricas de memória e tempo |
| `cct.native.available()/build()/unpack_bf16()` | decoder C |
| `get_codec(name)`, `compress_best(data)` | codecs |

### 5.3 Contratos técnicos

- **Precisão:** decompressão bit-exact; GEMM aproximado só na ordem de soma.
- **Threads:** `TileCache` usa lock; compressores zstd são thread-local.
- **Endianness:** BF16 little-endian; o decoder C escreve byte a byte.
- **Fallback:** sem compiler C, sem `zstandard`, ou `itemsize != 2` → caminhos
  alternativos automáticos, nunca erro silencioso de resultado.
- **Dependências:** núcleo = `numpy`, `torch`, `zstandard`. Baselines =
  `blosc2`, `lz4` (opcionais). Testes = `pytest` (opcional). Decoder nativo =
  `gcc`/`clang`/`cc`.

### 5.4 Exemplo de integração

```python
import torch.nn as nn
from cct.training import CCTLinear

# antes
self.fc = nn.Linear(192, 768)

# depois (inferência)
self.fc = CCTLinear.from_dense(nn.Linear(192, 768),
                               codec="l3_shuffle", tile_rows=192)
```

---

## 6. Estado medido (2026-10-23)

### 6.1 Compressão vs teto

Checkpoint real (`modelos/chat_en.pt`, 1.905.792 valores BF16):

| campo | entropia (bits/valor) |
|---|---:|
| sinal | 1,000 |
| expoente | 2,567 |
| mantissa | 6,971 (de 7) |
| **total ordem-0** | **10,538 ⇒ CR máx. 1,518x** |

L3 (zstd 19, tile 64) entrega **10,64 bits/valor (CR 1,503x)**: 0,7% do teto.

### 6.2 Codec vs baselines (mesmo checkpoint)

| método | CR | decode GB/s |
|---|---:|---:|
| zstd-19 | 1,311x | 0,26 |
| blosc zstd+shuffle | 1,442x | 0,64 |
| blosc zstd+bitshuffle | 1,391x | 0,99 |
| **CCT-L3** | **1,503x** | **0,56** |

### 6.3 Decoder nativo

| decode L3 (4.19 MB) | tempo | GB/s |
|---|---:|---:|
| numpy | 173 ms | 0,024 |
| C nativo | 7,5 ms | 0,56 |

~23x; resultado bit-exact nos dois caminhos.

### 6.4 Geração autorregressiva (1,87M, batch=1, KV cache)

| variante | ms/token | vs denso |
|---|---:|---:|
| denso fp32 | 3,07 | 1,00x |
| CCT-L3, tile 64 | 23,89 | 7,79x mais lento |
| CCT-L3, tile 192 | 16,30 | 4,45x mais lento |

Decode = 39–44% do tempo CCT; o resto é laço Python + GEMMs pequenos por tile.

### 6.5 Fase 4 (encode-on-update)

`CCTLinear(64→128)`: CR 1,52–1,57x; step denso 0,54–0,78 ms; com recompressão
1,77–3,19 ms (**+230–300%**). LCT reporta +7,2% — a diferença é o encoder
Python.

### 6.6 SLM (caso de uso)

- 1.866.432 params (192/6/4), 10 épocas, melhor `val_loss` 2,1696.
- Probe em intenções inéditas: 0/50. O modelo memoriza; qualidade inicial
  baixa. É problema de modelo/dados, **não** do compressor.
- Artefatos: `.pt`, `.onnx`, `.gguf` (7,67 MB), modelo `chat-en` no Ollama.

---

## 7. Roadmap

| fase | descrição | estado |
|---|---|---|
| P0 | Dataset + modelo 16M | ⏳ (1,9M/6,1M prontos) |
| P1 | Baseline BF16 CPU | ✅ |
| P2 | CCT-L0 zstd | ✅ |
| P3 | CCT-L1 entropy | ✅ (decode inviável) |
| P4 | CCT-L2 bit-plane | ✅ (não competitivo) |
| **P4b** | **CCT-L3 field shuffle** | ✅ **1,50x** |
| P5 | Treino dinâmico | ⚠️ harness |
| **P6** | **CPU otimizada (decoder nativo)** | ✅ C; 12% atrás do blosc |
| **P6b** | **runtime: fusão de tiles / cache** | ⏳ **próximo** |
| P7 | NPU | ⛔ sem hardware |
| P8–P10 | CUDA / fused / Tensor Core | ⛔ sem GPU |

**Próximo passo (P6b), em ordem de ganho esperado:**

1. **Fundir tiles por camada**: decodificar um grupo para um buffer transitório
   e fazer **um** GEMM por camada (memória = uma camada). Meta: 7,8x → ~3x.
2. **Cache LRU de tiles decodificados**: a partir do 2º token o decode some,
   ao custo de materializar pesos (mede memória × tempo).
3. **Batch > 1**: amortiza o decode por token.
4. Só depois: 16M (P0) + benchmark A–F; dataset de cotidiano em BR.

---

## 8. Testes e critérios de aceitação

| suite | cobre |
|---|---|
| `test_entropy.py` | Huffman: `Kraft = 1`, códigos >16 bits, regressão BF16 4 MB |
| `test_native_l3.py` | decoder C vs numpy, tamanhos 1..N e ímpares, ±inf/NaN |
| `test_integrate.py` | `from_dense`, entrada 3-D, `cctify_model`, `uncctify_model` |
| `test_l3_fase4.py` | L3 bit-exact (2-D/1-D/ímpar/zeros/denormais), `hybrid ≥ max`, ECD |
| `test_bit_exact.py` | codecs, `CCTTensor`, CCT-GEMM, cache/prefetch |

Critério de aceitação de release:

1. `python cct/tests/run_all.py` verde;
2. `torch.equal` em todos os tensores do checkpoint;
3. `l3_shuffle ≥ l0_zstd` em CR no checkpoint;
4. decoder nativo (se compilado) bit-exact contra numpy;
5. números de benchmark atualizados em `cct/README.md`.

---

## 9. Riscos e questões abertas

| risco / questão | impacto | mitigação |
|---|---|---|
| CR no teto de entropia (~1,52x neste checkpoint) | baixo | documentar; não prometer 2x |
| decode 12% atrás do blosc | médio | P6b (fusão/cache) |
| geração 4,5–7,8x mais lenta | alto | só resolve em regime memory-bound |
| `format.py` ausente (sem versão/checksum) | médio | próximo item de formato |
| sem runtime CCT em llama.cpp/GGUF | alto para deploy | projeto separado (P8+) |
| encoder Python lento (~240 ms/MB) | médio | encode é offline |
| modelo 1,9M memoriza / não generaliza | alto para o SLM | dados/alvo determinístico (fora do CCT) |
| sem GPU/NPU testada | alto para P7–P10 | hardware indisponível |

---

## 10. Changelog do escopo

| data | mudança |
|---|---|
| 2026-10-23 | reescrito como escopo de engenharia: compressor, pipeline e framework; incorpora teto de entropia, baselines, ECD corrigido, decoder nativo e o artefato inicial do SLM |
| (anterior) | crítica do README, usada como lista de diretrizes; substituída por este documento |
