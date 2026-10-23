<p align="center">
  <img src="../Logotipo_CCT_%20Tensor_Compactado.png" alt="CCT — Compressed Compute Tensor" width="420">
</p>

<h1 align="center">CCT — Compressed Compute Tensor</h1>

<p align="center">
  <em>Uma representação executável de tensor: o estado matemático é lossless,<br>
  o estado físico na memória é comprimido, e o compute consome o comprimido.</em>
</p>

---

## Sumário

1. [O que é](#1-o-que-é)
2. [A regra que não pode ser quebrada](#2-a-regra-que-não-pode-ser-quebrada)
3. [Estado atual (leia antes dos números)](#3-estado-atual-leia-antes-dos-números)
4. [Arquitetura](#4-arquitetura)
5. [Os codecs (L0 → L3) e por que o L3 ganha](#5-os-codecs-l0--l3-e-por-que-o-l3-ganha)
6. [Resultados medidos](#6-resultados-medidos)
7. [Baselines externos — a comparação que decide](#7-baselines-externos--a-comparação-que-decide)
8. [Fase 4 — treinar com encode-on-update](#8-fase-4--treinar-com-encode-on-update)
9. [Geração autorregressiva — o caso de uso real](#9-geração-autorregressiva--o-caso-de-uso-real)
10. [Como integrar num modelo](#10-como-integrar-num-modelo)
11. [Bit-exact tem dois sentidos](#11-bit-exact-tem-dois-sentidos)
12. [Por que não INT4](#12-por-que-não-int4)
13. [Conformidade com o `Compress.md`](#13-conformidade-com-o-compressmd)
14. [Testes e benchmarks](#14-testes-e-benchmarks)
15. [Limitações honestas](#15-limitações-honestas)
16. [Roadmap P0–P10](#16-roadmap-p0p10)

---

## 1. O que é

O CCT nasce de uma pergunta: **e se os pesos nunca precisassem existir
descomprimidos na memória?**

A resposta prática do mercado é *quantizar* (INT8/INT4/FP8) — o que reduz o
tamanho **alterando a matemática** do modelo. O CCT segue o caminho oposto
(definido em [`Compress.md`](../Compress.md)):

> `BF16 → compressão lossless → BF16 comprimido → descompressão lossless → BF16`
>
> e, mais importante: **`decompress` acontece dentro do cálculo, tile a tile,
> sem materializar a matriz inteira.**

O princípio tem nome no projeto: **CFCS — Compute From Compressed State**.

### O que o CCT é, sem eufemismo

O codec L3 é, na prática, um **field-shuffle/bitshuffle** de BF16 seguido de
zstd por fluxo — a mesma família de transformação que o **Blosc** aplica a
floats desde ~2013. Isso não é demérito: é a solução correta para o problema.
A medição da §7 mostra que o L3 **empata** com `blosc+zstd+shuffle` em CR
(1,440x vs 1,443x) e **perde** em velocidade de decode.

O diferencial do CCT, portanto, **não é o codec**: é o runtime (tiles
independentes + índice de acesso aleatório + CCT-GEMM que nunca materializa
`W` + encode-on-update). É isso que o projeto precisa provar.

## 2. A regra que não pode ser quebrada

| permitido | **proibido** |
|---|---|
| compressão lossless bit-exact | INT8, INT4, FP8, FP4 |
| tiles + acesso aleatório | pruning, esparsidade forçada |
| predição + entropia própria | qualquer redução de precisão |
| decode sob demanda | "compressão" que perde informação |

O teste que fecha a questão é sempre:

```python
torch.equal(W_original, W_reconstruido)   # True para TODO tensor
```

## 3. Estado atual (leia antes dos números)

| item | estado | evidência |
|---|---|---|
| Matemática (Porta 1) | ✅ | bit-exact em todos os casos-limite + todos os codecs |
| Codec próprio vs zstd puro (CR) | ✅ | L3: **1.50–1.51x** vs zstd 1.30–1.32x |
| Codec próprio vs **blosc+shuffle** (CR) | ✅ marginal | L3 1,503x vs blosc 1,442x (**+4,2%**) (§7) |
| **Teto de entropia** | ✅ medido | 10,54 bits/valor ⇒ CR máx. **1,518x**; L3 faz 10,64 (§5.1) |
| Eficiência de tempo (Porta 2) | ⚠️ quase | decode nativo 0,56 GB/s vs blosc 0,64 GB/s (**−12%**) |
| Decoder nativo C | ✅ | `cct/native/l3_native.c`; **23x** sobre numpy (§6.3) |
| Treino com encode-on-update (Porta 3) | ⚠️ | harness pronta; overhead medido (§8) |
| Geração autorregressiva | ❌ | 4,5–7,8x mais lento/token (§9) |
| Backends NPU/CUDA | ⛔ | sem hardware nesta máquina |

**A tese correta (a que os números sustentam):**

> Em CPU pequena, o CCT **não acelera** — ele reduz memória. Como a CPU é
> compute-bound nesse regime, o decode é trabalho extra que não se sobrepõe a
> nada. O CCT só melhora tempo quando o workload é **memory-bound** (modelo
> grande, GPU com FLOPs sobrando, edge com pouca RAM e compute suficiente).

Isso vale para qualquer codec lossless, inclusive o blosc. O que muda é *quem
paga o decode*. A §9 mede exatamente isso.

## 4. Arquitetura

```
                 ┌──────────────────────────────────────────┐
  torch.Tensor → │ CCTTensor.from_tensor(W, codec="l3_shuffle")
                 └──────────────────────────────────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    ▼                              ▼
             TileIndex (offset,            tiles comprimidos
             raw_size, codec)              (blocos independentes)
                    │                              │
                    └──────────────┬───────────────┘
                                   ▼
     cct_matmul(W, x):  compressed tile → decode → GEMM → acumula
                        (nunca materializa W inteiro)
```

| arquivo | papel |
|---|---|
| `cct/tensor.py` | `CCTTensor`: shape/dtype/tiles/índice/storage; `from_tensor`, `load_tile`, `decompress`, `save/load` |
| `cct/index.py` | `TileIndex`: acesso aleatório por tile + serialização binária |
| `cct/entropy.py` | Huffman canônico próprio (sem dependência externa) |
| `cct/codec/__init__.py` | L0 zstd, L1 entropy, L2 bit-plane + `hybrid` (best-of por tile) |
| `cct/codec/l3_structured.py` | **L3: shuffle de campos BF16 + zstd por fluxo** |
| `cct/native/l3_native.c` | **decoder C do L3** (remontagem BF16) + loader ctypes |
| `cct/ops.py` | `cct_matmul` (`autograd.Function`): decode on-the-fly, gradiente correto |
| `cct/runtime/cpu.py` | `TileCache` (LRU) + prefetch paralelo + medição de GB/s |
| `cct/training/__init__.py` | `CCTLinear`, `measure_encode_on_update`, `ecd/ecd_time/ecd_report` |
| `cct/integrate.py` | `cctify_model`, `refresh_all`, `cct_stats`, `uncctify_model` |

## 5. Os codecs (L0 → L3) e por que o L3 ganha

Todo tensor BF16 é 1 bit de sinal + 8 de expoente + 7 de mantissa.

| codec | ideia | resultado em pesos reais |
|---|---|---|
| **L0** `l0_zstd` | zstd puro no BF16 *interleaved* | 1.28x |
| **L1** `l1_entropy` | split alto/baixo + Huffman próprio | **1.44–1.50x** (após correção) |
| **L2** `l2_bitplane` | 8 bit-planes + RLE + Huffman | 1.09x (quase expande) |
| **L3** `l3_shuffle` | **de-interleave dos campos + zstd por fluxo** | 1.44–1.50x |
| `hybrid` | melhor codec **por tile** (garante CR ≥ 1) | igual ou melhor que o melhor |

### Por que L2 falhava (medido, não suposto)

1. A **mantissa é praticamente aleatória**: 6.98 bits de entropia em 7.
2. O **byte baixo** do BF16 carrega 7 bits de mantissa + 1 de expoente.
3. O **Huffman estático por tile** paga tabela por tile; em tiles pequenos o
   overhead come o ganho.

### A descoberta que mudou o jogo

Medições com **cross-entropy held-out**:

| medida | bits |
|---|---|
| H(expoente) ordem-0 | 6.18 |
| H(expoente \| vizinho de cima) | 5.54 |
| H(expoente \| vizinho da esquerda) | 6.98 (inútil) |
| teto honesto só com entropia (1 + 5.54 + 7) | 13.54 → **CR 1.18x** |

Ou seja: **1.3x é inatingível só com entropia**. O ganho vem de **reorganizar
os bytes para o compressor enxergar estrutura** (o *field shuffle*):

```
sign     : 1 bit/valor   → empacotado → zstd   (redundante, comprime muito)
expoente : 1 byte/valor  → zstd                (poucos valores, comprime muito)
mantissa : 7 bits/valor  → empacotado, cru     (~aleatória, nada a ganhar)
```

Tudo vetorizado em numpy + 2 chamadas de zstd por tile. `tile_rows=64` e
`zstd 19` são o ponto doce de CR; `tile_rows` maior reduz o custo por tile
(§9). O nível 19 (vs 12) dá **+1,5% de CR** no checkpoint e **não muda o
decode** (o nível afeta só o encode, que é offline); custa ~3x mais tempo de
encode.

### 5.1 O teto de entropia — por que o CR está acabando

Medido no checkpoint real de 1,9M (`modelos/chat_en.pt`), 1.905.792 valores:

| campo | entropia (bits/valor) |
|---|---:|
| sinal | 1,000 |
| expoente | 2,567 (cond. ao vizinho de cima: 2,554) |
| mantissa | 6,971 (de 7) |
| mantissa MSB \| expoente | 0,946 (de 1) |
| **total ordem-0** | **10,538 ⇒ CR 1,518x** |

O L3 com `zstd 19` e tile 64 entrega **10,64 bits/valor (CR 1,503x)**. Ou seja:
**0,1 bit/valor de folga (0,7%)**. Nenhum codificador sem perdas vai muito além
de ~1,52x neste checkpoint, porque 8 dos 10,5 bits são sinal+mantissa
praticamente incompressíveis.

Variantes testadas em busca de fôlego (mesmo checkpoint, tile 64):

| variante | CR |
|---|---:|
| L3 + zstd 12 (anterior) | 1,483x |
| **L3 + zstd 19 (atual)** | **1,503x** |
| expoente como delta vertical | 1,390x (pior) |
| tile transposto antes do shuffle | 1,484x (neutro) |
| mantissa também com zstd | 1,482x (neutro; overhead de frame) |
| um fluxo zstd por tensor (sem tile) | 1,442x (pior; perde adaptação local) |

**Conclusão:** o compressor está no teto. Melhorias futuras de CR são de 1–2%
nesse modelo; ganhos maiores exigem sair do lossless (INT4, §12) ou explorar
redundância entre tensores — que a medição não encontrou.

### Correção no estágio de entropia (2026-10-22)

O Huffman próprio tinha dois defeitos reais:

1. a soma de **Kraft era 0.5** — cada símbolo gastava 1 bit a mais;
2. o campo `code` tinha **2 bytes** e códigos > 16 bits (comuns no plano de
   expoente) eram truncados em silêncio → o decode **falhava** em 4 MB.

Corrigidos em `cct/entropy.py` (formato v2, campo de 4 bytes, `Kraft = 1`,
leitura do v1 preservada). Efeito medido: L1 passou de **1.25x para
1.44–1.50x** e ficou bit-exact em todos os tamanhos. O L1 agora bate o zstd,
mas o decode em Python continua inviável (§6).

## 6. Resultados medidos

Máquina: CPU (GTX 1660 Ti presente, **não usada**). Amostra principal: tensor
BF16 **2048×1024 (4.19 MB)**, pesos realistas `N(0, 0.02)`. Reproduza com
`codec_table.py`.

> **Checkpoint real (2026-10-22):** o `modelos/chat_en.pt` de 1.9M agora existe
> (treino de 10 épocas, melhor val 2.1696). Nele: **L3 = 1.488x**,
> zstd-19 = 1.311x, `blosc zstd+shuffle` = 1.442x. O L3 ganha ~3% do blosc em
> CR e continua perdendo em decode. O antigo "4.44x" vinha de um checkpoint
> pouco treinado e não se reproduz.

### 6.1 CR × T_decode × ECD

| codec | CR | encode ms/MB | decode ms/MB | decode GB/s | ECD_mem ops/B | ECD_time Mops/s |
|---|---:|---:|---:|---:|---:|---:|
| `l0_zstd` | 1.282x | 182.5 | 2.8 | 0.36 | 1.28 | 359 |
| `l1_entropy` | 1.462x | 425.7 | **1888.0** | 0.00 | 1.46 | **0.53** |
| `l2_bitplane` | 1.087x | 5920.4 | 2906.8 | 0.00 | 1.09 | 0.34 |
| **`l3_shuffle`** | **1.508x** | 240.6 | **1.8** | **0.56** | **1.51** | **534** |
| denso (referência) | 1.000x | — | 0.0 | ~80 GB/s | 0.50 | 10.700 |

O `l1_entropy` tem CR competitivo e o pior tempo: o Huffman em Python
decodifica ~0,0005 GB/s. Ele existe como **codec de entropia próprio**
(requisito do `Compress.md`), não como caminho de inferência.

> Tempos de decode medidos como **mediana de 9 execuções após aquecimento**.
> Em execuções com a máquina carregada os números caem bastante (já vimos
> 0,38 GB/s para o L3 e 0,83 GB/s para o blosc); as razões de CR são estáveis,
> os tempos absolutos não.

### 6.2 Tensores pequenos e dimensões ímpares

| shape | KB | `l3_shuffle` | `l0_zstd` |
|---|---:|---:|---:|
| (259, 192) — embedding | 97.1 | 1.501x | 1.272x |
| (192, 768) — MLP up | 288.0 | 1.438x | 1.279x |
| (768, 192) — MLP down | 288.0 | 1.502x | 1.274x |
| (1, 1152) | 2.2 | 1.445x | 1.229x |
| (3, 333) — ímpar | 2.0 | 1.429x | 1.221x |
| (17, 65) — ímpar | 2.2 | 1.445x | 1.233x |

Não há padding: `tile_rows` só agrupa linhas; o último tile é menor. Em
tensores pequenos o cabeçalho por tile pesa mais, e o CR do L3 cai de ~1.50x
para ~1.43x — ainda acima do zstd.

### 6.3 Decoder nativo (C) do L3

`cct/native/l3_native.c` faz a remontagem BF16 (sinal empacotado + expoente +
mantissa de 7 bits) com leituras de 64 bits e escrita endian-independente.
O zstd continua sendo o do `python-zstandard` (também C).

| decode L3 (4.19 MB) | tempo | GB/s |
|---|---:|---:|
| numpy (`_unpack_bits`) | 173 ms | 0.024 |
| **C nativo** | **7,5 ms** | **0.56** |
| blosc zstd+shuffle (mesmo buffer) | 6,6 ms | 0.64 |

Ganho de **~23x** sobre numpy, e o resultado é bit-exact nos dois caminhos (a
biblioteca faz autoteste contra a referência numpy ao carregar). Se não houver
compilador, `available()` é `False` e o numpy assume — nada quebra. O L3 fica
**12% atrás do blosc** em decode, não mais 2x.

```
python -c "import cct.native as n; n.build()"
```

### 6.4 CCT-GEMM

* `rel_err` vs `torch.matmul`: **3.7e-07** (não é bit-exact; §11)
* gradiente em relação a `x` propaga corretamente
* nunca materializa `W` inteiro: itera tiles, `decode → matmul → acumula`

## 7. Baselines externos — a comparação que decide

O `Projeto.md` §2.3 estava certo: comparar o L3 só com zstd puro é comparar
contra um adversário fraco. Mesmo buffer da §6 (4.19 MB BF16):

| método | CR | encode ms/MB | decode ms/MB | decode GB/s |
|---|---:|---:|---:|---:|
| zstd-19 | 1.299x | 247.3 | 3.19 | 0.31 |
| zstd-12 | 1.292x | 33.4 | 2.14 | 0.47 |
| zstd-3 | 1.282x | 1.8 | 1.57 | 0.64 |
| lz4 (frame) | 1.000x | 0.8 | 0.39 | 2.58 |
| blosc lz4+shuffle | 1.189x | 2.2 | 0.42 | 2.36 |
| blosc lz4+bitshuffle | 1.320x | 0.8 | 0.55 | 1.81 |
| **blosc zstd+shuffle** | **1.443x** | 202.9 | **1.21** | **0.83** |
| blosc zstd+bitshuffle | 1.363x | 107.6 | 0.95 | 1.06 |
| blosc zstd (sem filtro) | 1.292x | 165.7 | 2.68 | 0.37 |
| zstd-19 + dicionário (held-out) | 1.284x | 169.6 | 2.43 | 0.41 |
| **CCT-L3** | **1.440x** | 84.8 | 2.6 | 0.38 |

No **checkpoint real** de 1.9M (`baselines.py --checkpoint`, 3,81 MB):
L3 **1,503x** vs `blosc zstd+shuffle` **1,442x**, `blosc zstd+bitshuffle`
1,391x, zstd-19 1,311x. Decode (mediana, sem carga): L3 **0,56 GB/s** vs
blosc **0,64 GB/s**.

**Conclusões:**

1. O L3 **ganha do padrão de fato** em CR: 1,503x vs 1,442x (**+4,2%**).
   Também ganha do zstd puro (+15%) e do zstd com dicionário (+18%).
2. O L3 **perde pouco** em decode: 0,56 vs 0,64 GB/s (−12%). Com o decoder
   nativo, a diferença deixou de ser 2x.
3. `bitshuffle` **não** bate `shuffle` aqui (1,391x vs 1,442x): o shuffle de
   bytes já expõe o expoente, que carrega a estrutura.
4. Dicionário zstd não ajuda em pesos BF16 (1,278x ≈ zstd puro).
5. O ganho de CR está perto do teto de entropia (§5.1): não espere 1,6x+.

**Portanto:** o CCT **bate o blosc em CR e fica 12% atrás em decode**. Ele só
vira ganho de tempo quando o workload é memory-bound (o decode se sobrepõe à
banda). O diferencial estrutural continua sendo o acesso aleatório por tile e
o consumo direto no GEMM.

## 8. Fase 4 — treinar com encode-on-update

`cct/training/` implementa o loop do §15:

```
compressed W → decode → forward → backward → update → recompress → compressed W
```

Medido com `CCTLinear(64→128)` e L3:

```
CR = 1.52–1.57x
encode = 1.4–1.5 ms
step denso            : 0.54–0.69 ms
step com recompressão : 1.77–2.51 ms   (overhead +230% a +260%)
```

**Honestidade:** a referência LCT reporta +7,2% de tempo; os nossos +230–260%
vêm do encoder em Python. E durante o treino o peso precisa existir
descomprimido para o AdamW — a economia de memória real continua em aberto.
O que está entregue é a **harness de medição**.

### ECD corrigido

O README antigo dizia "ECD: 4.0 → 6.1" enquanto o tempo triplicava. Isso só
fecha se ECD ignorar tempo — e aí `ECD = ops/bytes` é só `1/CR` disfarçado
(6.1/4.0 = 1.525 ≈ CR 1.568). Hoje há duas métricas explícitas:

```python
ecd(flops, bytes_movidos)                  # ECD_mem  — ops/byte (≈ 1/CR)
ecd_time(flops, segundos)                  # ECD_time — ops/segundo
ecd_report(flops, raw_b, comp_b, t_dec, t_comp)  # ambos + overhead
```

Na amostra da §6: `ECD_mem` do L3 = **1.44 ops/B** (vs 0.50 do denso), mas
`ECD_time` = **379 Mops/s** contra **76.678 Mops/s** do denso. A segunda é a
que decide — e ela **cai** com o decode.

## 9. Geração autorregressiva — o caso de uso real

`cct/benchmarks/generation.py`: `TinyGPT` 1.87M, batch=1, KV cache. Na
geração token a token, **cada peso é lido a cada token** — o pior caso, e o
caso real de um SLM.

| variante | ms/token | tokens/s | vs denso |
|---|---:|---:|---:|
| denso fp32 | 3.07 | 326 | 1.00x |
| denso (pesos bf16, contas fp32) | 2.71 | 369 | 0.88x |
| **CCT-L3, tile_rows=64** | **23.89** | 42 | **7.79x mais lento** |
| **CCT-L3, tile_rows=192** | **16.30** | 61 | **4.45x mais lento** |

Detalhes que importam:

* Com `tile=64`, o decode puro é **39%** do tempo CCT; o resto é o laço
  Python + GEMM por tile. Com `tile=192` o decode sobe para **44%** e o total
  cai ~32% — porque há menos tiles (menos chamadas). O CR fica praticamente
  igual (1,503x → 1,507x). **Trade-off tile × tempo** que o README não
  documentava.
* Os tokens gerados são **idênticos** ao modelo com pesos arredondados para
  BF16 (é exatamente o que o CCT representa): a perda é zero.
* Os valores absolutos variam ~20–30% entre execuções nesta máquina; as razões
  são medidas dentro do mesmo processo.

**Leitura honesta:** em CPU pequena o CCT perde em tempo por um fator de 5–9x.
A memória cai ~31%. Para o workload real (geração batch=1), **não há ganho de
tempo**.

## 10. Como integrar num modelo

```python
import torch, torch.nn as nn
from cct.integrate import cctify_model, refresh_all, cct_stats

model = ...                      # seu TinyGPT/Transformer normal
info = cctify_model(model, codec="l3_shuffle", tile_rows=192)
print(info["layers"], info["CR"], info["raw_bytes"], info["compressed_bytes"])

model.eval()
y = model(x)                     # consome o estado comprimido, tile a tile

# se treinar com encode-on-update:
optimizer.step(); refresh_all(model)
```

Substituição de uma camada isolada:

```python
from cct.training import CCTLinear
# antes
self.fc = nn.Linear(192, 768)
# depois (inferência)
self.fc = CCTLinear.from_dense(nn.Linear(192, 768), codec="l3_shuffle")
```

`cctify_model` ignora embeddings e `head` por padrão (lookup/tied weight).
Em `model.train()` o `CCTLinear` usa o peso denso, porque o AdamW precisa do
gradiente e do estado do otimizador.

## 11. Bit-exact tem dois sentidos

| etapa | bit-exact? |
|---|---|
| `decompress(compress(W)) == W` | **sim**, provado para todos os tensores |
| `cct_matmul(W, x)` vs `torch.matmul(W, x)` | **não**: `rel_err ≈ 3.7e-07` |

A segunda diferença é esperada: a ordem de acumulação tile-a-tile difere da
ordem do `torch.matmul` em fp32. O CCT **não** promete bit-exact end-to-end no
GEMM; promete não perder nenhum bit do **peso**.

## 12. Por que não INT4

A regra da §2 (proibido quantizar) é uma decisão de escopo, não um argumento
de mercado. Os números:

| formato | 1.9M params | 16M params |
|---|---:|---:|
| BF16 | 3.8 MB | 32 MB |
| CCT-L3 (1.45x) | 2.6 MB | 22 MB |
| INT8 | 1.9 MB | 16 MB |
| INT4 | 0.95 MB | 8 MB |

INT4 ganha do CCT em memória **e** em velocidade isoladamente. O
posicionamento correto não é "CCT em vez de INT4", é:

> **CCT é ortogonal e cumulativo.** `CCT(INT4(BF16(W)))` pode chegar a 3–4x de
> compressão efetiva sem adicionar erro de quantização além do que o INT4 já
> adiciona. O CCT comprime a *representação*; a quantização muda a
> *matemática*. As duas coisas se somam.

O escopo atual isola a variável "compressão pura" para medir a hipótese. A
combinação com INT4 é uma linha de pesquisa separada (e a mais promissora).

## 13. Conformidade com o `Compress.md`

| Secção | Requisito | Estado | Evidência |
|---|---|---|---|
| §1-2 | Lossless bit-exact, sem quantização | ✅ | `torch.equal` em todos os tensores |
| §3 | `CCTTensor` (shape/dtype/tiles/codec/índice/storage) | ✅ | `cct/tensor.py` |
| §4 | Tiles independentes + índice serializado | ✅ | `TileIndex` + roundtrip |
| §5 | Compressor próprio (campos → entropia) | ✅ | L1 Huffman + L3 |
| §6 | Hierarquia L0→L3 com métrica própria | ✅ | 5 codecs + hybrid |
| §7 | Otimizar `T_effective = max(T_mem, T_decode, T_compute)` | ⚠️ | medido; decode domina |
| §8 | Runtime (compress/decompress/load_tile/prefetch/cache) | ✅ | `runtime/cpu.py` |
| §9 | **CCT-GEMM** sem materializar W | ✅ | `ops.py`, rel_err 3.7e-07 |
| §10-11 | CPU; modelo pequeno | ✅ | tudo em CPU |
| §13 | Bit-exact por tipo/tensor | ✅ | bf16, 1-D, ímpar, zeros, denormais |
| §14 | Mapa de compressibilidade | ✅ | `benchmarks/compression.py` |
| §15 | Encode-on-update no treino | ⚠️ | harness + medição (§8) |
| §16-17 | Decoder paralelo + cache | ✅ / ⚠️ | **decoder C nativo** + cache; ainda perde do blosc |
| §18-22 | GPU / fused / Tensor Core / pipeline | ⛔ | sem GPU/AMX |
| §23 | Benchmark A–F (16M) | ⏳ | falta o modelo 16M |
| §25 | Métrica **ECD** | ✅ | `ecd`/`ecd_time`/`ecd_report` (corrigida) |
| §29 | Porta 1 ✅ · Porta 2 (CR ✅, tempo ❌) · Porta 3 ⚠️ | — | §3, §6, §8 |

## 14. Testes e benchmarks

Dependências opcionais dos baselines: `pip install zstandard blosc2 lz4`.
O decoder nativo precisa de `gcc`, `clang` ou `cc` no PATH.

```powershell
# testes
python cct\tests\run_all.py                       # tudo
python cct\tests\test_bit_exact.py                # codecs + CCTTensor + GEMM
python cct\tests\test_l3_fase4.py                 # L3 + Fase 4 + ECD
python cct\tests\test_entropy.py                  # Huffman (Kraft, códigos longos)
python cct\tests\test_native_l3.py                # decoder C vs numpy
python cct\tests\test_integrate.py                # from_dense + cctify_model
python -m pytest cct\tests -q                     # se preferir pytest

# decoder nativo
python -c "import cct.native as n; n.build()"

# benchmarks
python cct\benchmarks\baselines.py                # blosc/zstd/lz4/...
python cct\benchmarks\codec_table.py              # CR x tempo x ECD
python cct\benchmarks\generation.py --layers 4    # tokens/s denso vs CCT
python cct\benchmarks\compression.py              # mapa por tensor/grupo
python cct\benchmarks\l3_check.py modelos\chat_en.pt
```

O que cada suíte cobre:

* **`test_bit_exact.py`** — Huffman exato; os 5 codecs em 4 padrões;
  `CCTTensor` por codec; CCT-GEMM (rel_err + gradiente); cache/prefetch.
* **`test_l3_fase4.py`** — L3 bit-exact em bf16 2-D/1-D/ímpar/zeros/constante/
  denormais; **L3 > zstd**; `hybrid ≥ max`; Fase 4; ECD_mem cresce e ECD_time
  cai.
* **`test_entropy.py`** — `Kraft = 1`, 256 símbolos, distribuição enviesada
  com códigos longos, plano de expoente BF16 em 4 MB (regressão), payload v1.
* **`test_native_l3.py`** — decoder C vs numpy, tamanhos 1..N e ímpares;
  skip limpo se a biblioteca não estiver compilada.
* **`test_integrate.py`** — `from_dense`, input 3-D, `cctify_model`,
  `uncctify_model`.

## 15. Limitações honestas

1. **O CCT perde em tempo em CPU pequena.** 4,5–7,8x por token (§9). Ele
   reduz memória ~31%; só ganharia tempo num regime memory-bound.
2. **O CR está no teto de entropia** (§5.1): 10,64 bits/valor medidos contra
   10,54 de limite ordem-0. Novas ideias de codec rendem 1–2%, não 20%.
3. **O decoder nativo resolveu o gargalo do numpy, não o do formato.** De
   0,024 para 0,56 GB/s; o blosc faz 0,64 GB/s (12% de diferença).
4. **A meta de 1.3x não vem de "entropia melhor"** — vem do shuffle (§5).
5. **Fase 4 é +230–260% de tempo** com encoder Python (LCT: +7,2%).
6. **Sem GPU/AMX nesta máquina**: bf16 em CPU é ~140x mais lento que fp32
   (6,96 ms vs 0,05 ms no matvec de referência).
7. **`format.py` ainda não existe**: a serialização vive em `index.py` +
   `tensor.py`; falta versão/checksum no cabeçalho.
8. **O encoder continua em numpy** (~85 ms/MB). Otimizar encode não muda a
   inferência, mas pesa no encode-on-update.

## 16. Roadmap P0–P10

| fase | descrição | estado |
|---|---|---|
| P0 | Dataset + modelo 16M | ⏳ (existe 1.9M/6.1M, não 16M) |
| P1 | Baseline BF16 em CPU | ✅ |
| P2 | CCT-L0 zstd | ✅ |
| P3 | CCT-L1 entropy | ✅ (CR ok, decode inviável) |
| P4 | CCT-L2 bit-plane | ✅ (não competitivo) |
| **P4b** | **CCT-L3 field shuffle** | ✅ **1.44–1.50x** |
| P5 | Treino dinâmico (encode-on-update) | ⚠️ harness pronta |
| **P6** | **CPU otimizada (decoder nativo)** | ✅ **C**; ainda 2,2x atrás do blosc |
| P7 | Backend NPU | ⛔ sem hardware |
| P8–P10 | CUDA / fused decode+GEMM / Tensor Core | ⛔ sem GPU |

**Próximo passo de maior valor:** o codec já empata com o blosc e o decoder
nativo já existe; o que falta é **provar a tese CFCS num regime memory-bound**
— modelo 16M (P0), medir `T_mem` vs `T_decode` com banda de memória saturada,
e só então justificar CUDA/NPU. Antes disso, otimizar mais o codec é otimizar
o que não decide.

---

<div align="center">

Documentação relacionada: [`Compress.md`](../Compress.md) (especificação) ·
[`README.md`](../README.md) (projeto) · [`marco.md`](../marco.md) (histórico)

</div>
