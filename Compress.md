Eu estruturaria isso como um projeto de arquitetura experimental, com uma distinção rigorosa entre compressão lossless real e quantização. A meta é criar uma nova abstração de tensor: o peso continua sendo BF16/FP32 exatamente como é, mas sua representação residente e seu transporte passam a ser comprimidos.

Há evidência recente de que essa direção é tecnicamente viável: o LCT, publicado em outubro de 2026, explora tensores lossless comprimidos durante treinamento; e o NWC demonstra pesos BF16 lossless comprimidos, decodificados dentro do kernel CUDA sem materializar o tensor descomprimido na memória. arXiv
Projeto: CCT — Compressed Compute Tensor
Eu daria ao projeto o nome:
CCT — Compressed Compute Tensor

E ao princípio arquitetural:
Compute From Compressed State — CFCS

A ideia é:
\[
\boxed{
\text{memória armazena compressão}
\rightarrow
\text{computação consome compressão}
}
\]
e não:
\[
\text{memória armazena compressão}
\rightarrow
\text{descomprime tensor inteiro}
\rightarrow
\text{computação}
\]
Essa diferença é o coração do projeto.
1. O problema que queremos resolver
Nosso experimento-alvo será um modelo de aproximadamente 16 milhões de parâmetros, treinado em um dataset de linguagem cotidiana.

Vamos comparar:
A — baseline
16M parâmetros
BF16 convencional

B — CCT
16M parâmetros
BF16 matematicamente idênticos
representação comprimida
decode on-the-fly

A meta não é apenas:
reduzir memória.

Queremos:
reduzir o número de bytes que atravessam a hierarquia de memória sem reduzir a precisão matemática dos parâmetros.

Portanto:
W_original == W_reconstructed

bit a bit.

Se isso não acontecer, o teste falhou.
2. O que NÃO vamos fazer
Isso precisa ficar congelado como requisito do projeto.

Não vamos inicialmente usar:
- INT8;
- INT4;
- FP8;
- FP4;
- pruning;
- sparsity forçada;
- redução de precisão;
- pesos aproximados.
Essas técnicas podem entrar posteriormente como linhas de pesquisa separadas.

Nosso primeiro CCT será:
BF16
   ↓
LOSSLESS COMPRESSION
   ↓
compressed BF16
   ↓
LOSSLESS DECOMPRESSION
   ↓
BF16 original

Portanto:
\[
D(C(W))=W
\]
3. A primeira inovação: CompressedTensor
Em vez de um tensor normal:
weight = torch.Tensor(...)


vamos criar conceitualmente:
weight = CCTTensor(...)


Ele terá algo parecido com:
CCTTensor
│
├── shape
├── dtype
├── tile_size
├── codec
├── index
├── metadata
└── compressed_storage

O tensor não precisa estar materializado em BF16 inteiro.

Por exemplo:
16M BF16
≈ 32 MB

podem residir como algo conceitualmente parecido com:
compressed storage
≈ 20–25 MB

dependendo da distribuição real dos dados.

O objetivo é descobrir empiricamente quanto conseguimos.
4. Dividir o tensor em tiles
Essa é uma decisão arquitetural fundamental.

Não comprimiremos uma matriz inteira como um único arquivo.

Por exemplo:
W
┌────────┬────────┬────────┬────────┐
│ Tile 0 │ Tile 1 │ Tile 2 │ Tile 3 │
├────────┼────────┼────────┼────────┤
│ Tile 4 │ Tile 5 │ Tile 6 │ Tile 7 │
├────────┼────────┼────────┼────────┤
│ Tile 8 │ Tile 9 │ Tile10 │ Tile11 │
└────────┴────────┴────────┴────────┘

Cada tile é:
compress(tile)

independentemente.

E teremos:
CCT Index
────────────────────
tile_id
offset
compressed_size
raw_size
codec
metadata

Isso resolve o problema de random access.

O Tensor Core não precisa descomprimir 32 MB para utilizar 128 KB.

Ele solicita:
Tile 17

e somente aquele bloco é recuperado.
5. O compressor
Aqui está uma das partes onde eu faria pesquisa própria.

Não começaria com ZSTD.

ZSTD será um baseline, não nosso algoritmo principal.

O pipeline inicial será:
BF16
 ↓
bit-plane separation
 ↓
prediction / delta
 ↓
run-length encoding
 ↓
entropy coding

Por quê?

Porque BF16 tem estrutura interna.

Um BF16 possui:
1 bit  sign
8 bits exponent
7 bits mantissa

Em pesos de redes neurais, essas partes têm distribuições diferentes.

Trabalhos recentes mostram que essa estrutura pode ser explorada para compressão lossless; o NWC, por exemplo, separa byte de mantissa e byte de expoente/sinal e codifica o segundo por entropia. GitHub
6. Nosso primeiro codec: CCT-L0
Eu criaria uma sequência de codecs.
CCT-L0
Baseline:
BF16
 ↓
ZSTD

Serve apenas para termos referência.
CCT-L1
BF16
 ↓
separate exponent/sign
 ↓
mantissa raw
 ↓
Huffman

CCT-L2
BF16
 ↓
bit planes
 ↓
delta
 ↓
RLE
 ↓
Huffman

CCT-L3
BF16
 ↓
adaptive predictor
 ↓
bit-plane transform
 ↓
adaptive entropy coding

E o objetivo é:
\[
CCT-L3 > CCT-L2 > CCT-L1 > CCT-L0
\]
em uma métrica que combine compressão + velocidade de decode.
7. Não queremos o maior compression ratio
Esse é um erro que eu evitaria desde o começo.

Imagine:
Codec A
3.0× compression
2 GB/s decode

Codec B
1.8× compression
800 GB/s decode

Para nossa arquitetura, B pode ser muito melhor.

Nossa função objetivo deve ser:
\[
Score =
f(
compression\ ratio,
decode\ throughput,
latency,
random\ access
)
\]
Ou, mais concretamente:
\[
T_{effective}
=
\max(
T_{memory},
T_{decode},
T_{compute}
)
\]
Queremos minimizar isso.
8. O segundo componente: CCT Runtime
Precisamos de um runtime:
CCT Runtime

responsável por:
compress
decompress
load_tile
prefetch_tile
cache_tile

A API poderia ser:
cct = CCTTensor.from_tensor(weight)tile = cct.load_tile(17)assert torch.equal(    tile,    weight_tile)


Depois:
y = cct.matmul(x)


O usuário não deveria precisar chamar decompress().
9. A operação mais importante: CCT-GEMM
Aqui começa a parte realmente nova.

Hoje:
W
 ↓
GEMM

CCT:
W_compressed
 ↓
CCT-GEMM

Conceitualmente:
\[
Y = CCT\_GEMM(W_c,X)
\]
internamente:
compressed tile
       ↓
decode
       ↓
registers
       ↓
Tensor Core / ALU
       ↓
accumulate

O que não deve acontecer:
compressed
 ↓
decompress
 ↓
full BF16 matrix
 ↓
GEMM

Esse seria o primeiro grande anti-pattern do projeto.
10. Mas vamos começar em CPU
Aqui sua observação é boa.

Se a arquitetura funcionar na CPU, conseguimos separar duas coisas:
Problema matemático
A compressão realmente funciona?
Problema de hardware
O decoder consegue acompanhar o hardware?

Primeiro resolveremos o primeiro.
11. Fase 0 — dataset
Usaria um dataset pequeno de linguagem cotidiana.

Por exemplo:
conversas
perguntas
respostas
instruções
descrições
pequenos textos

Não precisamos inicialmente de bilhões de tokens.

Algo como:
10M–100M tokens

é suficiente para testar a arquitetura.

O modelo poderia ser um Transformer pequeno:
~16M parâmetros

por exemplo:
vocab        = 16k
layers       = 6
hidden       = 384
heads        = 6
FFN          ≈ 4 × hidden
context      = 512/1024

Os números exatos seriam ajustados para fechar próximo dos 16M.
12. Fase 1 — referência matemática
Primeiro:
PyTorch
CPU
BF16/FP32

Treinamos o modelo convencional.

Registramos:
loss
validation loss
tokens/s
step time
memory

Isso vira:
BASELINE-0
Nada será comparado sem essa referência.
13. Fase 2 — compressão offline
Agora pegamos cada tensor:
compressed = cct.compress(weight)


e verificamos:
restored = cct.decompress(compressed)torch.equal(weight, restored)


Tem que ser:
True

para todos os tensores.

Testaremos:
embedding
attention Q
attention K
attention V
output projection
MLP
LayerNorm

Separadamente.
14. Fase 3 — descobrir onde existe redundância
Esse é um experimento científico importante.

Não assumiremos que todos os tensores são igualmente compressíveis.

Mediremos:
Tensor                  Ratio

Embedding               ?
Q projection            ?
K projection            ?
V projection            ?
O projection            ?
MLP up                  ?
MLP gate                ?
MLP down                ?
LayerNorm               ?

E:
layer 0
layer 1
...
layer 5

Isso nos mostrará onde está a informação redundante.
15. Fase 4 — compressão dinâmica durante treinamento
Agora entra a parte que diferencia armazenamento de arquitetura.

O loop vira:
                compressed W
                     │
                     ▼
                  decode
                     │
                     ▼
                  forward
                     │
                     ▼
                 backward
                     │
                     ▼
                  update
                     │
                     ▼
                 compress
                     │
                     └─────────► compressed W

Mediremos:
encode time
decode time
training time
memory
compression ratio

A pergunta:
quanto custa manter o tensor comprimido durante o treinamento?

O trabalho LCT mostra que esse custo é real: em seu experimento de 162M parâmetros, houve redução de memória de 24,6%, mas aumento de 7,2% no tempo por step. arXiv

Portanto nossa meta deve ser superar esse trade-off no caso específico de 16M.
16. Fase 5 — CPU otimizada
Agora vamos paralelizar:
CPU cores
│
├── decoder thread
├── decoder thread
├── decoder thread
└── decoder thread

E construir:
compressed tile
        ↓
parallel decode
        ↓
L1/L2/L3 cache
        ↓
matrix multiplication

Aqui vamos descobrir algo fundamental:
\[
Decode\ Throughput
\]
precisa ser comparado com:
\[
Memory\ Bandwidth
\]
Se:
decode = 100 GB/s
memory = 80 GB/s

estamos bem.

Se:
decode = 20 GB/s
memory = 100 GB/s

a compressão pode piorar o desempenho.
17. Fase 6 — NPU
Aqui eu não faria inicialmente uma integração proprietária profunda.

A NPU será usada como validação de conceito de execução heterogênea.

Arquitetura:
CPU
 │
 │ compressed tensors
 ▼
NPU
 │
 ├── decode
 ├── compute
 └── output

O problema é que NPUs variam enormemente por fabricante.

Por isso criaríamos uma abstração:
backend="cpu"backend="npu"backend="cuda"


e uma interface:
cct.matmul(    compressed_weight,    activation,    backend=...)


No NPU, inicialmente podemos aceitar:
decode → local buffer → NPU

Isso não é a arquitetura final.

É o teste de portabilidade.
18. Fase 7 — GPU
Agora entramos no objetivo principal.

Primeiro:
CPU
 ↓
compressed tensor
 ↓
PCIe
 ↓
GPU
 ↓
decode kernel
 ↓
GEMM

Depois eliminamos o buffer intermediário.
19. GPU versão 1
HBM
 ↓
compressed tile
 ↓
CUDA decoder
 ↓
global/shared memory
 ↓
BF16 GEMM

Isso testa:
A compressão reduz tráfego suficiente para compensar o decode?

20. GPU versão 2 — fused decode + GEMM
Agora:
HBM
 │
 │ compressed
 ▼
CUDA warp
 │
 │ decode
 ▼
register fragment
 │
 ▼
Tensor Core

Essa é a versão realmente importante.

O NWC já demonstrou que decode dentro do kernel de matvec pode eliminar a passagem intermediária pela memória e, em determinados modelos/hardware, superar BF16 convencional. GitHub

Nosso avanço experimental seria levar o conceito para:
treinamento + tiles + backward + optimizer + encode-on-update + Tensor Core.

21. GPU versão 3 — pipeline
Queremos:
Tile 0:
decode → compute

Tile 1:
          decode → compute

Tile 2:
                    decode → compute

Tile 3:
                              decode → compute

Enquanto isso:
HBM load
   ║
   ╠══ decode
   ║
   ╠══ Tensor Core
   ║
   ╚══ next tile prefetch

Ou seja:

compressão, memória e computação passam a ser um pipeline.
22. Blackwell seria uma etapa posterior
Há uma oportunidade interessante aqui.

O Blackwell possui um Decompression Engine dedicado, atualmente suportando LZ4, Snappy, Deflate/Gzip, com throughput anunciado de até 600 GB/s. Ele permite sobrepor descompressão e computação. NVIDIA Docs

Mas eu não faria nosso CCT depender desse hardware.

Primeiro:
CCT decoder
→ CUDA cores / warps

Depois poderíamos testar:
CCT
+
hardware decompression

para codecs compatíveis.

A nossa arquitetura deveria continuar funcionando sem ele.
23. O benchmark definitivo
Teremos seis configurações:
Sistema	Modelo	Representação
A	16M	BF16
B	16M	CCT-L0
C	16M	CCT-L1
D	16M	CCT-L2
E	16M	CCT-L3
F	16M	CCT-L3 + fused GEMM


E mediremos:
Memória
\[
M
\]
Compressão
\[
CR=\frac{size(raw)}{size(compressed)}
\]
Throughput
\[
tokens/s
\]
Latência
\[
ms/step
\]
Decoder
\[
GB/s_{decoded}
\]
Memória efetivamente transferida
\[
GB/step
\]
Energia
\[
J/token
\]
Qualidade
\[
loss
\]
24. O teste que realmente provará a hipótese
Queremos chegar a algo parecido com:
                    memória       tempo

16M BF16             32 MB        1.00×

16M CCT              20 MB        1.05×

16M CCT fused        20 MB        0.90×

E depois:
16M CCT
≈
6M BF16

em tempo.

Se conseguirmos isso, a hipótese está demonstrada.

Mas existe uma possibilidade ainda melhor:
16M CCT
<
6M BF16

Nesse caso conseguimos mais capacidade computacional com menor tempo.
25. A métrica mais importante não será compression ratio
Eu criaria uma métrica específica:
Effective Compute Density — ECD
\[
ECD=
\frac{\text{operações úteis}}
{\text{bytes movimentados}}
\]
O objetivo da CCT é aumentar:
\[
ECD
\]
sem reduzir:
\[
\text{mathematical fidelity}
\]
Ou seja:
               mesma matemática
                     │
                     ▼
             menos bytes
                     │
                     ▼
              mais compute
              por byte

Essa é a tese científica do projeto.
26. Roadmap prático
Eu faria em 10 etapas.
P0  Dataset + modelo 16M
 │
 ▼
P1  BF16 baseline CPU
 │
 ▼
P2  CCT-L0 ZSTD
 │
 ▼
P3  CCT-L1 BF16 entropy codec
 │
 ▼
P4  CCT-L2 bit-plane codec
 │
 ▼
P5  dynamic training CPU
 │
 ▼
P6  optimized CPU
 │
 ▼
P7  NPU backend
 │
 ▼
P8  CUDA decoder
 │
 ▼
P9  fused decode + GEMM
 │
 ▼
P10 training Tensor Core + encode-on-update

27. Stack tecnológico
Eu usaria:
Linguagem
Python

para:
- experimentos;
- treinamento;
- datasets;
- benchmark;
- controle;
- análise.
Framework
PyTorch
CPU
- C++/OpenMP posteriormente;
- inicialmente Python/NumPy/PyTorch;
- eventualmente SIMD.
NPU
Backend específico conforme o hardware disponível.
GPU
- CUDA;
- C++;
- CUDA kernels;
- Tensor Cores;
- torch.library / custom operators;
- eventualmente torch.compile.
Benchmark
PyTorch profiler
Nsight Systems
Nsight Compute
perf
memory bandwidth counters

28. Estrutura do projeto
Eu organizaria o repositório assim:
cct/
│
├── cct/
│   ├── tensor.py
│   ├── format.py
│   ├── index.py
│   ├── codec/
│   │   ├── l0_zstd.py
│   │   ├── l1_entropy.py
│   │   ├── l2_bitplane.py
│   │   └── l3_adaptive.py
│   │
│   ├── runtime/
│   │   ├── cpu.py
│   │   ├── npu.py
│   │   └── cuda.py
│   │
│   ├── kernels/
│   │   └── cct_gemm.cu
│   │
│   └── training/
│       ├── optimizer.py
│       └── trainer.py
│
├── models/
│   └── model_16m.py
│
├── datasets/
│   └── everyday.py
│
├── benchmarks/
│   ├── compression.py
│   ├── cpu.py
│   ├── npu.py
│   └── gpu.py
│
├── tests/
│   ├── bit_exact.py
│   ├── codec.py
│   ├── training.py
│   └── kernels.py
│
└── experiments/
    ├── 001_baseline
    ├── 002_l0
    ├── 003_l1
    ├── 004_l2
    └── ...

29. Critério de "funcionou"
Eu colocaria três portas de entrada.
Porta 1 — Matemática
Obrigatório:
\[
decode(encode(W))=W
\]
bit-exact.
Porta 2 — Eficiência
Queremos:
\[
CR>1
\]
e:
\[
T_{decode}<T_{memory\ saved}
\]
Porta 3 — Treinamento
Queremos:
loss(CCT) ≈ loss(BF16)

e, idealmente:
time(CCT) ≤ time(BF16)

30. O conceito final
O projeto não é:
"um compressor para modelos".

É:
uma nova representação executável de tensor na qual o estado matemático do modelo permanece lossless, mas o estado físico residente na memória é comprimido, e unidades de computação consomem esse estado diretamente por meio de decodificação sob demanda.

A arquitetura pode ser resumida como:
                 CCT ARCHITECTURE

       ┌───────────────────────────────┐
       │          MODEL STATE          │
       │                               │
       │      mathematically BF16      │
       └──────────────┬────────────────┘
                      │
                encode/update
                      │
                      ▼
       ┌───────────────────────────────┐
       │      COMPRESSED STATE         │
       │                               │
       │ tiles + index + entropy code  │
       └──────────────┬────────────────┘
                      │
                      │ HBM / RAM
                      │
                      ▼
       ┌───────────────────────────────┐
       │       ON-DEMAND DECODER       │
       │                               │
       │ tile → registers              │
       └──────────────┬────────────────┘
                      │
                      ▼
       ┌───────────────────────────────┐
       │        COMPUTE ENGINE         │
       │                               │
       │ CPU / NPU / Tensor Core       │
       └──────────────┬────────────────┘
                      │
                      ▼
                   RESULT

A sequência correta de desenvolvimento é, portanto:
CPU → codec lossless → treinamento comprimido → CPU otimizada → NPU → CUDA → fused decode/GEMM → Tensor Core → encode-on-update.

Isso também nos dá uma vantagem metodológica: não precisamos começar com uma GPU cara para saber se a hipótese fundamental funciona. Se não conseguirmos demonstrar, no CPU, que o nosso formato consegue comprimir os tensores de treinamento, reconstruí-los bit-exactamente e fazer o custo do decode ficar suficientemente baixo, não vale a pena escrever o kernel Tensor Core.

Por outro lado, se conseguirmos chegar a um decoder CPU altamente paralelo cuja vazão seja próxima ou superior à largura de banda efetiva da memória, temos uma base muito forte para migrar o mesmo formato para SIMD, NPU e finalmente CUDA.

E há um benchmark de referência muito útil para nosso projeto: o NWC já demonstra 31% de redução do tamanho de BF16 e decode dentro do kernel CUDA, enquanto o LCT demonstra o caminho de compressão lossless durante treinamento. Nossa proposta pode posicionar-se exatamente entre esses dois trabalhos: um CCT orientado desde o início a treinamento, random-access por tiles, encode-on-update e execução direta em CPU → NPU → Tensor Core, em vez de tratar compressão apenas como checkpoint ou inferência.