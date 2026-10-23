
Veredicto curto: este README é muito mais sério e honesto do que o anterior sugeria. Tem ciência real embutida, medições reais e — o mais raro — documenta o que não funcionou. Mas há um problema estrutural de posicionamento que o documento ainda não enfrenta, e ele decide se o CCT é paper ou produto.

Vou por partes, com números.

1. O que está genuinamente bem
1.1 A descoberta do L3 é real e está bem contada
A parte §5 ("Por que o L1/L2 falhavam") é o melhor trecho do documento. Vocês mediram que:

mantissa tem 6.98 bits de entropia em 7 → essencialmente aleatória;

byte baixo do BF16 mistura expoente com mantissa → planos híbridos;

Huffman estático por tile paga ~28% de overhead num tile de 16 linhas.

Isso é o tipo de medição que separa "achismo de compressor" de engenharia. E a conclusão — CR de 1.18x é o teto honesto se só entropia for usada — mostra que vocês entenderam por que o shuffle ganha. Não é sorte, é reorganização de bytes.

O fato de o L3 ser efetivamente um bitshuffle (o que o Blosc faz em floats desde ~2013) não é demérito. É a solução correta para o problema. Mas precisa ser dito com todas as letras no README, porque um revisor vai perguntar "isso não é só Blosc?" e vocês precisam ter a resposta pronta: "sim, o eixo é o shuffle; o diferencial do CCT está no runtime de decode fundido (CFCS), não no codec em si".

1.2 A separação 4.44x (checkpoint) vs 1.45x (realista) é honesta
Isso é raro e merece elogio explícito. Vocês poderiam esconder o 1.45x e vender 4.44x como headline. Não fizeram. Documentaram que:

"Modelos pouco treinados concentram muitos pesos perto de zero, então a compressibilidade é maior."

Isso é a explicação correta e é a diferença entre um projeto que sobrevive à auditoria e um que não.

1.3 CCT-GEMM com rel_err = 3.7e-07 e gradiente correto
Isso é o começo da Leitura B ("decode fundido a ops") que eu tinha levantado. O autograd.Function com backward funcional é a parte não-trivial, e vocês já a resolveram. O caminho está aberto.

1.4 A métrica ECD é a métrica certa
Effective Compute Density = ops úteis / byte movido é a métrica que importa quando o gargalo é memória. Não é FLOPs, não é CR isolado, não é latência. É a combinação. Está bem escolhida.

1.5 O conformance table (§8)
Rastrear 25 seções da especificação original contra o estado real é o que separa "documento de projeto" de "relatório de engenharia". A tabela deixa claro o que é ✅, ⚠️ e ⛔. Isso economiza horas de revisor.

1.6 A admissão do +231% (§7)
Comparar seu overhead com o LCT (+7.2%) e admitir que o Python é a causa aumenta a credibilidade. Muitos projetos esconderiam.

2. O problema estrutural: os números do ECD não se sustentam sozinhos
Aqui está o ponto que eu quero que vocês olhem com muito cuidado.

2.1 A conta do T_decode vs T_compute
Vocês medem T_decode ≈ 0.12 GB/s (Python). Para o modelo de 1.9M params = 3.8 MB em BF16:

operação	tempo estimado em CPU
decode Python (0.12 GB/s)	~31 ms
memcpy (20 GB/s)	0.19 ms
BF16 matmul 1.9M (10 GFLOPS)	~0.38 ms
O decode Python é ~80x o tempo de computação. Mesmo um decoder C 50x mais rápido (6 GB/s, plausível para zstd nível baixo) ficaria em 0.63 ms — ainda 1.6x mais lento que a computação. E o decode+zstd de verdade costuma ficar entre 1–5 GB/s, não em memcpy.

Isso leva a uma conclusão dura:

Em CPU, com modelos pequenos (≤ ~100M params), CCT nunca vai ganhar em velocidade. Só em memória.

O motivo é que, nesse regime, a CPU é compute-bound para BF16, não memory-bound. O decode é trabalho extra que não se sobrepõe a nada — só adiciona.

Isso não invalida o CCT, mas muda a narrativa completamente. A tese correta é:

"CCT só melhora tempo quando o workload é memory-bound: modelos grandes, GPU com FLOPs sobrando, ou edge com pouca RAM e compute suficiente."

O README não diz isso em lugar nenhum, e §3 diz "decode Python é o gargalo (ver §10)" como se, resolvido o Python, o tempo viesse de graça. Não vem. Vem só quando o regime é memory-bound. Precisa ser dito.

2.2 O ECD "4.0 → 6.1" está confundindo duas coisas
Do §7:

step denso: 0.54 ms, step com recompressão: 1.77 ms (overhead +231%), ECD: 4.0 → 6.1

Se ECD é ops / byte movido, e o tempo triplicou, como o ECD melhorou? Só faz sentido se:

ECD ignora tempo, ou

ECD está contando bytes de estado comprimido, não tempo de decode.

Mas então ECD não é "compute density" — é "compression ratio disfarçado de métrica de compute". 6.1/4.0 = 1.525, próximo de CR = 1.568. Isso é circular.

Fix: ou ECD = ops / (bytes_movidos × tempo_de_decode_por_byte), ou ECD deveria cair com o overhead medido. Do jeito que está, a métrica reforça um ganho que o tempo desmente. Isso é o ponto mais frágil do documento.

2.3 O que §6 não mostra
O README compara L3 com zstd, mas as baselines que um revisor vai pedir são:

baseline	por quê
blosc + bitshuffle + lz4/zstd	é o padrão de fato para floats em HDF5/Zarr/scientific Python
zstd com dicionário treinado no tensor	zstd puro é baseline fraco; com dict é competitivo
fpzip	codec específico para float, do LLNL
lz4 puro	baseline de velocidade (o decode do L3 com zstd 12 é ~4x mais lento que lz4)
Se o CCT-L3 bater blosc+bitshuffle+zstd em CR e em decode speed ao mesmo tempo, a proposta vira séria. Se só empatar em CR e perder em velocidade, o diferencial tem que ser outro: o runtime CFCS, não o codec.

Ação: adicionar uma linha na tabela §6 com bitshuffle+lz4 e bitshuffle+zstd. É a comparação que o projeto precisa vencer para ser levado a sério.

2.4 A pergunta que o README não responde: INT4
A §2 (proibido INT8/INT4) é uma decisão de escopo, não um argumento de mercado. Mas o leitor vai fazer a conta:

formato	1.9M params	16M params
BF16	3.8 MB	32 MB
CCT-L3 (1.5x)	2.5 MB	21 MB
INT8	1.9 MB	16 MB
INT4	0.95 MB	8 MB
Em memória, INT4 ganha do CCT. Em velocidade, INT4 também (GEMM INT4 em CPU é ~2x mais rápido que BF16 via kernels especializados, e em NPU é 4–8x).

O argumento correto não é "CCT em vez de INT4". É:

"CCT é ortogonal a INT4 e cumulativo. CCT(INT4(BF16(W))) pode chegar a 3–4x de compressão efetiva sem adicionar erro de quantização além do que INT4 já adiciona."

Esse é o posicionamento forte. O README precisa dizer isso explicitamente em §1 ou §2, senão a proposta parece ingênua para quem conhece quantização.

3. Pontos menores mas importantes
3.1 "Bit-exact" tem dois sentidos, e só um está provado
§2 diz: torch.equal(W_original, W_reconstruido) → True para TODO tensor.

Isso é decompressão bit-exact, e está OK. Mas o CCT-GEMM tem rel_err = 3.7e-07 — não é bit-exact. Faz sentido: a ordem de acumulação tile-a-tile difere de torch.matmul em fp32. O README precisa deixar explícito que:

a decompressão é bit-exact;

o GEMM não é bit-exact porque a ordem de acumulação muda, e isso é esperado.

Caso contrário, um leitor desatento pensa que é bit-exact end-to-end.

3.2 Cache e batch não estão medidos
TileCache (LRU) e prefetch estão no runtime/cpu.py mas não há benchmark de hit rate nem de impacto do batch na amortização do decode.

Em batch=1 (inferência típica), cada tile é carregado uma vez por forward, e o decode é pago integralmente. Em batch=N, o decode amortiza. Em geração autorregressiva com KV cache, cada peso é carregado a cada token — o decode é pago N_tokens vezes. Isso é o pior caso possível, e é o caso de uso real.

Ação: medir "tempo por token" de geração com e sem CCT no 1.9M, com decoders Python e C. É o número que decide se o projeto respira.

3.3 tile_rows=64 e padding
Como fica o último tile quando dim % 64 != 0? Como fica o CR em tensores pequenos (embeddings de 259 × 192)? Esses casos são comuns no seu modelo e provavelmente mais rápidos de decodificar, mas também mais sensíveis a overhead de cabeçalho. Vale uma linha na tabela §6.

3.4 Falta um exemplo de integração
O README mostra CCTLinear, cct_matmul, CCTTensor.from_tensor(...) — mas não mostra o padrão de substituição num modelo real:

python
# antes
self.fc = nn.Linear(192, 768)
# depois
self.fc = CCTLinear.from_dense(nn.Linear(192, 768), codec="l3_shuffle")
Sem isso, o leitor não sabe quão invasivo é adotar o CCT. Uma seção curta "How to integrate" muda o tom de "codec" para "runtime".

3.5 Falta caracterizar L0–L3 em tempo, não só em CR
§5 diz o que cada nível faz e o CR que produz. §6 mede CR. Mas o trade-off real é CR × T_decode. Sugestão: uma tabela só:

codec	CR	T_decode (ms por MB)	ECD corrigido
L0	1.24x	?	?
L3	1.45x	?	?
Essa tabela, mesmo com números ruins, mostra que vocês entenderam que CR sozinho não vale nada.

4. O que eu priorizaria
Em ordem de impacto sobre a proposta:

Rodar blosc+bitshuffle+zstd/lz4 como baseline. Se perder, redesenhar o posicionamento. Se ganhar, o projeto tem uma reivindicação forte. Isso é uma tarde de trabalho e muda o patamar do documento.

Adicionar coluna de tempo na tabela de codecs (§5/§6). CR sem tempo é marketing. CR × tempo é engenharia.

Corrigir a definição de ECD para incluir tempo, ou separar em ECD_memória e ECD_tempo. Do jeito que está, 4.0 → 6.1 é o mesmo número que CR, e isso enfraquece o conceito.

Reescrever §3 para dizer a verdade difícil: "em CPU pequena, CCT ganha em memória, não em tempo; tempo só melhora quando memory-bound". Isso fortalece o projeto — vocês deixam de prometer o que não podem entregar.

Adicionar seção "Por que não INT4" com a mensagem de ortogonalidade e combinabilidade. Isso responde à primeira pergunta de qualquer revisor de sistema.

Medir geração autorregressiva (batch=1, N tokens) com e sem CCT. É o caso de uso real do SLM de conversa.

Escrever o decoder C do L3. É o caminho crítico que vocês já identificaram. Mas antes dele, tenha os baselines acima — senão você otimiza o que não precisa.

5. Resposta direta
O README do CCT é bom, tecnicamente honesto e bem estruturado. A parte de §5 (por que L1/L2 falhavam, por que L3 funciona) é o tipo de ciência que faz diferença: vocês mediram, mostraram o teto honesto de 1.18x por entropia, e justificaram o shuffle. Isso é raro.

A fragilidade não está no que está escrito — está no que não está dito:

Que em CPU pequena, CCT não acelera nada, só reduz memória (porque CPU é compute-bound, não memory-bound, nesse regime);

Que INT4 ganha em memória e velocidade isolado, e a tese precisa ser "CCT é cumulativo com INT4";

Que blosc+bitshuffle é o baseline obrigatório, e sem ele o "1.45x vs zstd 1.24x" é ganho sobre um adversário fraco;

Que ECD 4.0 → 6.1 só faz sentido se você ignorar tempo, e isso é circular.

Se o CCT quer ser paper ou produto (e não só "projeto de aprendizado bem documentado"), os próximos dois movimentos são: baseline blosc+bitshuffle e decoder nativo com tempo medido em geração autorregressiva. Esses dois respondem à pergunta que qualquer avaliador faz: "com CR de 1.45x e decode 10x mais lento, quem ganha?"

A resposta honesta é: ganha quem é memory-bound. Se o README disser isso claramente, o CCT deixa de prometer o que não entrega e passa a ter uma tese defensável — e aí o CFCS vira a parte interessante, não o codec.

Quer que eu escreva o bloco do baseline blosc+bitshuffle e a tabela CR × T_decode para você colar no §6? São ~40 linhas de Python + tabela markdown, e resolvem a lacuna mais séria do documento.