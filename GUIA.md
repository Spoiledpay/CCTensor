# Guia — Mini-GPT do Alfabeto (treinar + conversar)

Modelo minúsculo que lê um alfabeto em JSON e responde perguntas:
`Mostre o alfabeto?`, `Ultimas 3?`, `Total?`, `Primeira?`, `Ultima?`,
`Depois de C?`, `Antes de B?`.

Você **não precisa passar o alfabeto**: o chat usa sozinho o alfabeto do
dataset gravado no checkpoint. Só digite a pergunta.

---

## 1. Ambiente (uma vez)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install onnx onnxruntime
```

> Proxy corporativo com `SSL: CERTIFICATE_VERIFY_FAILED`? Adicione
> `--trusted-host pypi.org --trusted-host files.pythonhosted.org --timeout 300`.

---

## 2. Treinar

```powershell
# Treino único: train.py --mode especialista|generalista|slm (defaults no config.json)

# Especialista padrão (rápido, ~1-2 min, 3 alfabetos de 6-10 letras)
.\.venv\Scripts\python.exe train.py --mode especialista

# Especialista do alfabeto COMPLETO (A-Z, o da pasta input/) — recomendado
.\.venv\Scripts\python.exe train.py --mode especialista --min-length 26 --max-length 26 `
  --block-size 192 --n-alphabets 1 --steps 400 --out modelos/especialista_full.pt

# Generalista (experimento: tenta generalizar p/ alfabetos novos, ~10-18 min)
.\.venv\Scripts\python.exe train.py --mode generalista

# SLM (dataset curado input/dialogos.jsonl -> fine-tune Qwen -> GGUF/Ollama)
.\.venv\Scripts\python.exe train.py --mode slm --build-dataset-only
.\.venv\Scripts\python.exe train.py --mode slm

# 50 epocas (curva de aprendizado: barra/heartbeat, probe de respostas, CSV)
.\.venv\Scripts\python.exe train.py --mode especialista --min-length 26 --max-length 26 `
  --block-size 192 --n-alphabets 1 --epochs 50 --eval-every 5 --out modelos/especialista_full_6m.pt
```

* O treino lê as letras-base de `input/alfabeto.json`.
* Ao final, o `.pt` é salvo **e o `.onnx` é reexportado sozinho**
  (`especialista.pt` → `especialista.onnx`).
  Pule com `--no-onnx`. Gere INT8 com `--onnx-int8`.
* `config.json` tem os defaults; qualquer flag da CLI sobrepõe o config.

---

## 3. Conversar (chat)

```powershell
# Chat padrão (PyTorch)
.\.venv\Scripts\python.exe chat.py

# Chat com o alfabeto completo (A-Z)
.\.venv\Scripts\python.exe chat.py --checkpoint modelos/especialista_full.pt

# Chat via ONNX (prova que o .onnx responde igual ao .pt)
.\.venv\Scripts\python.exe chat.py --checkpoint modelos/especialista_full.pt --backend onnx
```

Dentro do chat, pode falar livre — o chat traduz para a forma do treino:

```
> quantas letras sao no alfabeto?
  (entendi como: Total?)
A: 26
> mostre a ultima letra
  (entendi como: Ultima?)
A: Z
> qual vem depois do d?
  (entendi como: Depois de D?)
A: E
> Mostre o alfabeto?
A: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
```

Comandos:

| comando | o que faz |
|---|---|
| `/alfabetos` | lista os alfabetos do dataset |
| `/usar 2` | troca para o alfabeto nº 2 |
| `/alfabeto` | mostra o alfabeto em uso |
| `/exemplos` | sugere perguntas |
| `/help` | ajuda |
| `/sair` | encerra |

Exemplo de troca (só no `especialista.pt`, que tem 3 alfabetos):

```
> /alfabetos
  1. ABCDEFGHIJ <-- em uso
  2. XRKWLSNOU
  3. QPMZYT
> /usar 2
> Mostre o alfabeto?
A: X, R, K, W, L, S, N, O, U
```

> O JSON é montado sozinho a cada pergunta. `--alfabeto` manual só existe
> como override avançado.

---

## 4. Uma pergunta só (sem entrar no chat)

```powershell
.\.venv\Scripts\python.exe infer.py --checkpoint modelos/especialista_full.pt --prompt "Mostre o alfabeto?"
.\.venv\Scripts\python.exe infer.py --prompt "Depois de C?"
.\.venv\Scripts\python.exe infer.py --backend onnx --prompt "Total?"
```

* Sem `--letters`: usa o alfabeto do dataset (nº 1).
* Avançado: `--letters MNYBI`, `--usar 2`, `--alphabet canonical`.
* `--raw`: envia o texto cru ao modelo, sem traduzir via regex
  (FASE 1: funciona p/ paráfrases treinadas, ex.: `quantas letras tem`,
  `mostre a ultima letra` — sem `?` e em minúsculas também).

---

## 5. GGUF / Ollama (exportador único)

`export_gguf.py` deteta o tipo de entrada e faz o caminho certo:

```powershell
# TinyGPT (.pt) -> GGUF na arquitetura gpt2 (Ollama) + Modelfile oficial
.\.venv\Scripts\python.exe export_gguf.py modelos/tinygpt_curado.pt --create
.\.venv\Scripts\python.exe export_gguf.py modelos/tinygpt_curado.pt --out modelos/tinygpt.gguf --ollama-name tinygpt

# SLM HF (Qwen fine-tuned) -> GGUF via llama.cpp (+ quantizacao)
.\.venv\Scripts\python.exe export_gguf.py --from-hf modelos/slm-qwen --quant q4_k_m --create

# estado do ambiente (ollama, gguf, llama.cpp)
.\.venv\Scripts\python.exe export_gguf.py --check
```

* O TinyGPT é exportado como **`gpt2`** (arquitetura real do llama.cpp) —
  exportar como "tinygpt" seria recusado pelo Ollama.
* Os nomes dos tokens usam o **byte-encode do GPT-2** (espaço → `Ġ`): sem isto
  o llama.cpp descarta o espaço e o modelo vê outro prompt (`[UNK_BYTE_0x20]`).
* O `Modelfile` (`ollama/Modelfile.tinygpt`) monta o prompt no formato de
  treino: `<bos>{"alfabeto":"A-Z"}\nQ: {{ .Prompt }}\nA: `
  — sem o template, o modelo devolve o eco do JSON em vez de responder.
  **Com o template, escreve-se só a pergunta** (`ollama run tinygpt "Total?"`).
* **Armadilha do Windows**: o `Modelfile` tem de ser escrito com LF (`\n`).
  Com CRLF, o `\r` entra no prompt e a resposta sai corrompida. O exportador
  já grava com `newline="\n"`.
* Teste: `ollama run tinygpt "Depois de C?"` → `D`
* Bateria: `.\.venv\Scripts\python.exe teste_ollama_tinygpt.py --model tinygpt --n 2`

---

## 6. ONNX / NPU

* `.pt` = checkpoint PyTorch (treino + chat). `.onnx` = grafo de inferência (deploy).
* O `.onnx` é gerado **a partir** do `.pt` ao fim de todo treino. Se retreinar, não precisa exportar na mão.
* Export manual:
  ```powershell
  .\.venv\Scripts\python.exe export_onnx.py --checkpoint modelos/especialista_full.pt --out modelos/especialista_full.onnx
  .\.venv\Scripts\python.exe export_onnx.py --int8
  ```
* Fluxo NPU: `PyTorch -> ONNX -> (INT8) -> OpenVINO / QNN / RKNN / Core ML`.

---

## 7. Deu ruim?

| sintoma | causa / solução |
|---|---|
| Resposta cortada (`A, B, ... N`) | resposta longa precisa de mais tokens — o padrão já é `auto (block_size)`. Se passou `--max-new-tokens` pequeno, tire o flag. |
| `excede block_size` no treino | exemplo não cabe. Aumente `--block-size` (A-Z precisa de `192`). |
| Resposta lixo (`U`, letra solta) | alfabeto fora do treino. Use `/alfabetos` + `/usar N`, ou o checkpoint full para A-Z. |
| `Caractere fora do vocabulario` no `--eval` antigo | versão antiga avaliava tarefa não treinada. Atualizado: o eval filtra por `tasks` do checkpoint. |
| `ONNX nao encontrado` com `--backend onnx` | reexporte (seção 5) ou rode sem `--backend onnx`. |
| `next/prev` erra em alfabeto novo | esperado: especialista **memoriza**, generalização de indução é item em aberto. |

Detalhes de arquitetura e resultados por categoria: `Projeto.md`.
