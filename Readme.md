<p align="center">
  <img src="assets/logo.png" width="280">
</p>

<h1 align="center">Turbo-LLM</h1>

<p align="center">
Fast Memory-Efficient Inference Engine for Large MoE Models
</p>

<p align="center">
🚀 ~18× Faster Than Initial Prototype • 🧠 30B MoE on ~6GB VRAM • ⚡ ~2 tok/s Warm Decode
</p>

---

## Turbo-LLM

Turbo-LLM is an experimental inference engine designed to run large language models under strict VRAM limits using dynamic expert execution and adaptive GPU residency.

Instead of loading the full model into memory, Turbo-LLM executes only the required components during generation.

Current architecture focuses on:

* Dynamic expert routing
* Sequential MoE execution
* Adaptive expert caching
* Layer streaming
* KV cache generation
* Low VRAM inference

Current tested configuration:

```text
Ram:
16 GB

Model:
Qwen3-30B-A3B-Instruct-FP8

GPU:
RTX 3050 6GB

Peak VRAM:
~5.3GB
```

---

## Benchmark

<p align="center">
  <img src="assets/benchmark.png" width="900">
</p>

### Throughput Evolution

| Version             | Tokens/sec |
| ------------------- | ---------: |
| Baseline            |       0.11 |
| Prefetch + Cache    |       0.87 |
| Decode Optimization |       1.14 |
| Static Buffers      |       1.33 |
| Active pinning      |       1.96 |
| Current             |       1.96 |
| Warm Cache          |       2.22 |

Latest long generation benchmark:

```text
Prompt:
Generate ~500 words

Output:
500 tokens

Total Runtime:
255.1 sec

Speed:
1.96 tok/s

Peak VRAM:
5.31 GB

```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kaushikharsh99/Turbo-LLM.git

cd Turbo-LLM
```

Create environment:

```bash
python -m venv .venv

source .venv/bin/activate
```

Install dependencies and package (Editable Development Mode):

```bash
pip install -r requirements.txt
pip install -e .
```
Installing the package in editable mode registers the `turbo-llm` command globally on your terminal path.

---

## Download Model & Discovery

Turbo-LLM supports automatic downloading and caching of models directly from Hugging Face, as well as loading local weight directories.

### Option A: Auto-Download from Hugging Face
Specify the Hugging Face repo ID. Turbo-LLM automatically downloads and structures it under `~/.turbollm/models/`:

```bash
turbo-llm \
--model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
--prompt "Hello"
```

### Option B: Pre-Download Manually
You can pre-download weight structures using `huggingface-cli`:

```bash
huggingface-cli download \
Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
--local-dir ./model
```

After downloading, verify the files structure like:

```text
Turbo-LLM/
├── model/
│   ├── config.json
│   ├── model.safetensors.index.json
│   └── ...
└── phase2_generate.py
```

---

## Run Inference

You can run inference using either the globally registered `turbo-llm` CLI tool, or directly executing `run.py`.

### 1. Global CLI (Recommended)

Generate text:

```bash
turbo-llm \
--model ./model \
--prompt "Hello my name is"
```

Long generation:

```bash
turbo-llm \
--model ./model \
--prompt "Choose any topic and write an informative article" \
--max_new_tokens 500
```

### 2. Script Execution

Generate text:

```bash
python run.py \
--model ./model \
--prompt "Hello my name is"
```

---

## Advanced Usage

### Sampling Control
Control sampling parameters directly from command line:

```bash
turbo-llm \
--model ./model \
--prompt "Explain Mixture of Experts" \
--max_new_tokens 300 \
--temperature 0.7 \
--top_p 0.95
```

Useful CLI parameters:

```text
--config
Path to custom YAML config file (defaults to config/default.yaml)

--model
Model directory path or Hugging Face repository ID

--prompt
Input prompt string

--max_new_tokens
Max output length (tokens to generate)

--temperature
Sampling temperature

--top_p
Nucleus sampling threshold (top_p)

--benchmark
Enable performance logging
```

### Configuration Files (YAML)
Instead of passing arguments on the command line, you can specify default parameters using configuration YAML files:

```bash
turbo-llm \
--config config/default.yaml \
--prompt "Explain Mixture of Experts"
```

Default configuration layout (`config/default.yaml`):
```yaml
model:
  path: "./model"

runtime:
  max_new_tokens: 50
  temperature: 0.7
  top_p: 0.95

cache:
  gpu_limit: auto
  ram_limit: auto
  expert_limit: auto

memory:
  max_vram_mb: 5800
  max_ram_percent: 35

execution:
  dtype: fp16
  profiling: true
```

---

## Repository Structure

```text
Turbo-LLM/

run.py

assets/
├── logo.png
├── benchmark.png

cli/
├── __init__.py
├── args.py

runtime/
├── __init__.py
├── generate.py

config/
├── __init__.py
├── config.py
├── default.yaml

loader/
├── __init__.py
├── expert_loader.py
├── model_manager.py

execution/
├── router.py
├── moe.py
├── layer_executor.py

cache/
├── kv_cache.py

phase1_probe.py
phase2_generate.py
README.md
```

---

## Testing

Architecture probe:

```bash
python phase1_probe.py
```

Generation:

```bash
python run.py \
--model ./model \
--prompt "Hello"
```

Benchmark:

```bash
python phase2_generate.py
```

Expected benchmark output:

```text
Tokens/sec
Peak VRAM
Peak RAM
Layer timings
```

---

## Features

* Sequential MoE execution
* Dynamic expert routing
* KV cache
* Adaptive expert residency
* Static GPU buffers
* Warm decode acceleration

---

## Roadmap

* [x] Sequential execution
* [x] Dynamic expert loading
* [x] Adaptive residency
* [x] Static GPU buffers
* [ ] CUDA Graph
* [ ] Kernel fusion
* [ ] 3–5 tok/s target

---

## Contributing

Ideas, benchmarks and pull requests are welcome.

If you find this useful:

⭐ Star the repository
