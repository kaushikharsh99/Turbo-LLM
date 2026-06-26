<p align="center">
  <img src="assets/logo.png" width="280">
</p>

<h1 align="center">Turbo-LLM</h1>

<p align="center">
Fast Memory-Efficient Inference Engine for Large MoE Models
</p>

<p align="center">
🚀 ~18× Faster Than Initial Prototype • 🧠 30B MoE on ~6GB VRAM • ⚡ 2.01 tok/s (RTX 3050) • 🍎 0.31 tok/s (Apple M4)
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

Currently supported and tested model:

* **Model ID**: `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`

---
Current tested configurations:

### NVIDIA RTX 3050 Laptop

```text
Model:
Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

RAM:
16 GB

GPU:
RTX 3050 Laptop (6 GB)

Speed:
~2.01 tok/s

Peak VRAM:
~5.3 GB
```

### Apple M4

```text
Model:
Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

Unified Memory:
16 GB

GPU:
Apple M4

Speed:
~0.31 tok/s
```
---

## Benchmark

<p align="center">
  <img src="assets/benchmark.png" width="900">
</p>

### Throughput Evolution (RTX 3050)

| Version             | Tokens/sec |
| ------------------- | ---------: |
| Baseline            |       0.11 |
| Prefetch + Cache    |       0.87 |
| Decode Optimization |       1.14 |
| Static Buffers      |       1.33 |
| Active pinning      |       1.96 |
| Current             |       1.96 |
| Warm Cache          |       2.22 |


### Latest Performance

| Platform | Throughput |
|----------|-----------:|
| RTX 3050 Laptop (6 GB) | **2.01 tok/s** |
| Apple M4 (16 GB Unified Memory) | **0.31 tok/s** |
---

## Installation

Clone the repository:

```bash
git clone https://github.com/kaushikharsh99/Turbo-LLM.git
cd Turbo-LLM
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install Turbo-LLM:

```bash
pip install -r requirements.txt
pip install -e .
```

Installing in editable mode registers the `turbo-llm` command globally.
---

## Download Model & Discovery

Turbo-LLM supports automatic downloading and caching of models directly from Hugging Face, as well as loading local weight directories.

### Option A: Auto-Download from Hugging Face
Specify the Hugging Face repo ID. Turbo-LLM automatically downloads and structures it under `~/.turbollm/models/`:

```bash
turbo-llm \
--model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
--prompt "Who is Donald Trump?" \
--chat \
--system "You are a frank, funny, joking explainer."
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

## Usage

You can run inference using either the globally registered `turbo-llm` CLI tool, or directly executing `run.py`.
### Chat Mode

Launch an interactive chat session:

```bash
turbo-llm \
--model ./model \
--prompt "Hello!" \
--chat
```

Start a chat with a custom system prompt:

```bash
turbo-llm \
--model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
--prompt "Who is Donald Trump?" \
--chat \
--system "You are a frank, funny, joking explainer."
```

Chat with custom generation parameters:

```bash
turbo-llm \
--model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
--prompt "Explain quantum computing." \
--chat \
--system "You are a helpful AI assistant." \
--temperature 0.7 \
--top_p 0.95 \
--max_new_tokens 512
```

### Useful CLI Parameters

```text
--model
Model directory path or Hugging Face repository ID.

--prompt
Initial user prompt (required for both completion and chat modes).

--chat
Launch interactive chat mode.

--system
System prompt used during chat mode.

--max_new_tokens
Maximum number of tokens to generate.

--temperature
Sampling temperature.

--top_p
Nucleus sampling threshold.

--benchmark
Enable performance logging.

--config
Path to a custom YAML configuration file.
```

### Configuration Files (YAML)

Instead of specifying runtime options on every command, you can store them in a YAML configuration file.

Example:

```bash
turbo-llm \
--config config/default.yaml \
--prompt "Explain Mixture of Experts"
```

Default configuration (`config/default.yaml`):

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

## Features

- Dynamic expert streaming
- SSD → RAM → VRAM hierarchical caching
- Dynamic VRAM cache sizing
- RAM expert cache
- Layer-by-layer execution
- Double buffering (Ping-Pong buffers)
- Asynchronous prefetching
- Persistent GPU buffers
- KV cache
- Interactive chat mode
- System prompt support
- Automatic model detection
- Multi-model support
- Cross-platform support (CUDA, CPU, macOS)
---

## Contributing

Ideas, benchmarks and pull requests are welcome.

If you find this useful:

⭐ Star the repository
