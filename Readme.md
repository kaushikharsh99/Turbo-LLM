<p align="center"><img src="assets/logo.png" width="240" alt="Turbo-LLM"></p>

# Turbo-LLM

Turbo-LLM is an experimental inference runtime for memory-constrained machines.
Its native MoE path keeps backbone weights resident and loads selected experts
from safetensors through a bounded device cache and RAM cache. Prefill processes
experts sequentially, so it does not expand the complete expert set onto the GPU.
Dense Llama inference and an optional llama.cpp GGUF backend are also available.

**GGUF execution uses llama.cpp, not Turbo's native expert-streaming engine.**
The two paths have separate memory policies and validation evidence.

## Current compatibility

| Checkpoint / architecture | Execution path | Evidence and limits |
| --- | --- | --- |
| `LlamaForCausalLM` | Turbo dense | Tiny local checkpoint: greedy output matches Hugging Face through prefill and cached decode. Dense weights must fit on the selected device. |
| `Qwen3MoeForCausalLM` | Turbo expert streaming | Tiny individual and packed unquantized expert fixtures: CPU and CUDA prefill and cached-decode logits match Hugging Face, including a one-slot expert cache. Large real checkpoints remain to be validated. |
| `Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MoeForCausalLM` | Turbo expert streaming | Existing adapters retained; experimental, without end-to-end numerical validation. |
| `OlmoeForCausalLM` | Turbo expert streaming | Existing adapter retained; experimental, without end-to-end numerical validation. |
| Local `.gguf` file | Optional `llama-cpp-python` | Real Stories15M Q4_0 CLI generation and comparison with direct llama.cpp pass. Architecture and quantization support come from the installed llama.cpp build. |

Registration is exact, not a promise that every checkpoint for that architecture
works. Native streaming supports indexed or unsharded safetensors, individual
expert projections, and Qwen packed **unquantized** `gate_up_proj` / `down_proj`
storage sliced per expert. Packed quantized scales are unsupported. The existing
FP8 implementation assumes 128×128 blocks. See [development and validation notes](DEVELOPMENT.md).

The current large-model target is approximately 30B parameters at low GGUF
quantizations, prioritizing Q1–Q4 variants where the runtime supports them, with a
Q8 ceiling. This is a testing target, not a verified compatibility matrix. A 30B
Q2_K_L download is underway; no 30B GGUF throughput is claimed yet.

## Install and test

Python 3.10+ is required. The local revival was tested with Python 3.13,
PyTorch 2.11.0, Transformers 5.14.1 and llama-cpp-python 0.3.31.

```sh
git clone https://github.com/kaushikharsh99/Turbo-LLM.git
cd Turbo-LLM
python -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -c constraints-tested.txt -e '.[test]'
python -m pytest -q
```

Choose the appropriate PyTorch CUDA wheel separately for native GPU execution.
Install the optional GGUF backend with:

```sh
pip install -c constraints-tested.txt '.[gguf]'
```

That command may compile llama.cpp. A CUDA-enabled PyTorch installation does not
make llama.cpp CUDA-enabled; check the backend build separately. The console
command `turbo-llm` is installed into the active environment.

## Run local inference

```sh
# Native dense or registered MoE safetensors checkpoint
python run.py --backend turbo --model /path/to/checkpoint \
  --prompt 'Hello' --max_new_tokens 8 --temperature 0

# llama.cpp-backed GGUF completion; CPU by default
python run.py --backend llama_cpp --model /path/to/model.gguf \
  --prompt 'Once upon a time' --max_new_tokens 24 --temperature 0
```

`--backend auto` is the default: `.gguf` files select llama.cpp and other model
paths select Turbo. For GGUF, download the specific file first and pass its local
path. A Hugging Face repository ID selects the directory-checkpoint downloader,
which can download the whole repository into `~/.turbollm/models/`; check storage
requirements before using it. Always pass `--model` explicitly on a fresh clone:
the tracked `model` symlink refers to the original developer's local cache.

`--chat` formats a single prompt with the model's chat template; it is not an
interactive chat loop. `--system` supplies its optional system message. GGUF chat
requires embedded `tokenizer.chat_template`. GGUF does not expose routing
collection or the `--thinking on` control. Native models disable remote code.

The packaged `config/default.yaml` provides runtime defaults. Copy it before
customizing it:

```sh
cp config/default.yaml local.yaml
python run.py --config local.yaml --model /path/to/model.gguf \
  --prompt 'Hello' --max_new_tokens 8
```

For native execution, set `execution.device` to `auto`, `cpu`, `cuda` or `mps` and
`execution.dtype` to `fp32`, `fp16` or `bf16`. The shipped YAML explicitly requests
fp16; choose fp32 when testing native CPU correctness. Native expert-cache
controls are `cache.expert_limit`, `memory.max_vram_mb` and
`memory.max_ram_percent`. These do not control llama.cpp.

For GGUF, add this section to the copied YAML:

```yaml
gguf:
  n_ctx: 512
  n_batch: 64
  n_threads: 4
  n_gpu_layers: 0
  seed: 0
```

Increase `n_gpu_layers` only with a CUDA-enabled llama.cpp build and enough free
VRAM. Short contexts are useful for initial memory-constrained checks.

## Validation and contribution

The default test suite creates tiny local dense and MoE fixtures; no model
download is required. Real GGUF testing runs when `TURBO_TEST_GGUF` points to a
local file, or the local Stories15M test file exists:

```sh
TURBO_TEST_GGUF=/path/to/stories15M-q4_0.gguf python -m pytest -q
python scripts/benchmark_gguf.py /path/to/model.gguf \
  --context 512 --tokens 24 --output validation/my-run.json
```

The benchmark script measures llama.cpp loading, combined prefill/generation
time and process peak RSS. It also accepts `--gpu-layers`, `--threads`, `--chat`
and `--prompt`, and samples whole-GPU memory (including other processes). The
recorded [tiny-model CPU run](validation/stories15m-q4.json) establishes neither
native Turbo streaming speed nor GPU performance; it is not a factual-answer
quality evaluation. The runtime rounded its requested 128-token context to 256. Model revisions and SHA-256 values are recorded in
[model-sources.json](validation/model-sources.json). Original
README throughput figures are preserved as [unreproduced upstream claims](docs/UPSTREAM_BENCHMARKS.md).

For architecture boundaries, reproducible checks and remaining work, read
[DEVELOPMENT.md](DEVELOPMENT.md). AI contributors should start with
[AGENTS.md](AGENTS.md). Keep model weights, virtual environments, credentials and
machine-specific caches out of commits.
