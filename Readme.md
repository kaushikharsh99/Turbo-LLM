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
| `OlmoeForCausalLM` | Registered stub; unavailable | Adapter lacks required cache/layout methods. Preflight raises a compatibility error before opening checkpoint tensors. |
| Local `.gguf` file | Optional `llama-cpp-python` | Real Stories15M Q4_0 checks and Qwen3-30B-A3B-Instruct-2507 Q2_K_L CPU/CUDA CLI chat plus 16- and 22-layer CUDA completion checks pass. These are short runs; support depends on the llama.cpp build. |

Registration is exact, not a promise that every checkpoint for that architecture
works. Native streaming supports indexed or unsharded safetensors, individual
expert projections, and Qwen packed **unquantized** `gate_up_proj` / `down_proj`
storage sliced per expert. Packed quantized scales are unsupported. The existing
FP8 decoder uses 128×128 inverse-scale blocks, validates the scale grid and
handles partial edge blocks. CPU/CUDA fixture checks pass; real large FP8
checkpoints remain untested. See [development and validation notes](DEVELOPMENT.md).

The current large-model target is approximately 30B parameters at low GGUF
quantizations, prioritizing Q1–Q4 variants where the runtime supports them, with a
Q8 ceiling. A checksum-verified 30B Q2_K_L CPU smoke test produced eight tokens
at about 1.06 tok/s including prefill, after about 40.35 s loading. This short run
experienced strong system memory pressure; its 6.65 GiB process peak RSS is not
the total model/system footprint. See the [raw run](validation/qwen30b-q2-cpu.json)
and [measurement caveats](DEVELOPMENT.md). A separate
[30B CLI chat check](validation/qwen30b-q2-cli.json) returned `Paris` successfully;
its 28.45 s wall time includes startup/loading/generation, not just decoding.
Actual CUDA partial-offload checks also succeeded: 16 layers used about
3,841 MiB sampled whole-GPU memory and the llama.cpp log reported 2.10 decode
tok/s over seven decode tokens. The combined prefill/generation rate was
0.68 completion tok/s. These short, memory-constrained runs are not a controlled
CPU/GPU comparison; see [GPU results](validation/qwen30b-q2-cuda.json) and
[22-layer results](validation/qwen30b-q2-cuda-22.json). The actual
[CUDA CLI chat check](validation/qwen30b-q2-cuda-cli.json) also returned `Paris`
with 16/49 layers verified offloaded. Other quantizations and longer-context
behavior remain unverified.

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
make llama.cpp CUDA-enabled; check the backend build separately. Requests for
GPU layers fail early when the installed llama.cpp build has no offload support. The console
command `turbo-llm` is installed into the active environment. Its packaged default
configuration is found independently of the working directory. The GGUF path
lazily imports its backend and does not load PyTorch for inference startup.

A working upstream cu132 wheel was tested separately in ignored `.local-cuda`:

```sh
.venv/bin/python -m pip install --target .local-cuda --no-deps --only-binary=:all: \
  --index-url https://abetlen.github.io/llama-cpp-python/whl/cu132 \
  llama-cpp-python==0.3.31
PYTHONPATH=.local-cuda .venv/bin/python -c \
  'from llama_cpp import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())'
```

This uses the GGUF dependencies already installed above. Select this wheel for
inference with `PYTHONPATH=.local-cuda .venv/bin/python run.py ...` and set
`gguf.n_gpu_layers: 16` in YAML for the cautious tested 30B setting. CPU remains
the default otherwise. This exact wheel worked locally on Python 3.13; upstream
installation guidance lists Python 3.10–3.12, so broader Python/platform
compatibility is not established.

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

The packaged `config/default.yaml` provides runtime defaults. A user YAML file
overlays its sections onto those defaults, so a sparse file containing only
`gguf:` is supported. Copying the complete defaults is also useful:

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

For GGUF, this section can be added to a copied YAML or saved on its own:

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

For repeated native requests, load the model once with a session:

```python
from runtime.session import open_turbo_session

config = {"execution": {"device": "cpu", "dtype": "fp32"}}
with open_turbo_session("/path/to/checkpoint", config) as engine:
    first = engine.generate("Hello", max_new_tokens=8, config=config)
    second = engine.generate("Another prompt", max_new_tokens=8, config=config)
```

Each call gets a fresh KV cache; the loaded model and expert cache are reused.
Exiting the context closes storage/cache resources, including on an exception;
the engine cannot generate after it closes. CLI routing collection similarly
reuses one native session across prompts and closes its collector on exit. GGUF
uses its separate per-call llama.cpp lifetime.

Both backends require a nonnegative integer `max_new_tokens`, finite nonnegative
`temperature`, and finite `top_p` in `(0, 1]`. Zero new tokens returns empty text.

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
