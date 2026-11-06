# Development and handoff

This revival started from upstream commit `9fc73e1`. Commit `ab456d6` restored the
initial local baseline and was pushed to `main`. Use `git log`, `git status` and
current test output for later changes; this document is not a substitute for
checking the checkout.

## Execution boundaries

| Responsibility | Files |
| --- | --- |
| CLI and configuration | `run.py`, `cli/args.py`, `config/config.py`, `config/default.yaml` |
| Backend selection and model lifetime | `phase2_generate.py`, `runtime/generate.py` |
| Exact architecture capabilities | `runtime/registry.py`, `runtime/model_factory.py` |
| Adapter contract and model-specific behavior | `models/base.py`, `models/*.py`, `execution/layouts/` |
| Safetensors discovery and per-expert views | `loader/checkpoint.py`, `loader/model_layout.py` |
| Native expert loading and caches | `loader/expert_loader.py`, `cache/ram_cache.py`, `cache/kv_cache.py` |
| Native layer, routing and expert execution | `runtime/engine.py`, `execution/layer_executor.py`, `execution/router.py`, `execution/moe.py` |
| Execution and cache policy | `runtime/policy.py`; structural extension contracts in `runtime/interfaces.py` |
| Optional GGUF backend | `runtime/gguf.py` |
| Correctness fixtures and CI | `tests/test_baseline.py`, `.github/workflows/tests.yml` |
| CPU GGUF probe and observed reports | `scripts/benchmark_gguf.py`, `validation/*.json` |

Native MoE constructs a meta model, preloads the backbone, and fetches selected
experts from storage as needed. Prefill processes experts one at a time. Decode
uses bounded static cache slots; RAM stores raw expert weights. Preserve this
approach when adding adapters. Dense Llama loads the full model onto the selected
device and bypasses the expert loader.

GGUF deliberately delegates metadata, embedded tokenization, tensor mapping and
quantized kernels to llama.cpp through `llama-cpp-python`. It does not use the
Turbo expert cache, and native YAML device/dtype/cache budgets do not configure
it. A native GGUF parser would not supply quantized kernels or model execution;
native Turbo GGUF inference remains separate future work. See the
[llama-cpp-python API](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/).

## Compatibility evidence

| Path | Completed checks | Unverified limits |
| --- | --- | --- |
| Llama dense | Tiny saved checkpoint; greedy prefill and cached generation compared with HF | Real large checkpoints, GPU performance, broad dtype/device combinations |
| Qwen3 MoE streaming | Tiny individual and packed unquantized experts; CPU/CUDA prefill and cached-decode logits compared with HF; cache eviction exercised with one slot | Real large model generation, production FP8 checkpoints, large-model GPU throughput |
| Qwen3.5 MoE and OLMoE | Existing adapters remain registered | End-to-end numerical correctness and actual checkpoint layouts |
| GGUF | Real Stories15M Q4_0 CLI generation; deterministic output comparison with direct llama.cpp; mock API/cleanup check | 30B generation and CUDA offload are pending; no general Q1–Q8 guarantee |

Native safetensors can be indexed/sharded or unsharded. Individual expert weights
use `mlp.experts.N.{gate_proj,up_proj,down_proj}.weight`. Qwen packed unquantized
`gate_up_proj` and `down_proj` tensors are sliced on CPU per expert. Packed
quantized scale layouts are explicitly unsupported. Existing FP8 dequantization
assumes 128×128 blocks. Backbone/expert preflight catches missing required keys;
this is not comprehensive validation of every tensor shape or quantization.

Unknown architecture names fail before weight loading. Adding a registry entry
alone does not establish support: implement its adapter/layout, verify checkpoint
mapping and cache semantics, compare numerical results with a reference, and
update the evidence matrix only after those checks pass. Remote code is disabled.

## Reproducible local workflow

```sh
python -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -c constraints-tested.txt -e '.[test]'
python -m pytest -q
```

The existing development `.venv` was created with `--system-site-packages` to
avoid duplicating the installed PyTorch CUDA stack. That setup is convenient on
this machine, but not a clean dependency-isolation test. Local versions are
recorded in `constraints-tested.txt`; they are a tested stack, not a universal
platform lockfile. CI uses Python 3.11 and CPU fixtures. Local work uses Python
3.13. Check CI results separately from local passes.

Optional GGUF installation and a real checkpoint comparison:

```sh
pip install -c constraints-tested.txt '.[gguf]'
TURBO_TEST_GGUF=/path/to/stories15M-q4_0.gguf python -m pytest -q
python run.py --backend llama_cpp --model /path/to/stories15M-q4_0.gguf \
  --prompt 'Once upon a time' --max_new_tokens 24 --temperature 0
```

Without a local GGUF fixture, the real GGUF test skips. CI still exercises the
mock backend contract, which cannot establish tokenizer or kernel correctness.
The tiny dense and Qwen tests require no external checkpoint download. Passing
those fixtures does not reproduce the original README's large-model benchmarks.

Use the packaged default YAML as a complete starting point when running the CLI:
`cp config/default.yaml local.yaml`. The CLI reads model and runtime sections
from the supplied YAML, so a file containing only `gguf:` is insufficient. Pass
`--backend` explicitly when selecting a backend through the CLI. Native CPU
correctness checks should select `execution.device: cpu` and `dtype: fp32`; the
shipped default YAML explicitly selects fp16.

## Large-model validation in progress

The current target is roughly 30B parameters in supported low GGUF quantizations,
prioritizing Q1–Q4 and never going above Q8. Labels such as Q2_K_L are mixtures of
tensor encodings, not a universal bit-width promise. Test each concrete file and
runtime combination. A 30B Q8 download is not required if it cannot fit.

The initial real file selected is
[bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF),
`Qwen_Qwen3-30B-A3B-Instruct-2507-Q2_K_L.gguf` (11,212,519,424 bytes).
Its local output filename omits the initial `Qwen_` prefix:
`.local-models/Qwen3-30B-A3B-Instruct-2507-Q2_K_L.gguf`. Its embed/output
weights use Q8_0. Immutable source revisions and expected SHA-256 values for
both this file and Stories15M are in
[validation/model-sources.json](validation/model-sources.json). The large download
is pinned to revision `6c6e8692f43e4ca663f7ece8229a1361090d3a4c` and configured
to verify the recorded checksum on completion; this does not mean the partial
file is already verified. A resumable download is in `.local-models/`; check the process
and `/tmp/turbo-aria.log` before restarting it. A sparse partial file can have an
apparently large size: an existing filename or `ls -lh` output does not establish
that the download finished. Do not run inference until the downloader succeeds.

The original installed llama-cpp-python 0.3.31 is CPU-only. A CUDA build was
started in `.venv` using the local CUDA 13.3 toolkit, SM86 and two build workers.
GCC 16 failed toolkit validation; GCC 15 is the current build choice:

```sh
CC=gcc-15 CXX=g++-15 CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS='-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15' \
CMAKE_BUILD_PARALLEL_LEVEL=2 \
.venv/bin/python -m pip install --no-cache-dir --force-reinstall --no-deps llama-cpp-python==0.3.31
```

This is a machine-specific build attempt, not a confirmed working installation.
Check `/tmp/turbo-cuda-build.log` and verify capability in a fresh process:

```sh
.venv/bin/python -c 'from llama_cpp import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())'
```

The host has approximately 16 GiB RAM and a 6 GiB RTX 3050. Disk space is changing
as the user frees it; recheck `df -h .`, `free -h` and `nvidia-smi` before large
runs. The first 30B test should be short CPU mmap inference with a small context,
then bounded partial GPU offload if the CUDA build works. Allow space for the
checkpoint, build artifacts, KV cache and runtime workspace. Memory-mapped files
can page from disk and may be slow when RAM is scarce. Do not equate mapped file
size with peak RSS or claim a 6 GiB-only model footprint.

Record real runs under `validation/`. Include model source and revision/checksum,
file quantization, runtime version/build, device/offload settings, context,
prompt, token counts, loading/prefill/decode timing definitions, output and memory
measurements. The current script accepts `--gpu-layers`, `--threads`, `--context`, `--chat`
and `--prompt`; it measures generation as a combined call including prefill, so
its throughput is not decode-only. `max_rss_mib` is process peak RSS, not VRAM.
`system_gpu_peak_used_mib` samples the whole GPU, including desktop and unrelated
processes, and may miss short peaks. Record requested and actual context sizes:
the Stories15M runtime rounded 128 requested tokens to 256. Its recorded text is
a completion smoke test, not evidence of factual quality. Do not infer
large-model speed from the tiny fixture.

## Next work and keeping this handoff current

1. Finish the 30B Q2_K_L download; verify the file and record provenance.
2. Finish or diagnose the CUDA build. Establish a bounded real 30B CPU result,
   then evaluate partial offload with measured CPU/RAM/VRAM use.
3. Exercise real large Qwen safetensors streaming separately from GGUF. Validate
   FP8 block/scale handling, memory budget enforcement and checkpoint mappings.
4. Add numerical coverage before promoting Qwen3.5 or OLMoE beyond experimental.
5. Broaden supported dependency/device combinations only with recorded results.

After each implementation change, update the relevant README compatibility row
and this document's boundaries/remaining work. After each measured model run,
add its raw report and update the evidence table with the exact tested
model/quant/backend. Remove transient “in progress” notes when resolved. Keep
[AGENTS.md](AGENTS.md) focused on durable contributor instructions; put changing
run status here. Do not commit model weights, local environments, credentials or
machine-specific download logs.
