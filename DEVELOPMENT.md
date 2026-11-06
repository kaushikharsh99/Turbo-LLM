# Development and handoff

This revival started from upstream commit `9fc73e1`. Commit `ab456d6` restored the
initial local baseline and was pushed to `main`. Use `git log`, `git status` and
current test output for later changes; this document is not a substitute for
checking the checkout.

## Execution boundaries

| Responsibility | Files |
| --- | --- |
| CLI and configuration | `run.py`, `cli/args.py`, `config/config.py`, `config/default.yaml` |
| Backend selection and native model lifetime | `phase2_generate.py`, `runtime/generate.py`, `runtime/session.py` |
| Shared generation option validation | `runtime/options.py` |
| Exact architecture capabilities | `runtime/registry.py`, `runtime/model_factory.py` |
| Adapter contract and model-specific behavior | `models/base.py`, `models/*.py`, `execution/layouts/` |
| Safetensors discovery and per-expert views | `loader/checkpoint.py`, `loader/model_layout.py` |
| Native expert loading and caches | `loader/expert_loader.py`, `cache/ram_cache.py`, `cache/kv_cache.py` |
| Shared native FP8 inverse-scale decoding | `loader/quantization.py` |
| Native layer, routing and expert execution | `runtime/engine.py`, `execution/layer_executor.py`, `execution/router.py`, `execution/moe.py` |
| Execution and cache policy | `runtime/policy.py`; structural extension contracts in `runtime/interfaces.py` |
| Optional GGUF backend | `runtime/gguf.py` |
| Correctness fixtures and CI | `tests/test_baseline.py`, `.github/workflows/tests.yml` |
| CPU GGUF probe and observed reports | `scripts/benchmark_gguf.py`, `validation/*.json` |

Native MoE constructs a meta model, preloads the backbone, and fetches selected
experts from storage as needed. Prefill processes experts one at a time. Decode
uses bounded static cache slots; RAM stores raw expert weights. Preserve this
approach when adding adapters. Dense Llama loads the full model onto the selected
device and bypasses the expert loader. The native engine projects only the final
hidden position to vocabulary logits, avoiding a full prefill sequence of logits.

`runtime.session.open_turbo_session(model, config)` owns native model/loader
lifetime. Its context loads once and yields an engine for repeated `generate`
requests, then closes the loader on normal exit or error and invalidates the
engine. Each generation creates a fresh KV cache while model/expert residency
can be reused. The one-shot `phase2_generate.generate` wraps this context; CLI
routing collection keeps one context for the full prompt collection and closes
the collector in `finally`. GGUF is not accepted by the native session API.

`runtime/options.py` validates generation options for both native and GGUF paths:
new-token count is a nonnegative integer, temperature is finite/nonnegative and
`top_p` is finite in `(0, 1]`. A zero-token request returns empty text without
running inference. Keep this shared contract consistent when adding backends.

GGUF deliberately delegates metadata, embedded tokenization, tensor mapping and
quantized kernels to llama.cpp through `llama-cpp-python`. It does not use the
Turbo expert cache, and native YAML device/dtype/cache budgets do not configure
it. The GGUF entrypoint lazily imports the backend without importing PyTorch;
requests for nonzero GPU layers fail early if llama.cpp reports no offload
support. A native GGUF parser would not supply quantized kernels or model execution;
native Turbo GGUF inference remains separate future work. See the
[llama-cpp-python API](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/).

## Compatibility evidence

| Path | Completed checks | Unverified limits |
| --- | --- | --- |
| Llama dense | Tiny saved checkpoint; greedy prefill and cached generation compared with HF | Real large checkpoints, GPU performance, broad dtype/device combinations |
| Qwen3 MoE streaming | Tiny individual and packed unquantized experts; CPU/CUDA prefill and cached-decode logits compared with HF; cache eviction exercised with one slot | Real large model generation, production FP8 checkpoints, large-model GPU throughput |
| Native FP8 decoder | CPU/CUDA 128×128 inverse-scale checks against explicit scaling, including a 129×257 edge matrix and invalid-grid rejection | Real large FP8 checkpoint generation and packed quantized scales |
| Qwen3.5 MoE | Existing adapters remain registered; experimental | End-to-end numerical correctness and actual checkpoint layouts |
| OLMoE | Registered stub; adapter-capability preflight rejects it before checkpoint tensors are opened | Unavailable for inference until required cache/layout methods and numerical checks exist |
| GGUF | Real Stories15M Q4_0 CPU/reference and CUDA checks; checksum-verified Qwen3-30B-A3B-Instruct-2507 Q2_K_L CPU/CUDA CLI chat and 16-/22-layer CUDA completions | Other quantizations, longer contexts and controlled throughput comparisons remain unverified; no general Q1–Q8 guarantee |

Native safetensors can be indexed/sharded or unsharded. Individual expert weights
use `mlp.experts.N.{gate_proj,up_proj,down_proj}.weight`. Qwen packed unquantized
`gate_up_proj` and `down_proj` tensors are sliced on CPU per expert. Packed
quantized scale layouts are explicitly unsupported. `loader/quantization.py`
shares native 128×128 inverse-scale decoding across weight-loading paths,
validates the scale-grid shape and pads/crops partial edge blocks. Backbone/expert
preflight catches missing required keys; this is not comprehensive validation of
every tensor shape or quantization.

Unknown architecture names fail before weight loading. Adapter-capability
preflight also rejects incomplete registered adapters before checkpoint tensors
are opened: OLMoE currently lacks `create_cache` and `create_layer_layout`, so its
registration is a stub and cannot run inference. Adding a registry entry
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

`config/config.py` merges user YAML sections into packaged defaults independent
of the current directory. Sparse overlays such as a file containing only
`gguf:` are supported. `cp config/default.yaml local.yaml` remains a convenient
complete starting point. Pass `--backend` explicitly when selecting a backend
through the CLI. Native CPU correctness checks should select
`execution.device: cpu` and `dtype: fp32`; the shipped default YAML explicitly
selects fp16, which remains in effect unless overridden.

A built wheel was inspected for the packaged `config/default.yaml` and runtime
modules. The installed `turbo-llm` command was also exercised from `/tmp` with the
real tiny GGUF, confirming that configuration discovery does not depend on the
repository working directory. This lightweight GGUF run is separate from native
PyTorch execution; do not compare its process footprint to a test process that
already imported PyTorch. Generated egg-info is no longer tracked.

## Large-model CPU and CUDA validation

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
[validation/model-sources.json](validation/model-sources.json). The large file
finished downloading from pinned revision
`6c6e8692f43e4ca663f7ece8229a1361090d3a4c`; both downloader verification and an
independent SHA-256 check matched
`2ce6e6a71bc117d8293f3b4e0bb8436dda128e6d06676ad375383a9581aef22a`.
For future downloads, an existing filename or large `ls -lh` size does not prove
completion: sparse partial files may appear nearly full-sized.

The initial [30B CPU report](validation/qwen30b-q2-cpu.json) records
llama-cpp-python 0.3.31 without GPU offload, CPU mmap, four threads and an actual
256-token context. Prompt `The capital of France is` used five tokens and
produced eight completion tokens:

```text
 Paris. What is the capital of Germany
```

Loading took 40.35 s; the combined prefill/generation call took 7.57 s
(about 1.06 completion tok/s including prefill). Peak process RSS was
6,807.71 MiB (about 6.65 GiB). This is a short execution smoke test, not a quality
evaluation or stable decode-only benchmark. CUDA compilation was concurrent.
System swap usage rose from roughly 1.9 GiB to 6.1 GiB during the run, indicating
strong memory pressure across the host. Do not present process RSS as total
model/system memory or assume the model runs comfortably in 6.65 GiB RAM.
The sampled whole-GPU 266 MiB was desktop/other-process usage, not model offload.
A separate [30B CLI chat report](validation/qwen30b-q2-cli.json) records exit 0
and output `Paris` for `What is the capital of France? Answer in one word.` with
an eight-token generation limit and CPU-only 256-context configuration. Wall time
was 28.45 s including startup, loading and generation; peak process RSS was
7,942.34 MiB (about 7.76 GiB). `/usr/bin/time` also recorded 499,384 major page
faults. CUDA compilation was concurrent and prior inference had already caused
substantial system swapping. The CLI does not expose actual token count or TTFT,
so this timing cannot be converted to a reliable decode-throughput number.

To reproduce the completion probe after checking available resources:

```sh
.venv/bin/python scripts/benchmark_gguf.py \
  .local-models/Qwen3-30B-A3B-Instruct-2507-Q2_K_L.gguf \
  --context 256 --tokens 8 --threads 4 --gpu-layers 0 \
  --output validation/qwen30b-q2-cpu-new-run.json
```

The original installed llama-cpp-python 0.3.31 remains CPU-only. A separate
upstream cu132 prebuilt wheel now works in ignored `.local-cuda`. Its
[Stories15M CUDA check](validation/stories15m-q4-cuda.json) actually offloaded the
output layer. The redundant source compilation was intentionally stopped after
the prebuilt wheel worked; it did not produce the tested runtime. During that
attempt CUDA 13.3 rejected GCC 16, while GCC 15 advanced to kernel compilation.

The exact successful layered install and runtime selection were:

```sh
.venv/bin/python -m pip install --target .local-cuda --no-deps --only-binary=:all: \
  --index-url https://abetlen.github.io/llama-cpp-python/whl/cu132 \
  llama-cpp-python==0.3.31
PYTHONPATH=.local-cuda .venv/bin/python -c \
  'from llama_cpp import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())'
PYTHONPATH=.local-cuda .venv/bin/python scripts/benchmark_gguf.py \
  .local-models/Qwen3-30B-A3B-Instruct-2507-Q2_K_L.gguf \
  --context 256 --tokens 8 --threads 4 --gpu-layers 16 \
  --output validation/qwen30b-q2-cuda-new-run.json
```

This `--no-deps` installation assumes the optional GGUF dependencies are already
available in `.venv`. Use an absolute `PYTHONPATH` if running outside the checkout.
For CLI inference, select the same `PYTHONPATH` and set `gguf.n_gpu_layers: 16` in
a YAML overlay. Without that selection, the existing environment still uses the
CPU-only backend. This cu132 wheel worked on local Python 3.13.9 even though
upstream installation guidance lists 3.10–3.12; this is evidence for this exact
machine/wheel, not a general Python 3.13 support guarantee.

Both real 30B GPU probes used four threads, a 256-token context, five prompt
tokens and eight completion tokens. Both returned exactly the CPU continuation.
Verbose logs verified actual offload, not just requested settings:

| Measurement | [16-layer run](validation/qwen30b-q2-cuda.json) | [22-layer run](validation/qwen30b-q2-cuda-22.json) |
| --- | ---: | ---: |
| Verified offload, including output | 16/49 layers | 22/49 layers |
| CUDA model buffer | 3,487.31 MiB | 4,724.91 MiB |
| CUDA compute buffer | 90 MiB | 90 MiB |
| Sampled whole-GPU peak, including desktop | 3,841 MiB | 5,085 MiB |
| Process peak RSS | 5,763.76 MiB | 7,889.46 MiB |
| Loading | 18.61 s | 22.71 s |
| Combined prefill/generation | 11.74 s | 12.29 s |
| Completion tok/s including prefill | 0.68 | 0.65 |
| Verbose-log prompt evaluation, 5 tokens | 8.39 s | 8.91 s |
| Verbose-log decode evaluation, 7 tokens | 3.34 s / 2.10 tok/s | 3.38 s / 2.07 tok/s |

These are individual short runs, not a fair CPU/GPU comparison: compilation,
cache warmth and paging conditions differed. Twenty-two layers used more VRAM
and did not improve the observed decode rate over sixteen; this does not establish
a general ordering. Sixteen is the cautious tested setting. Twenty-two fit this
short run with roughly 1 GiB of sampled GPU headroom, but system swap remained
high (about 6.7 GiB observed while it ran). Longer contexts or other processes can
change the result. These are llama.cpp results, not native Turbo streaming
benchmarks.

The actual [CUDA CLI chat check](validation/qwen30b-q2-cuda-cli.json) also
returned `Paris` with exit 0 for the one-word capital question. Verbose logs
confirmed 16/49 layers offloaded and a 3,487.31 MiB CUDA model buffer. Wall time
was 29.72 s including startup/loading/generation; peak process RSS was
7,833.55 MiB (about 7.65 GiB), with 422,279 major page faults and high preexisting
system swap. No whole-GPU peak was sampled for this CLI run. Its timing is not
a decode-only rate, and the eight-token limit does not reveal the actual number
of generated tokens.

The host has approximately 16 GiB RAM and a 6 GiB RTX 3050. Disk space is changing
as the user frees it; recheck `df -h .`, `free -h` and `nvidia-smi` before large
runs. Short CPU mmap and partial GPU-offload tests succeeded with the caveats
above; do not assume larger contexts will fit. Allow space for the
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

1. Repeat CPU/partial-offload measurements under comparable cache and paging
   conditions, then test longer generations/contexts within measured budgets.
   Evaluate other supported low quants individually without exceeding Q8.
2. Exercise real large Qwen safetensors streaming separately from GGUF. Validate
   FP8 block/scale handling, memory budget enforcement and checkpoint mappings.
3. Add numerical coverage before promoting Qwen3.5 beyond experimental. Complete
   OLMoE cache/layout methods and numerical checks before making it available.
4. Broaden supported dependency/device combinations only with recorded results.

After each implementation change, update the relevant README compatibility row
and this document's boundaries/remaining work. After each measured model run,
add its raw report and update the evidence table with the exact tested
model/quant/backend. Remove transient “in progress” notes when resolved. Keep
[AGENTS.md](AGENTS.md) focused on durable contributor instructions; put changing
run status here. Do not commit model weights, local environments, credentials or
machine-specific download logs.
