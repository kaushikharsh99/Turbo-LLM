# Rust migration guide

This is the implementation brief for an incremental, complete migration of
Turbo-LLM's first-party engine and tooling from Python to Rust. A future AI can
start from [RUST_PORT_PROMPT.md](RUST_PORT_PROMPT.md), then use this document as
its detailed specification. Update both this guide and `DEVELOPMENT.md` as facts
change. The Python implementation is the migration reference, not proof that
every registered architecture works.

## Goal and engineering decisions

Deliver a Rust CLI and library that replace the existing first-party Python
runtime, scripts, tests, packaging and CI. A user must be able to install, run,
test and benchmark the supported paths without Python or PyTorch. Retire the
Python code only after its behavior is covered and the Rust replacements pass
the acceptance gates below. Keep the migration reviewable in small, working
commits; do not present a scaffold or a wrapper around Python as a port.

Rust owns checkpoint discovery, model selection, configuration, sessions,
execution orchestration, token generation, expert residency and policy. Calling
maintained native libraries from Rust is allowed: in particular, C/C++
llama.cpp and CUDA kernels may implement the optional GGUF backend. Identify
that boundary, version and build features explicitly. Using a Rust binding to
llama.cpp is a Rust application using a C/C++ inference backend; it does **not**
make GGUF execution a native Turbo expert-streaming implementation. Native
Turbo safetensors streaming must still exist and work independently. Do not
make all inference a llama.cpp subprocess or shell out to Python.

Rust is not a performance guarantee. Today the Python process already calls
compiled tensor and CUDA kernels, and for these models disk paging, memory
traffic, kernel choice, and GPU transfers may dominate. Establish a comparable
baseline, profile, and optimize observed bottlenecks. Do not claim speedups
without measured evidence or quietly change model, quantization, prompt,
context, device placement or timing definition.

## Existing evidence and constraints

Read `AGENTS.md`, `Readme.md`, `DEVELOPMENT.md`, current `git status`, current
tests and `validation/*.json` before changing anything. At the time this guide
was written, the tested baseline is:

- Tiny dense Llama: greedy prefill/cached decode compared with Hugging Face.
- Tiny Qwen3 MoE: CPU/CUDA prefill and cached-decode logits compared with the
  reference, including individual and packed **unquantized** safetensors,
  routing and one-slot cache eviction. FP8 block decoding is tested separately,
  including edge blocks; a real large FP8 streaming run is not validated.
- GGUF: real Stories15M Q4_0 and Qwen3-30B-A3B-Instruct-2507 Q2_K_L runs via
  llama.cpp, including CPU and 16/22-layer RTX 3050 CUDA offload. Q2_K_L uses
  mixed tensor quantizations up to Q8_0. Read the recorded timings and their
  paging, process RSS and whole-GPU-memory caveats rather than treating them as
  stable performance targets. No other 30B quant is yet verified.
- Qwen3.5 MoE is experimental and unverified. OLMoE is a registered but
  unavailable adapter stub. Preserve honest early errors unless a new
  implementation is independently validated; do not call either supported based
  on registration alone.

The current host has roughly 16 GiB RAM and an RTX 3050 with 6 GiB VRAM. At the
last check it had approximately 24 GiB free disk; recheck instead of relying on
that number. `.local-models/` may contain the 19 MB Stories15M GGUF and the
11.2 GB Qwen 30B Q2_K_L GGUF. Check file existence **and checksum** against
`validation/model-sources.json`; an incomplete sparse file may look full size.
The real 30B tests caused significant system swapping. Never delete user files,
environments, downloads or caches to make room. Do not assume this hardware can
hold the full model or that a 30B Q8 download fits.

## Dependency and implementation boundaries

Make and document an explicit, tested dependency choice before relying on it.
Candidate building blocks, subject to version, license, build and architecture
checks, include:

- [Candle](https://github.com/huggingface/candle) for Rust tensor operations and
  CPU/CUDA, [safetensors](https://github.com/huggingface/safetensors) for tensor
  metadata/views, and [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)
  for tokenizer JSON. Candle's advertised models are a starting point, not proof
  of this project's required streaming/cache behavior or a given quantization.
- A maintained Rust binding to [llama.cpp](https://github.com/ggml-org/llama.cpp)
  for GGUF execution, for example [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs).
  Verify the actual pinned crate/upstream revision and its CPU, CUDA, chat,
  tokenization, context and Q2_K_L support before adopting it. Bindings and
  APIs change; do not assume an unverified crate version or CUDA toolchain.
- Cargo workspaces, Rust serde/configuration, error types, CLI parsing and
  structured JSON output; keep dependencies focused and pin reproducibly.

Do not replace custom streaming with a library's full-resident MoE model. A
library may supply tensors, math, tokenizer, kernels or GGUF inference; Turbo
must choose and fetch routed experts under explicit RAM/VRAM limits. Document any
`unsafe`, FFI ownership, native C/C++/CUDA dependency, license, and buffer
lifetime. Use safe abstractions at the boundary and account for asynchronous GPU
work before evicting or reusing its buffers. Do not execute downloaded model code.

## Proposed structure and contracts

Choose a minimal Cargo workspace; package boundaries may evolve if the evidence
warrants it. Keep logical responsibilities separate even if initially in one
crate:

| Rust responsibility | Python source to audit | Required behavior |
| --- | --- | --- |
| CLI, config and validation | `run.py`, `cli/args.py`, `config/`, `runtime/options.py` | Preserve documented flags and sparse YAML overlays; actionable invalid-option errors |
| Backend selection and lifetime | `runtime/generate.py`, `phase2_generate.py`, `runtime/session.py` | One-shot and reusable sessions, cleanup after error, fresh KV per request |
| Registry and model adapters | `runtime/registry.py`, `runtime/model_factory.py`, `models/`, `execution/layouts/` | Exact architecture/capability matching, early errors, model-specific mapping |
| Checkpoints and FP8 | `loader/checkpoint.py`, `loader/model_layout.py`, `loader/quantization.py` | Indexed/unsharded safetensors, individual and packed expert views, validated scale grids |
| Storage and cache policy | `loader/expert_loader.py`, `cache/ram_cache.py`, `runtime/policy.py` | Bounded expert loading, safe eviction and cleanup; honest CPU/GPU/RAM accounting |
| Tensor execution and KV | `runtime/engine.py`, `execution/`, `cache/kv_cache.py`, `cache/qwen35_cache.py` | Correct causal mask, positions, attention, routing, expert math, decode and cache |
| GGUF | `runtime/gguf.py` | Real CPU/CUDA completion and chat through a documented native backend |
| Routing collection and probes | `utils/`, `phase1_probe.py`, `inspect_mapping.py`, `scripts/` | Preserve supported collection/profiling and replace or explicitly retire each utility |
| Tests, build and distribution | `tests/`, `setup.py`, `pyproject.toml`, `requirements.txt`, `.github/` | Rust-native tests, release build, CI, installation and reproducible results |

Suggested interfaces: `CheckpointReader` exposes metadata and bounded tensor
views; `ModelAdapter` declares actual architecture/layout capabilities and
builds per-layer execution; `TensorBackend` owns CPU/CUDA operations and explicit
synchronization; `ExpertStore` controls SSD/RAM/device residency; `KvCache` owns
request state; `GenerationBackend` owns a model session and tokenizer. Use
traits, concrete types and errors appropriate to the chosen libraries rather
than copying Python's dynamic method shapes blindly. The architecture registry
must reject an unsupported model, layout, dtype or kernel **before** substantial
allocation and explain the supported alternative.

The documented CLI currently has `--model`, `--prompt`, `--config`,
`--backend auto|turbo|llama_cpp`, `--max_new_tokens`, `--temperature`, `--top_p`,
`--chat`, `--thinking`, `--system`, `--collect routing`, `--prompts`, `--output`,
and `--benchmark`. Audit actual behavior, including currently inert flags, rather
than asserting every flag already works. Preserve useful current behavior and
spell out intentional changes. Routing JSONL currently records token id/text,
position, generation step, thinking, temperature, top-p and per-layer expert
indices/scores; retain this schema or provide a documented migration. GGUF does
not currently expose Turbo routing collection or the thinking switch.

## Phases and acceptance gates

Each phase should leave an executable, tested state, update the compatibility
matrix, and end with a concise descriptive commit. Preserve real user changes.
The phases can be split further, but do not stop merely because one passes.

### 0. Freeze the reference and produce portable fixtures

1. Record baseline commit, toolchain/device, fixture sources/checksums, commands,
   feature flags and current Python test results. Check disk/RAM/VRAM first.
2. Export tiny deterministic dense and Qwen MoE reference fixtures and expected
   token IDs/logits/routes for prefill **and multiple decode steps**. Include
   individual experts, packed unquantized experts, one-slot eviction, FP8 scale
   grid edges and malformed layouts. Use fixed seeds and record dtype/tolerances.
   No giant weights or user data in Git. Include a small redistributable tokenizer
   fixture, or generate it deterministically in Rust.
3. Preserve the real GGUF provenance and the JSON reports. Ensure fixtures can
   ultimately run without Python: Python may generate or cross-check golden data
   during transition, but checked-in portable fixtures or Rust generators must
   be sufficient for final CI.

Gate: fixtures reproduce the relevant reference behavior and do not accidentally
change the baseline; document any reference defect instead of cementing it.

### 1. Cargo foundation and user-facing contract

Add a Rust workspace, binary, core error types and CI. Choose and record a
minimum supported Rust version and tested dependency versions; commit `Cargo.lock`
for the application. Establish `cargo fmt --check`, `cargo clippy` and
`cargo test` in CPU CI. Add feature flags that make CPU-only builds possible
without CUDA headers, drivers or Python. Implement config defaults/overrides,
validation, backend dispatch, model-path handling and clear unsupported errors.
Do not silently download a repository when a local path was intended; verify
storage before explicit downloads. Document changes to current CLI aliases and
configuration defaults.

Gate: release binary starts outside the checkout, `--help` and error paths are
tested, clean CPU installation/build instructions work, and no Rust runtime path
invokes Python. Preserve the Python reference during migration.

### 2. Safetensors, tokenization and memory primitives

Implement single-file and indexed/sharded safetensors, safe file-boundary and
missing/duplicate-name checks, exact tensor name/shape validation, and bounded
individual expert reads. Support packed gate/up and down tensors by selecting an
expert view, without materializing the entire MoE tensor. Explicitly reject
packed quantized scales until implemented and verified. Implement FP8 128×128
inverse-scale decoding with proper dtype, shape/scale validation and edge
blocks; compare CPU and available CUDA against the frozen oracle.

Implement tokenizer encode/decode, special tokens, EOS, and chat templates for
the supported dense/MoE models. Test UTF-8, multi-token output, tied embeddings,
BOS/EOS and template variations. Do not assume a tokenizer library implements
Hugging Face chat templates: select and verify template handling explicitly.

Build request-scoped KV state and cache policy with explicit bytes for allocated
and active KV, resident expert slots, raw RAM cache, backbone, temporary buffers
and GPU headroom. Respect configured limits; handle impossible budgets and
oversized prompts as errors, not panics or silent overcommit. Map only within
validated files, and establish safe lifetimes for mmap-backed tensor views.

Gate: Rust unit/integration tests reproduce reader, tokenization, scale and cache
fixtures; memory-bound tests prove bounded behavior even with more experts than
slots, including error cleanup.

### 3. Dense Llama inference

Implement embeddings, rotary positions, causal masking, per-layer attention and
MLP, final normalization and final-position logits. Use a correct KV cache for
prefill and incremental decode. Implement greedy EOS handling and temperature /
top-p sampling with an explicit seed and validated bounds. Match token IDs for
greedy decoding and compare logits to the oracle within stated dtype-specific
tolerances across prefill and multiple decode steps. Sampling need not match a
foreign RNG bit-for-bit: test reproducibility with the Rust seed, valid
probabilities and top-p boundary behavior. Support repeated session requests,
reset KV between requests, and release resources on success and failure.

Gate: real tiny local dense model generation succeeds from the Rust binary on
CPU; CUDA parity passes when the available device/build supports it. No Python
process or Python model runtime participates in the Rust path.

### 4. Qwen3 MoE with dynamic expert streaming

Port backbone preloading, layer ordering, router logits, configured top-k and
`norm_topk_prob`, selected expert IDs and normalized/unnormalized weights,
SwiGLU expert computation, optional shared-expert behavior, and residuals.
Prefill must work for multiple tokens routed to different experts; decode must
work across layers and tokens with a cache smaller than the selected set. Keep
model weights on SSD, raw expert cache in bounded RAM, and a bounded device
cache; only bring selected experts to the device. Do not assume one expert size
or architecture layout is universal. Pin or hold leases until GPU operations
complete, never overwrite an in-use slot, propagate I/O/worker errors, and
ensure deterministic cleanup on cancellation or failure. If concurrency or
prefetch is introduced, demonstrate bounded queues and deadlock-free shutdown.

Gate: CPU and available CUDA match frozen tiny Qwen prefill/decode logits and
routes for individual and packed layouts; exercise one-slot eviction, cache
hits/misses, error recovery, and repeated sessions. A real large Qwen FP8
safetensors test is desirable only if a suitable checkpoint already fits or a
hardware/storage check justifies an authorized download. Record the real test or
the exact validation gap; do not claim it from tiny fixtures.

### 5. Complete GGUF backend

Through the selected Rust-accessible native library, load actual GGUF metadata,
tokenizer, chat templates and quantized tensors; implement plain completion,
chat, EOS, greedy/sampling options, context and CPU/partial-GPU controls. Verify
model-file magic, errors, unsupported quantizations, context limits, backend
capabilities and cleanup. Requesting CUDA on a CPU-only build must produce an
actionable error instead of silently claiming offload. Expose/log **actual**
offloaded layers where supported. Preserve the existing distinction: llama.cpp
owns GGUF mapping and quantized execution; it does not exercise Turbo's expert
cache. Do not claim native GGUF Turbo streaming unless separately implemented
and validated.

Gate: compare the Rust GGUF path with a trusted direct runtime for the local
Stories15M Q4_0. On the available machine run the checksum-verified 30B Q2_K_L
file with a short 256-token context, CPU and a cautiously chosen partial CUDA
offload (start at the previously tested 16 layers). Verify output/token behavior,
actual offload, RSS, whole-GPU memory, swap/paging and completion. Report any
hardware/toolchain blocker honestly. Q1, Q3, Q4 and Q8 variants are individual
future tests; never download or claim above the user's Q8 ceiling.

### 6. Collection, lifecycle, observability and tools

Port routing JSONL collection and prompt JSONL reading. Reuse a loaded model
across a batch of prompts, create fresh KV per prompt, flush/close on normal exit
and errors, and reject unsupported backend/collection combinations before work.
Implement useful benchmarking with clear boundaries: model load, time to first
token, prefill, decode, total wall time, token counts, allocated and peak VRAM,
process RSS and system swap/paging. Synchronize GPU measurements where required.
Treat whole-GPU samples as including unrelated processes. Keep instrumentation
optional and avoid perturbing production performance unnecessarily.

Inventory `phase1_probe.py`, `inspect_mapping.py`, `phase2_generate.py`,
`scripts/benchmark_gguf.py` and each Python test. Provide an equivalent Rust
command/test or document why a legacy experiment is obsolete and how its users
migrate. Port CI and packaging, not just the inference library.

Gate: end-to-end CLI and JSONL tests cover happy paths, invalid input, cleanup,
interrupted/error handling and repeated sessions. Reports distinguish actual
measurements from estimates and stored upstream claims.

### 7. Performance and final cutover

Measure release builds, not debug builds. Compare with the recorded Python
baseline using the same checkpoint checksum, prompt/token IDs, generation
settings, context, offloaded layers, CPU threads, device, cache state and system
load. Re-run the Python baseline when needed while it still exists. Separate cold
load, warm runs, prefill, first token and decode. Record repeated runs where
feasible, min/median/range, synchronization, RAM/VRAM/swap, and hardware/software
versions. Investigate regressions and optimize only while preserving correctness
and memory bounds; no specific speedup is promised. A GGUF binding may have
roughly the same kernel performance as Python's llama.cpp binding.

Once all required gates pass, switch the supported installation, `turbo-llm`
entry point and docs to Rust. Remove first-party Python runtime, utilities,
Python tests/build metadata/dependencies and Python CI jobs **only after** a
mapped Rust equivalent, valid retirement rationale, or unsupported-capability
error exists for each. Keep useful historical JSON/Markdown evidence. Do not
remove user models, local environments or untracked files. Avoid retaining a
hidden Python compatibility subprocess. An optional external C/C++/CUDA library
is allowed and must remain explicitly documented.

Final gate: on a fresh supported CPU setup, Rust build/install/CLI and mandatory
tests run without Python/PyTorch installed; the repository's supported
first-party executable, tests and maintenance utilities are Rust. CUDA is an
optional documented build and is actually tested when hardware is available.
Confirm `cargo fmt --check`, `cargo clippy`, `cargo test`, release build, CLI
smoke tests, actual small GGUF and the available 30B CPU/CUDA runs. Confirm
`git status`, update README/architecture/compatibility/benchmarks and record the
final evidence and remaining explicitly unsupported models. Check the repository
for first-party `*.py`, Python packaging/dependency files and Python CI commands;
justify any remaining non-runtime fixture data explicitly. Do not declare
completion while a supported path or an acceptance gate is merely planned.

## Ongoing handoff and progress

Maintain `docs/RUST_PORT_PROGRESS.md` from the first implementation commit. It
must state current commit, owner/date, phase and precise next action, completed
and failing commands, unresolved issues, architecture/format/device matrix,
measured resources and baseline sources, and any hardware/permission blocker.
Update it after each meaningful phase or before context runs out. Keep evidence
honest: a mock is not inference, a small fixture is not a 30B benchmark, and
C++ llama.cpp is not Turbo expert streaming. Any future agent should inspect the
actual tree and state rather than trusting this plan as an implementation.

Continue autonomously across milestones and context compaction, without asking
for routine implementation choices. Follow the user's current authorization for
commits/pushes and never force-push. Stop only when the final gate is met, or
when a real external blocker prevents further independent work; in the latter
case save a concrete handoff, report the blocker and what was verified. Elapsed
time, a difficult milestone, or uncertainty alone is not completion.
