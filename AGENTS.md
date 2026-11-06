# Working on Turbo-LLM

Start with `Readme.md` and `DEVELOPMENT.md`, then inspect `git status` and the
current tests. Other agents may be editing this checkout; preserve unrelated
work and agree on file ownership before parallel edits.

## Preserve the runtime boundaries

- Native Turbo MoE uses safetensors and dynamic expert streaming. Keep backbone
  loading, checkpoint mapping, architecture behavior and memory/cache policy
  separate. Do not replace streaming with full-model residency as a shortcut.
- Native model lifetime belongs to `runtime/session.py`: use its context for
  repeated requests, close resources on success/error, and reject use after close.
  Keep native/GGUF option validation shared through `runtime/options.py`.
  Keep GGUF imports lazy so that its startup does not load the native PyTorch
  stack; test the installed CLI outside the checkout when changing packaging.
- Dense Llama is a separate resident-model path. GGUF delegates metadata,
  tokenization, mapping and quantized execution to optional llama.cpp; it does
  not validate or use Turbo's native expert-cache implementation.
- Keep native FP8 inverse-scale decoding in `loader/quantization.py`, including
  scale-grid validation and partial-block behavior. Packed quantized scale
  support still requires separate mapping and reference tests.
- Register exact architecture names in `runtime/registry.py`. Unsupported
  checkpoints should fail early with an actionable compatibility error. An
  architecture name or adapter stub is not evidence of numerical support. Keep
  adapter-capability preflight before checkpoint tensor access; OLMoE is currently
  an unavailable registered stub, not a runnable experimental path.
- Preserve `trust_remote_code=False`. Keep model weights and local environments
  outside tracked source; `.local-models/` and `.venv/` are ignored.

## Validate meaningful behavior

- Use `.venv/bin/python -m pytest -q` for the local suite. Tiny dense and Qwen MoE
  fixtures compare with Hugging Face; the MoE check covers CPU and available CUDA,
  prefill/cached decode,
  individual and packed experts, and bounded cache slots.
- The real GGUF check runs only with a local fixture (see `TURBO_TEST_GGUF` in
  `tests/test_baseline.py`). A skipped real-model check or passing mock cannot
  establish quantized inference correctness.
- Before downloading/running a large model, inspect available disk, RAM and VRAM.
  Check ongoing downloads/builds described in `DEVELOPMENT.md`; do not duplicate
  them or treat a sparse partial file as complete. Leave user disk cleanup alone.
- Use short, bounded initial generations and record the exact file, quantization,
  runtime build, context, device/offload settings and measured memory/timing.
  The current target is roughly 30B in low GGUF quants, prioritizing supported
  Q1–Q4 variants with a Q8 ceiling. Never imply every quant label is supported.
- Check GPU offload capability in llama.cpp itself. CUDA-enabled PyTorch does not
  make a CPU-only llama.cpp build use the GPU. The tested layered CUDA wheel
  requires explicit `PYTHONPATH` selection from ignored `.local-cuda`; see
  `DEVELOPMENT.md`. Distinguish requested from verified offloaded layers and
  combined prefill throughput from decode-only timing.

## Keep the handoff usable

- Update `Readme.md` for user-visible setup, commands and compatibility changes.
- Update `DEVELOPMENT.md` for architecture boundaries, transient model/build
  status, evidence and actionable remaining work. Resolve stale in-progress notes.
- Save actual benchmark reports in `validation/`; explain whether time includes
  loading, prefill or decode and distinguish RSS from VRAM. Original upstream
  performance claims are archived in `docs/UPSTREAM_BENCHMARKS.md`, not reproduced
  facts. Never fabricate throughput or generalize tiny-model speed to 30B.
- Keep dependency evidence in `constraints-tested.txt` and relevant CI setup in
  `.github/workflows/tests.yml`; a tested local stack is not a universal lockfile.
- Commit coherent tested changes with short, informal but descriptive messages.
  Follow the current user's publication authorization, avoid force pushes, and
  never include weights, secrets or machine-specific caches in a commit.
