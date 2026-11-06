# Rust port progress

Status: **not started**. This document is an initial handoff, not evidence of a
Rust implementation. The Python reference before the planning documents is
commit `2a97665`; check `git log` and `git status` for the current state.

- Current phase: 0 — freeze the reference and produce portable fixtures.
- Next action: inspect the current tree and hardware, run the applicable Python
  reference tests, record the results, and build small portable golden fixtures as
  specified in [RUST_PORT_GUIDE.md](RUST_PORT_GUIDE.md).
- Last reported reference result: 25 tests passed in the previous revival session;
  this documentation task did not rerun them. Real 30B Q2_K_L CPU and 16-/22-layer
  CUDA GGUF reports are under `validation/`. Those use llama.cpp, not Turbo's
  native expert streaming.
- Known gaps before migration: no validated real 30B native safetensors streaming;
  Qwen3.5 is experimental; OLMoE is an unavailable stub; 30B quantizations other
  than Q2_K_L are not verified. See `DEVELOPMENT.md` for details.
- Hardware: previous session observed 16 GiB RAM and RTX 3050 6 GiB. That 30B
  inference test caused substantial swapping. Disk and available memory change;
  recheck them and model checksums before building or running large models.
- Blockers: none assessed for the migration yet. Recheck CUDA toolchain, Rust
  dependencies, available storage and upstream library capabilities as work
  begins; a difficult task alone is not a blocker.

The implementation agent must replace this initial summary with dated, exact
results and current commit after each milestone, and leave a specific next action
before context handoff. Never put model files or local environments in Git.
