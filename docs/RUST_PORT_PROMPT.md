# Copy/paste prompt for the implementation agent

```text
You are taking over the Turbo-LLM Rust migration. Work in the existing local
checkout at /home/harsh/Documents/ChatGPT/Turbo-LLM (if this path differs in
your environment, locate the user's existing clone; don't overwrite files).

Read AGENTS.md, docs/RUST_PORT_GUIDE.md, DEVELOPMENT.md, the current tests and
git status first. Implement the guide, not another plan. The user wants all
first-party runtime, CLI, maintenance tools, tests, packaging and CI migrated
from Python to Rust, incrementally, until the final acceptance gate is met.
Keep the existing low-memory custom MoE expert-streaming architecture. Using
optimized Rust crates and native C/C++/CUDA dependencies such as llama.cpp is
allowed, but a Rust wrapper around the old Python process does not count as a
port. GGUF through llama.cpp remains distinct from native Turbo streaming.

Start by preserving the Python reference, recording its current baseline, and
creating portable small fixtures. Progress through the guide's Cargo foundation,
checkpoint/FP8/tokenization/cache, dense Llama, Qwen3 MoE streaming, GGUF,
collection/tooling, performance validation and final cutover. Preserve tested
features and early actionable errors. Qwen3.5 is experimental and OLMoE is an
unavailable stub; do not claim support without implementation and evidence.
Never replace dynamic expert streaming with full-model residency.

Test each meaningful increment. Compare deterministic tiny Llama and Qwen MoE
prefill/decode results to the frozen reference, including one-slot cache
behavior and FP8 block edges. Test actual GGUF inference, not just mocks. The
existing verified local targets include Stories15M Q4_0 and Qwen3-30B-A3B-
Instruct-2507 Q2_K_L; consult validation/model-sources.json and check hashes.
Use a short context and cautious partial GPU offload when available; check disk,
RAM, VRAM and swap first. The machine has around 16 GiB RAM and a 6 GiB RTX
3050, and prior 30B tests caused substantial swapping. Don't delete user files
or environments for space. Other low GGUF quants may be tested when feasible,
but never download above Q8 or imply untested formats are verified.

Benchmark release builds against equivalent baselines with the same model,
prompt, settings, device placement and timing boundaries. Record startup, TTFT,
prefill, decode, RSS, GPU memory and swap/paging separately where measurable.
Do not promise a speedup simply because the code is Rust. Report real errors,
unsupported configurations, and hardware limitations honestly.

Maintain docs/RUST_PORT_PROGRESS.md with the current state, evidence and next
specific action at every milestone and before context compaction. Work
independently, make reasonable implementation decisions, and continue across
milestones; don't stop after a scaffold or ask me to choose routine details.
Commit coherent, tested changes with concise descriptive messages. Push only if
my current session authorizes it, never force-push, and keep model files,
environments and secrets out of Git. Preserve unrelated user changes.

Finish only when the guide's final gate is satisfied: the supported first-party
engine, CLI, tools, tests, build and CI are Rust; clean CPU build and usage work
without Python/PyTorch; supported dense, native MoE streaming and real GGUF paths
are validated; documentation and compatibility limits match the evidence; and
no required first-party Python runtime or Python-only tooling remains. Native
C/C++/CUDA library dependencies may remain, explicitly documented. If a genuine
external blocker prevents all further independent progress, leave a concrete
handoff with what was attempted, passed and remains blocked; do not call an
unfinished port complete.
```
