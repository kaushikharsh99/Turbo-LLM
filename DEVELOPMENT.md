# Local revival baseline

Checkout: public upstream commit `9fc73e1`. Changes are local and unpushed.

Use Python 3.10+ (locally tested with 3.13), PyTorch 2.11.0 and Transformers
5.14.1. The local `.venv` uses system site packages to avoid duplicating the
large CUDA installation on a nearly full disk:

```sh
source .venv/bin/activate
python -m pytest -q
python run.py --model /path/to/llama-checkpoint --prompt 'Hello' --max_new_tokens 8
python run.py --model /path/to/model.gguf --prompt 'Hello' --max_new_tokens 8
```

For a clean CPU development environment:

```sh
python -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test]'
pip install '.[gguf]'  # optional; may compile llama.cpp
python -m pytest -q
```

## Compatibility and evidence

| Path | Backend | Current evidence / limits |
| --- | --- | --- |
| LlamaForCausalLM | Turbo dense | Tiny local checkpoint: greedy prefill + cached decode compared to HF; full dense weights resident |
| Qwen3MoeForCausalLM | Turbo streaming | Tiny individual-expert fixture: prefill and cached decode logits match HF; real large streaming model untested |
| Qwen3_5MoeForConditionalGeneration / Qwen3_5MoeForCausalLM | Turbo streaming | Existing adapter retained; experimental, no end-to-end validation |
| OlmoeForCausalLM | Turbo streaming | Existing adapter retained; experimental, no end-to-end validation |
| GGUF | optional llama-cpp-python | Stories15M Q4_0: actual CLI generation and deterministic comparison with direct llama.cpp; 30B Q2_K_L downloading |

Architecture registration is exact and rejects unknown architectures before
weight loading. Registration is not a claim of complete numerical validation.
Streaming currently requires separately named `mlp.experts.N.gate_proj`,
`up_proj`, `down_proj` tensors and a router tensor. Packed expert layouts are
unsupported. Safetensors storage supports indexed shards and unsharded files.
FP8 dequantization retains the existing 128x128 block implementation; arbitrary
quantization formats are not supported by Turbo streaming.

GGUF is intentionally delegated to llama.cpp, which owns GGUF metadata,
embedded tokenization, tensor-name mapping, architecture support and quantized
kernels. It does **not** use Turbo's SSD/RAM/GPU expert cache or expose routing
collection. A native GGUF reader alone would not supply quantized kernels or
architecture execution, so a native Turbo GGUF path is deferred. See the
[backend API](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/).
Chat requires embedded `tokenizer.chat_template`; ordinary completion does not.
Configure `gguf.n_ctx`, `gguf.n_gpu_layers`, `gguf.n_threads`, `gguf.seed` in YAML; CPU is default.

Configure `execution.device` (`auto`, `cpu`, `cuda`, `mps`) and `execution.dtype`
(`fp32`, `fp16`, `bf16`). CPU defaults to fp32 unless YAML overrides it.
The shipped default YAML explicitly requests fp16. Remote model code is disabled.

## Remaining validation work

Stories15M Q4_0 (19 MB) was downloaded from ggml-org/tiny-llamas and tested.
Qwen3-30B-A3B-Instruct-2507 Q2_K_L (about 11.2 GB) from bartowski is
downloading for a short CPU mmap test. This quant uses Q8_0 for embed/output
weights. No quantization above Q8 is targeted. Large-model performance remains
unverified until that test completes. Local
hardware: 16 GiB RAM and RTX 3050 6 GiB; free root disk was about 3 GiB at start.
Existing README throughput figures are upstream claims, not reproduced results.
Before real model installation, recheck disk space and checkpoint requirements.

Still required for full revival: real large Qwen MoE generation tests,
30B quantized GGUF inference tests, validated dependency combinations beyond
the local stack, stricter packed/FP8 layout preflight and comprehensive memory
budget enforcement. CI currently exercises small CPU fixtures; the GGUF contract
test uses a fake runtime. An optional real GGUF test runs when
`TURBO_TEST_GGUF` is set (or the Stories15M file is present locally).
