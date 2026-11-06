# Historical upstream benchmark claims

These values came from the original README at upstream commit `9fc73e1`. They
have **not** been reproduced by the revival tests and do not describe the new
llama.cpp GGUF backend. Keep them separate from measured reports in `validation/`.

| Platform | Reported model | Reported throughput |
| --- | --- | ---: |
| RTX 3050 Laptop, 6 GB VRAM, 16 GB RAM | Qwen/Qwen3.6-35B-A3B-FP8 | ~2.3 tok/s; ~5.4 GB peak VRAM |
| Apple M4, 16 GB unified memory | Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 | ~0.31 tok/s |
| Intel Ultra 7, 16 GB RAM | Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 | ~0.42 tok/s |

The upstream RTX 3050 evolution table reported:

| Stage | Reported tokens/s |
| --- | ---: |
| Baseline | 0.11 |
| Prefetch + Cache | 0.87 |
| Decode Optimization | 1.14 |
| Static Buffers | 1.33 |
| Active pinning | 1.82 |
| Warm Cache | 1.92 |
| Current | 2.30 |

The upstream artwork remains in `assets/benchmark.png`. No reproducible run
metadata accompanied these figures in the original README. Do not use them as
current acceptance criteria or evidence of verified architecture, quantization
or device support.
