"""Reproducible bounded inference probe; no implicit downloads."""
import argparse
from contextlib import closing
from importlib.metadata import version
import json
import os
import platform
import psutil
import resource
import subprocess
import threading
import time
from pathlib import Path
from llama_cpp import Llama, llama_cpp


def gpu_used_mib():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3, check=True)
        return sum(float(line) for line in result.stdout.splitlines())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--tokens', type=int, default=24)
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--prompt', default='The capital of France is')
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--note', default='')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stop = threading.Event()
    gpu_samples = []
    def monitor():
        while not stop.is_set():
            sample = gpu_used_mib()
            if sample is not None:
                gpu_samples.append(sample)
            stop.wait(0.5)
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    available_ram_before = psutil.virtual_memory().available
    started = time.perf_counter()
    try:
        with closing(Llama(model_path=str(args.model), n_ctx=args.context, n_batch=64,
                           n_threads=args.threads, n_gpu_layers=args.gpu_layers,
                           use_mmap=True, use_mlock=False, seed=0, verbose=False)) as llm:
            loaded = time.perf_counter()
            kwargs = dict(max_tokens=args.tokens, temperature=0., top_p=1.)
            if args.chat:
                result = llm.create_chat_completion(messages=[{'role': 'user', 'content': args.prompt}], **kwargs)
                text = result['choices'][0]['message']['content']
            else:
                result = llm.create_completion(prompt=args.prompt, **kwargs)
                text = result['choices'][0]['text']
            finished = time.perf_counter()
            report = {
                'model': args.model.name, 'bytes': args.model.stat().st_size,
                'backend': 'llama_cpp (mmap, not Turbo streaming)',
                'llama_cpp_version': version('llama-cpp-python'),
                'gpu_offload_supported': bool(llama_cpp.llama_supports_gpu_offload()),
                'python': platform.python_version(), 'logical_cpus': os.cpu_count(),
                'available_ram_before_mib': available_ram_before / 1024**2,
                'available_ram_after_mib': psutil.virtual_memory().available / 1024**2,
                'n_ctx_requested': args.context, 'n_ctx_actual': llm.n_ctx(),
                'n_threads': args.threads, 'n_gpu_layers': args.gpu_layers,
                'prompt': args.prompt, 'chat': args.chat, 'note': args.note,
                'load_seconds': loaded - started, 'generation_seconds': finished - loaded,
                'completion_tokens_per_second_including_prefill': result['usage']['completion_tokens'] / (finished - loaded),
                'usage': result['usage'], 'text': text,
                'max_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                'system_gpu_peak_used_mib': max(gpu_samples) if gpu_samples else None,
                'gpu_measurement_note': 'Whole GPU including desktop/other processes; sampled every 0.5s',
            }
    finally:
        stop.set()
        worker.join(timeout=4)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
