"""Small reproducible CPU inference probe; no implicit downloads."""
import argparse
from contextlib import closing
import json
import resource
import time
from pathlib import Path
from llama_cpp import Llama


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--tokens', type=int, default=24)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    with closing(Llama(model_path=str(args.model), n_ctx=args.context, n_batch=64, n_threads=4,
               n_gpu_layers=0, use_mmap=True, use_mlock=False, seed=0, verbose=False)) as llm:
        loaded = time.perf_counter()
        result = llm.create_completion(prompt='The capital of France is', max_tokens=args.tokens,
                                       temperature=0., top_p=1.)
        finished = time.perf_counter()
        report = {
            'model': args.model.name, 'bytes': args.model.stat().st_size,
            'backend': 'llama_cpp (CPU mmap, not Turbo streaming)',
            'n_ctx': args.context, 'n_threads': 4, 'n_gpu_layers': 0,
            'load_seconds': loaded - started, 'generation_seconds': finished - loaded,
            'usage': result['usage'], 'text': result['choices'][0]['text'],
            'max_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
