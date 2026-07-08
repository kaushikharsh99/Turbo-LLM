import argparse
import sys
from pathlib import Path
import torch

from loader.model_loader import load_model
from memory.memory_manager import MemoryManager
from runtime.device import DeviceManager
from runtime.executor import Executor
from runtime.chat_executor import ChatExecutor
from runtime.batch_executor import BatchExecutor
from runtime.profiler import Profiler

def main():
    parser = argparse.ArgumentParser(description="Turbo-LLM: High-efficiency layer-by-layer inference engine.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/harsh/.turbollm/models/Qwen_Qwen3.6-35B-A3B-FP8",
        help="Path to the safetensors model weights directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for single generation",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Run in interactive chat mode",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Path to a JSONL file containing prompts for batch generation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for batch inference",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance and execution profiling",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        default=True,
        help="Enable reasoning generation (default)",
    )
    parser.add_argument(
        "--no-think",
        action="store_false",
        dest="think",
        help="Disable reasoning generation",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--cpu_memory_gb",
        type=float,
        default=8.0,
        help="CPU memory limit for weight caching in GB",
    )
    parser.add_argument(
        "--gpu_memory_gb",
        type=float,
        default=2.5,
        help="GPU memory limit for active layer weights in GB",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Target device to run execution on (auto, cuda, cpu, mps)",
    )
    args = parser.parse_args()

    # Load model and initialize MemoryManager
    model = load_model(args.model_path)
    
    device_manager = DeviceManager(args.device)
    print(f"Using device: {device_manager.device}")
    
    max_cpu_bytes = int(args.cpu_memory_gb * 1024**3)
    max_gpu_bytes = int(args.gpu_memory_gb * 1024**3)
    memory_manager = MemoryManager(model, max_cpu_bytes, max_gpu_bytes, device_manager=device_manager)
    
    executor = Executor(model, memory_manager)
    profiler = Profiler(enabled=args.profile, num_experts=model.config.num_experts, device_manager=device_manager)

    # Route execution based on flags
    if args.batch:
        print(f"Running in Batch mode with file: {args.batch} (batch size: {args.batch_size})")
        batch_executor = BatchExecutor(
            executor=executor,
            model_path=args.model_path,
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens,
            think_mode=args.think,
            profiler=profiler
        )
        batch_executor.generate_from_jsonl(args.batch, "output.jsonl")
        print("Batch generation complete. Outputs written to output.jsonl")
        
    elif args.chat or args.prompt:
        chat_executor = ChatExecutor(
            executor=executor,
            model_path=args.model_path,
            think_mode=args.think,
            profiler=profiler
        )
        
        if args.prompt:
            # Single prompt execution
            print(f"Prompt: {args.prompt}")
            sys.stdout.write("Turbo-LLM: ")
            sys.stdout.flush()
            for token_text in chat_executor.generate(args.prompt, max_new_tokens=args.max_tokens):
                sys.stdout.write(token_text)
                sys.stdout.flush()
            print()
        else:
            # Interactive chat loop
            print("Turbo-LLM Interactive Chat Mode")
            print("Type '/exit' or '/quit' to exit.")
            print("-" * 50)
            
            while True:
                try:
                    user_prompt = input("\nUser: ")
                except (KeyboardInterrupt, EOFError):
                    break
                if user_prompt.strip() in ("/exit", "/quit"):
                    break
                    
                sys.stdout.write("Turbo-LLM: ")
                sys.stdout.flush()
                for token_text in chat_executor.generate(user_prompt, max_new_tokens=args.max_tokens):
                    sys.stdout.write(token_text)
                    sys.stdout.flush()
                print()
    else:
        # Fallback to single prompt run with default prompt if no flags specified
        print(f"Running fallback single prompt: Explain relativity.")
        chat_executor = ChatExecutor(
            executor=executor,
            model_path=args.model_path,
            think_mode=args.think,
            profiler=profiler
        )
        sys.stdout.write("Turbo-LLM: ")
        sys.stdout.flush()
        for token_text in chat_executor.generate("Explain relativity in one sentence.", max_new_tokens=args.max_tokens):
            sys.stdout.write(token_text)
            sys.stdout.flush()
        print()

if __name__ == "__main__":
    main()
