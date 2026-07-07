import time
import torch
import resource
from collections import defaultdict
from typing import Dict, List, Any

class Profiler:
    """
    Measures and aggregates performance metrics, layer-by-layer execution times,
    expert activations, and memory footprints.
    """

    def __init__(self, enabled: bool = False, num_experts: int = 256):
        self.enabled = enabled
        self.num_experts = num_experts
        self.start_time = 0.0
        self.end_time = 0.0
        self.prompt_tokens = 0
        self.generated_tokens = 0
        
        # Timing metrics
        self.prefill_time = 0.0
        self.decode_times = []
        
        # Layer timings: layer_id -> metric -> list of times
        self.layer_timings = defaultdict(lambda: defaultdict(float))
        
        # Memory metrics
        self.peak_gpu_bytes = 0
        self.peak_cpu_bytes = 0
        self.ram_to_gpu_bytes = 0
        self.ssd_to_ram_bytes = 0
        self.gpu_cache_hits = 0
        self.gpu_cache_misses = 0
        self.cpu_cache_hits = 0
        self.cpu_cache_misses = 0
        self.tensor_loads = 0
        self.tensor_evictions = 0
        
        # Expert statistics: expert_id -> dict
        self.expert_stats = defaultdict(lambda: {
            "activation_count": 0,
            "tokens_routed": 0,
            "load_count": 0,
            "execution_time": 0.0,
            "batch_sizes": [],
            "cache_hits": 0,
            "cache_misses": 0,
        })
        
        # Batch statistics
        self.step_batch_sizes = []
        self.active_requests_per_step = []
        self.layer_unique_experts = []
        self.expert_gemm_tokens = []

    def record_memory_movement(
        self,
        gpu_cache_hit: bool = False,
        cpu_cache_hit: bool = False,
        ram_to_gpu_bytes: int = 0,
        ssd_to_ram_bytes: int = 0,
        tensor_load: bool = False,
        tensor_eviction: bool = False,
    ):
        if not self.enabled:
            return
        if gpu_cache_hit:
            self.gpu_cache_hits += 1
        else:
            self.gpu_cache_misses += 1
            if cpu_cache_hit:
                self.cpu_cache_hits += 1
            else:
                self.cpu_cache_misses += 1
                
        self.ram_to_gpu_bytes += ram_to_gpu_bytes
        self.ssd_to_ram_bytes += ssd_to_ram_bytes
        
        if tensor_load:
            self.tensor_loads += 1
        if tensor_eviction:
            self.tensor_evictions += 1

    def start(self):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        # Reset max memory tracking in PyTorch
        torch.cuda.reset_peak_memory_stats()

    def stop(self, prompt_tokens: int, generated_tokens: int):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = generated_tokens
        
        # Record peak CPU memory (on Linux ru_maxrss is in KB)
        cpu_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.peak_cpu_bytes = cpu_kb * 1024
        
        # Record peak GPU memory
        self.peak_gpu_bytes = torch.cuda.max_memory_allocated()

    def record_time(self, layer_id: int, metric: str, duration: float):
        if not self.enabled:
            return
        self.layer_timings[layer_id][metric] += duration

    def record_expert_load(self, expert_id: int, is_hit: bool):
        if not self.enabled:
            return
        stats = self.expert_stats[expert_id]
        if is_hit:
            stats["cache_hits"] += 1
        else:
            stats["cache_misses"] += 1
            stats["load_count"] += 1

    def record_expert_execution(self, expert_id: int, tokens_count: int, exec_time: float):
        if not self.enabled:
            return
        stats = self.expert_stats[expert_id]
        stats["activation_count"] += 1
        stats["tokens_routed"] += tokens_count
        stats["execution_time"] += exec_time
        stats["batch_sizes"].append(tokens_count)
        self.expert_gemm_tokens.append(tokens_count)

    def record_layer_active_experts(self, layer_id: int, count: int):
        if not self.enabled:
            return
        self.layer_unique_experts.append(count)

    def record_step_stats(self, batch_size: int, active_reqs: int):
        if not self.enabled:
            return
        self.step_batch_sizes.append(batch_size)
        self.active_requests_per_step.append(active_reqs)

    def print_summary(self, mode: str = "Batch", batch_size_limit: int = 1):
        if not self.enabled:
            return
            
        import json
        import csv

        total_time = self.end_time - self.start_time
        total_tokens = self.prompt_tokens + self.generated_tokens
        
        prompt_speed = self.prompt_tokens / max(self.prefill_time, 1e-6)
        decode_speed = self.generated_tokens / max(sum(self.decode_times), 1e-6)
        overall_throughput = total_tokens / max(total_time, 1e-6)
        
        gpu_gb = self.peak_gpu_bytes / (1024**3)
        cpu_gb = self.peak_cpu_bytes / (1024**3)
        
        avg_unique_experts = sum(self.layer_unique_experts) / max(len(self.layer_unique_experts), 1)
        avg_expert_batch = sum(self.expert_gemm_tokens) / max(len(self.expert_gemm_tokens), 1) if self.expert_gemm_tokens else 0.0
        max_expert_batch = max(self.expert_gemm_tokens) if self.expert_gemm_tokens else 0
        
        total_gpu_hits = self.gpu_cache_hits
        total_gpu_misses = self.gpu_cache_misses
        gpu_hit_rate = total_gpu_hits / max(total_gpu_hits + total_gpu_misses, 1) * 100
        
        total_cpu_hits = self.cpu_cache_hits
        total_cpu_misses = self.cpu_cache_misses
        cpu_hit_rate = total_cpu_hits / max(total_cpu_hits + total_cpu_misses, 1) * 100
        
        # 1. Print Console Summary
        if mode == "Chat":
            print("\n" + "=" * 52)
            print("Turbo-LLM Profile".center(52))
            print("=" * 52)
            print(f"Mode                Chat")
            print(f"Think Mode          Enabled") # Or handled dynamically
            print(f"Batch Size          {batch_size_limit}")
            print("")
            print(f"Prompt Tokens       {self.prompt_tokens}")
            print(f"Generated Tokens    {self.generated_tokens}")
            print(f"Total Tokens        {total_tokens}")
            print("")
            print(f"TTFT                {self.prefill_time:.2f} s")
            print(f"Generation Time     {total_time:.2f} s")
            print("")
            print(f"Prompt Throughput   {prompt_speed:.2f} tok/s")
            print(f"Decode Throughput   {decode_speed:.2f} tok/s")
            print(f"Overall Throughput  {overall_throughput:.2f} tok/s")
            print("")
            print(f"Peak GPU Memory     {gpu_gb:.2f} GB")
            print(f"Peak RAM            {cpu_gb:.2f} GB")
            print(f"Peak KV Cache       {self.peak_gpu_bytes / 1024 / 1024 * 0.02:.2f} MB") # estimate or display
            print("")
            print(f"GPU Cache Hit       {gpu_hit_rate:.1f}%")
            print(f"CPU Cache Hit       {cpu_hit_rate:.1f}%")
            print("=" * 52 + "\n")
        else:
            print("\n" + "=" * 52)
            print("Turbo-LLM Batch Summary".center(52))
            print("=" * 52)
            print(f"Requests                 {len(self.step_batch_sizes) and max(self.active_requests_per_step) or 1}")
            print(f"Completed                {len(self.step_batch_sizes) and max(self.active_requests_per_step) or 1}")
            print("")
            print(f"Batch Size               {batch_size_limit}")
            print("")
            print(f"Prompt Tokens            {self.prompt_tokens:,}")
            print(f"Generated Tokens         {self.generated_tokens:,}")
            print(f"Total Tokens             {total_tokens:,}")
            print("")
            print(f"Generation Time          {total_time:.1f} s")
            print("")
            print(f"Overall Throughput       {overall_throughput:.1f} tok/s")
            print(f"Prompt Throughput        {prompt_speed:.1f} tok/s")
            print(f"Decode Throughput        {decode_speed:.1f} tok/s")
            print("")
            print(f"Average Active Experts   {avg_unique_experts:.1f} / layer")
            print(f"Average Tokens/Expert    {avg_expert_batch:.1f}")
            print(f"Largest Expert Batch     {max_expert_batch} tokens")
            print("")
            print(f"Peak GPU Memory          {gpu_gb:.2f} GB")
            print(f"Peak RAM                 {cpu_gb:.2f} GB")
            print("=" * 52 + "\n")

        # 2. Write generated files
        # profile_summary.json
        summary_data = {
            "mode": mode,
            "requests": len(self.step_batch_sizes) and max(self.active_requests_per_step) or 1,
            "batch_size": batch_size_limit,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "total_tokens": total_tokens,
            "generation_time": total_time,
            "prompt_throughput": prompt_speed,
            "decode_throughput": decode_speed,
            "overall_throughput": overall_throughput,
            "peak_gpu_memory_gb": gpu_gb,
            "peak_cpu_memory_gb": cpu_gb,
            "average_unique_experts": avg_unique_experts,
            "average_expert_batch_size": avg_expert_batch,
            "largest_expert_batch": max_expert_batch,
            "gpu_cache_hit_rate": gpu_hit_rate,
            "cpu_cache_hit_rate": cpu_hit_rate
        }
        with open("profile_summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)

        # layer_profile.csv
        # Timings for each layer (Embedding, Attention, Linear Attention, Router, Expert Dispatcher, Shared Expert, Final RMSNorm, LM Head)
        with open("layer_profile.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer_id", "Embedding", "Attention", "Linear Attention", "Router", "Expert Dispatcher", "Shared Expert", "Final RMSNorm", "LM Head", "Total"])
            
            # Aggregate across layers
            for lid in sorted(self.layer_timings.keys()):
                metrics = self.layer_timings[lid]
                # Total for layer
                total_l = sum(metrics.values())
                writer.writerow([
                    lid,
                    metrics.get("Embedding", 0.0),
                    metrics.get("Attention", 0.0),
                    metrics.get("Linear Attention", 0.0),
                    metrics.get("Router", 0.0),
                    metrics.get("Expert Dispatcher", 0.0),
                    metrics.get("Shared Expert", 0.0),
                    metrics.get("Final RMSNorm", 0.0),
                    metrics.get("LM Head", 0.0),
                    total_l
                ])
                
        # memory_profile.csv
        memory_metrics = [
            ("ssd_to_ram_bytes", self.ssd_to_ram_bytes),
            ("ram_to_gpu_bytes", self.ram_to_gpu_bytes),
            ("gpu_cache_hits", self.gpu_cache_hits),
            ("gpu_cache_misses", self.gpu_cache_misses),
            ("cpu_cache_hits", self.cpu_cache_hits),
            ("cpu_cache_misses", self.cpu_cache_misses),
            ("tensor_loads", self.tensor_loads),
            ("tensor_evictions", self.tensor_evictions),
            ("peak_gpu_memory_bytes", self.peak_gpu_bytes),
            ("peak_cpu_memory_bytes", self.peak_cpu_bytes)
        ]
        with open("memory_profile.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerows(memory_metrics)

        # expert_profile.csv
        with open("expert_profile.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["expert_id", "activation_count", "tokens_routed", "load_count", "execution_time_sec", "avg_batch_size", "max_batch_size", "cache_hits", "cache_misses"])
            for exp_id in range(self.num_experts):
                stats = self.expert_stats.get(exp_id, {
                    "activation_count": 0, "tokens_routed": 0, "load_count": 0, "execution_time": 0.0, "batch_sizes": [], "cache_hits": 0, "cache_misses": 0
                })
                avg_b = sum(stats["batch_sizes"]) / max(len(stats["batch_sizes"]), 1) if stats["batch_sizes"] else 0.0
                max_b = max(stats["batch_sizes"]) if stats["batch_sizes"] else 0
                writer.writerow([
                    exp_id,
                    stats["activation_count"],
                    stats["tokens_routed"],
                    stats["load_count"],
                    stats["execution_time"],
                    avg_b,
                    max_b,
                    stats["cache_hits"],
                    stats["cache_misses"]
                ])

        # generation_profile.json
        gen_profile = {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "ttft_sec": self.prefill_time,
            "decode_times": self.decode_times,
            "avg_decode_time": sum(self.decode_times) / max(len(self.decode_times), 1) if self.decode_times else 0.0,
            "throughput_tok_per_sec": overall_throughput
        }
        with open("generation_profile.json", "w") as f:
            json.dump(gen_profile, f, indent=2)
