import sys
import os
import argparse
import time
import torch
import torch.nn.functional as F

# Make sure we can load the project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug.profiler import Profiler
from debug.events import EventType
from debug.timers import CPUTimer, CUDATimer
from runtime.engine import TurboEngine
from execution.layer_executor import LayerExecutor
from execution.router import RouterExecutor
from execution.moe import MoEExecutor
from loader.expert_loader import ExpertLoader
from debug.exporters.json_export import export_json
from debug.exporters.csv_export import export_csvs
from debug.exporters.trace_export import export_chrome_trace
from debug.exporters.summary import export_summary

try:
    import turbollm_cpp
except ImportError:
    turbollm_cpp = None

def main():
    parser = argparse.ArgumentParser(description="Turbo-LLM Developer Profiling Tool")
    parser.add_argument("--model", type=str, required=True, help="Model name or directory")
    parser.add_argument("--prompt", type=str, default="Explain how transformers work.", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    parser.add_argument("--chat", action="store_true", help="Use chat template formatting")
    parser.add_argument("--thinking", choices=["on", "off"], default="off", help="Qwen thinking mode")
    parser.add_argument("--system", type=str, default=None, help="System prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Inference temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling threshold")
    args = parser.parse_args()

    profiler = Profiler.get_instance()
    profiler.reset()

    from config.config import load_config
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_config_path = os.path.join(package_root, "config", "default.yaml")
    cfg = load_config(default_config_path)

    # 1. Resolve model path
    model_arg = args.model if args.model is not None else cfg["model"]["path"]
    from loader.model_manager import get_model
    model_path = get_model(model_arg)
    
    # Load adapter config and models
    from runtime.model_factory import create_adapter
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights

    # 2. Patch initialization
    orig_load_weight = ExpertLoader.load_weight
    def load_weight_patched(self, weight_name, device="cuda", dtype=None):
        with CPUTimer(f"Load base weight {weight_name}", profiler, EventType.LOAD):
            return orig_load_weight(self, weight_name, device, dtype)
    ExpertLoader.load_weight = load_weight_patched

    print(f"Loading weights index from {model_path}...")
    loader = ExpertLoader(model_path, config=cfg)
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.float16
    if cfg and "execution" in cfg and "dtype" in cfg["execution"]:
        dtype_str = cfg["execution"]["dtype"]
        if dtype_str in ("bf16", "bfloat16"):
            dtype = torch.bfloat16
        elif dtype_str in ("fp32", "float32"):
            dtype = torch.float32

    with init_empty_weights():
        causal_model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True, torch_dtype=dtype)
    adapter = create_adapter(causal_model, loader, hf_config)

    # Initialize Engine
    with CPUTimer("Engine Initialization", profiler, EventType.GLOBAL):
        engine = TurboEngine(adapter)

    # 3. Apply telemetry wraps on the initialized engine
    loaded_this_layer = []
    active_experts_layer = []
    current_step = [0]

    # Patch ExpertLoader.load_expert_raw to capture SSD vs RAM loading times and sources
    orig_load_expert_raw = engine.loader.load_expert_raw
    def load_expert_raw_patched(layer_id, expert_id, is_decode=True):
        key = (layer_id, expert_id)
        with engine.loader.lock:
            in_ram = engine.loader.ram_cache.get(key) is not None
        source = "RAM" if in_ram else "SSD"
        
        loaded_this_layer.append((expert_id, source))
        
        with CUDATimer(f"GPU Copy ({source})", profiler, EventType.GPU_COPY, layer_id=layer_id):
            with CPUTimer(f"Expert Load ({source})", profiler, EventType.LOAD, layer_id=layer_id):
                res = orig_load_expert_raw(layer_id, expert_id, is_decode)
        return res
    engine.loader.load_expert_raw = load_expert_raw_patched

    # Patch C++ execute_moe_with_cache (if compiled) to capture grouped GEMM times
    if turbollm_cpp is not None and hasattr(turbollm_cpp, "execute_moe_with_cache"):
        orig_execute_moe_with_cache = turbollm_cpp.execute_moe_with_cache
        def execute_moe_with_cache_patched(layer_id, expert_ids, hidden_states, top_k_weights, loader):
            profiler.increment_crossing("py_to_cpp")
            t_start = time.perf_counter()
            with CUDATimer("Grouped GEMM", profiler, EventType.GEMM, layer_id=layer_id):
                res = orig_execute_moe_with_cache(layer_id, expert_ids, hidden_states, top_k_weights, loader)
            duration_ms = (time.perf_counter() - t_start) * 1000.0
            profiler.record_event("C++ Execution (Dequant + GEMM)", EventType.OVERHEAD, duration_ms, layer_id=layer_id)
            return res
        turbollm_cpp.execute_moe_with_cache = execute_moe_with_cache_patched

    # Patch LayerExecutor.execute_layer
    orig_execute_layer = engine.layer.execute_layer
    def execute_layer_patched(layer_id, hidden_states, attention_mask, position_ids, position_embeddings, kv_cache=None):
        if layer_id == 0:
            profiler.set_token(current_step[0])
            current_step[0] += 1
            
        profiler.set_layer(layer_id)
        loaded_this_layer.clear()
        
        profiler.record_memory(f"Before Layer {layer_id}")
        
        with CUDATimer(f"Layer {layer_id}", profiler, EventType.LAYER):
            # Time attention sub-op
            t_attn_start = time.perf_counter()
            residual = hidden_states
            layer_module = engine.layer.adapter.layers()[layer_id]
            normed_hidden = engine.layer.layer_layout.input_norm(layer_module)(hidden_states)
            
            if hasattr(layer_module, "linear_attn"):
                attn_output = layer_module.linear_attn(
                    hidden_states=normed_hidden,
                    cache_params=kv_cache,
                    attention_mask=attention_mask,
                )
            else:
                attn_output, _ = layer_module.self_attn(
                    hidden_states=normed_hidden,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=kv_cache,
                )
            hidden_states = residual + attn_output
            attn_duration_ms = (time.perf_counter() - t_attn_start) * 1000.0
            profiler.record_event("Attention", EventType.ATTENTION, attn_duration_ms, layer_id=layer_id)

            # Time MoE execution
            t_moe_start = time.perf_counter()
            residual = hidden_states
            normed_attn = engine.layer.layer_layout.post_norm(layer_module)(hidden_states)
            
            top_k_indices, top_k_weights = engine.layer.router_exec.compute_routing(layer_id, normed_attn)
            
            original_shape = normed_attn.shape
            hidden_flat = normed_attn.view(-1, original_shape[-1])
            
            moe_output = engine.layer.moe_exec.execute_layer(layer_id, hidden_flat, top_k_indices, top_k_weights)
            moe_output = moe_output.view(original_shape)
            hidden_states = residual + moe_output
            
            moe_duration_ms = (time.perf_counter() - t_moe_start) * 1000.0
            profiler.record_event("MoE / MLP", EventType.GEMM, moe_duration_ms, layer_id=layer_id)

        profiler.record_memory(f"After Layer {layer_id}")
        return hidden_states, None
        
    engine.layer.execute_layer = execute_layer_patched

    # Patch compute_routing to log Router time
    orig_compute_routing = engine.layer.router_exec.compute_routing
    def compute_routing_patched(layer_id, hidden):
        with CUDATimer("Router", profiler, EventType.ROUTER, layer_id=layer_id):
            indices, weights = orig_compute_routing(layer_id, hidden)
        
        active_experts_layer.clear()
        active_experts_layer.extend(indices[-1].tolist())
        return indices, weights
    engine.layer.router_exec.compute_routing = compute_routing_patched

    # Patch execute_decode to log Shared Expert and capture hits
    orig_execute_decode = engine.moe.execute_decode
    def execute_decode_patched(layer_id, hidden_states, top_k_indices, top_k_weights):
        with CUDATimer("MoE Decode Loop", profiler, EventType.GEMM, layer_id=layer_id):
            res = orig_execute_decode(layer_id, hidden_states, top_k_indices, top_k_weights)
            
        # Determine GPU/RAM/SSD hit states
        loaded_ids = [item[0] for item in loaded_this_layer]
        for exp_id in active_experts_layer:
            if exp_id in loaded_ids:
                source = next(item[1] for item in loaded_this_layer if item[0] == exp_id)
            else:
                source = "GPU"
            profiler.record_cache_lookup(layer_id, exp_id, source)
        return res
    engine.moe.execute_decode = execute_decode_patched

    # Patch final norm & head projection
    orig_final_norm = engine.adapter.final_norm
    def final_norm_patched(hidden):
        with CUDATimer("Norm", profiler, EventType.NORM):
            return orig_final_norm(hidden)
    engine.adapter.final_norm = final_norm_patched

    orig_lm_head = engine.adapter.lm_head
    def lm_head_patched(hidden):
        with CUDATimer("LM Head", profiler, EventType.LM_HEAD):
            return orig_lm_head(hidden)
    engine.adapter.lm_head = lm_head_patched

    # Instantiate tokenizer locally for statistics
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(engine.loader.snapshot_path, trust_remote_code=True)

    # Patch global engine generate loop
    orig_generate = engine.generate
    def generate_patched(*g_args, **g_kwargs):
        prompt = g_args[0] if len(g_args) > 0 else g_kwargs.get("prompt", "")
        profiler.model_info["model"] = args.model
        profiler.model_info["prompt"] = prompt
        
        inputs = tokenizer(prompt, return_tensors="pt")
        profiler.model_info["prompt_len"] = inputs.input_ids.shape[1]
        
        t_start_global = time.perf_counter()
        
        # We profile each token generation step
        res_text = orig_generate(*g_args, **g_kwargs)
        
        total_duration = time.perf_counter() - t_start_global
        profiler.model_info["total_duration"] = total_duration
        
        generated_token_ids = tokenizer(res_text).input_ids
        n_toks = len(generated_token_ids)
        profiler.model_info["generated_tokens"] = n_toks
        profiler.model_info["decode_tokens"] = max(0, n_toks - 1)
        denom = max(1, n_toks)
        profiler.model_info["decode_duration"] = total_duration - (total_duration / denom)
        profiler.model_info["ttft"] = total_duration / denom
        
        profiler.resolve_cuda_timers()
        return res_text
        
    engine.generate = generate_patched

    # Run generation!
    print("Starting profiled inference...")
    res = engine.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        config=cfg,
        chat=args.chat,
        thinking=args.thinking,
        system_prompt=args.system
    )
    profiler.loader = engine.loader
    print("\nGeneration finished! Exporting profile reports...")

    # Write files under debug/ and debug/output/
    debug_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(debug_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Export summary.txt
    summary_text = export_summary(profiler, os.path.join(debug_dir, "summary.txt"))
    export_summary(profiler, os.path.join(output_dir, "summary.txt"))
    
    # 2. Export Heat Map Report (Milestone 2.5)
    from debug.exporters.heat_map import export_expert_heat_map
    heat_map_text = export_expert_heat_map(profiler, os.path.join(debug_dir, "heat_map.txt"))
    export_expert_heat_map(profiler, os.path.join(output_dir, "heat_map.txt"))

    # 3. Export profile.json & trace.json
    export_json(profiler, os.path.join(debug_dir, "profile.json"))
    export_json(profiler, os.path.join(output_dir, "profile.json"))
    
    export_chrome_trace(profiler, os.path.join(debug_dir, "trace.json"))
    export_chrome_trace(profiler, os.path.join(output_dir, "trace.json"))
    
    # 4. Export CSV files
    export_csvs(profiler, debug_dir)
    export_csvs(profiler, output_dir)

    print("\n" + heat_map_text)
    
    # Display the final summary profile in the console
    print("\n" + summary_text)

if __name__ == "__main__":
    main()
