import torch
import time
import os
import resource
import gc
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers.models.qwen3_moe.modeling_qwen3_moe import create_causal_mask

from loader.expert_loader import ExpertLoader
from execution.router import RouterExecutor
from execution.moe import MoEExecutor
from execution.layer_executor import LayerExecutor
from cache.kv_cache import KVCache

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
SNAPSHOT_PATH = "/home/harsh/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507-FP8/snapshots/5a5a776300a41aaa681dd7ff0106608ef2bc90db"

@torch.no_grad()
def generate(model, prompt, max_new_tokens=50, config=None):
    # 1. Initialize Loader
    print(f"Loading weights index from {model}...")
    loader = ExpertLoader(model, config=config)
    
    # 2. Load Config and Meta Model
    hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    
    # Determine dtype from config
    dtype = torch.float16
    if config and "execution" in config and "dtype" in config["execution"]:
        dtype_str = config["execution"]["dtype"]
        if dtype_str in ("bf16", "bfloat16"):
            dtype = torch.bfloat16
        elif dtype_str in ("fp32", "float32"):
            dtype = torch.float32

    with init_empty_weights():
        causal_model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True, torch_dtype=dtype)

    # 3. Initialize Executors
    router_exec = RouterExecutor(loader, hf_config.num_hidden_layers)
    moe_exec = MoEExecutor(loader)
    layer_exec = LayerExecutor(causal_model, loader, router_exec, moe_exec)
    kv_cache = KVCache()

    # 4. Tokenizer and Input
    tokenizer = AutoTokenizer.from_pretrained(model)
    print(f"Prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]
    
    print(f"Generating {max_new_tokens} tokens...")
    
    # Reset CUDA stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    next_token_id = None
    generated_text = ""
    
    for step in range(max_new_tokens):
        step_start_time = time.time()
        
        # Prefill vs Decode setup
        if step == 0:
            input_to_model = input_ids
            position_ids = torch.arange(prompt_len, device="cuda").unsqueeze(0)
        else:
            input_to_model = next_token_id.view(1, 1)
            position_ids = torch.tensor([[prompt_len + step - 1]], device="cuda")
            
        # 5. Embeddings Step
        hidden_states = causal_model.model.embed_tokens(input_to_model)
        
        # 6. Rotary Embeddings & Attention Mask Setup
        position_embeddings = causal_model.model.rotary_emb(hidden_states, position_ids=position_ids)
        
        causal_mask = create_causal_mask(
            config=hf_config,
            inputs_embeds=hidden_states,
            attention_mask=None,
            past_key_values=kv_cache,
            position_ids=position_ids
        )

        # 7. Layer-by-Layer Execution
        for layer_id in range(hf_config.num_hidden_layers):
            hidden_states, _ = layer_exec.execute_layer(
                layer_id=layer_id,
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                kv_cache=kv_cache
            )

        # 8. Final LayerNorm
        hidden_states = causal_model.model.norm(hidden_states)

        # 9. LM Head Projection
        logits = causal_model.lm_head(hidden_states)

        # 10. Argmax & Get Token
        next_token_logits = logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        next_token = tokenizer.decode([next_token_id.item()])
        generated_text += next_token
        
        # Memory tracking
        vram_allocated = torch.cuda.memory_allocated()
        # Assert VRAM < memory limit
        max_vram_bytes = 5.8 * 1024**3
        if config and "memory" in config and "max_vram_mb" in config["memory"]:
            max_vram_bytes = config["memory"]["max_vram_mb"] * 1024**2
        assert vram_allocated < max_vram_bytes, f"VRAM allocation {vram_allocated / 1024**3:.2f} GB exceeds limit!"
        
        # Calculate cache size
        kv_bytes = 0
        for k in kv_cache.keys.values():
            if k is not None:
                kv_bytes += k.element_size() * k.nelement()
        for v in kv_cache.values.values():
            if v is not None:
                kv_bytes += v.element_size() * v.nelement()
        kv_mb = kv_bytes / 1024**2
        
        peak_vram_step = torch.cuda.max_memory_allocated() / 1024**2
        step_duration = time.time() - step_start_time
        
        exp_cache_count = len(loader.expert_cache)
        exp_cache_limit = loader.cache_limit
        print(f"Step {step:02d} | Token: {repr(next_token):<10} (ID: {next_token_id.item():<5}) | "
              f"Peak VRAM: {peak_vram_step:.2f} MB | Cache VRAM: {kv_mb:.2f} MB | "
              f"Expert Cache: {exp_cache_count}/{exp_cache_limit} | Time: {step_duration:.2f}s")
        
        # Clear unused memory
        # gc.collect()
        # torch.cuda.empty_cache()

    duration = time.time() - start_time
    tokens_per_sec = max_new_tokens / duration
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Prompt            : {prompt}")
    print(f"Generated text    : {generated_text}")
    print(f"Generated tokens  : {max_new_tokens}")
    print(f"Total time taken  : {duration:.2f} seconds")
    print(f"Speed (tokens/sec): {tokens_per_sec:.2f} tok/s")
    print(f"Peak VRAM used    : {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print(f"Peak System RAM   : {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f} MB")
    if hasattr(layer_exec, "attn_times") and len(layer_exec.attn_times) > 0:
        avg_attn = sum(layer_exec.attn_times) / len(layer_exec.attn_times)
        avg_moe = sum(layer_exec.moe_times) / len(layer_exec.moe_times)
        print(f"Avg Attention time per layer: {avg_attn*1000:.2f} ms")
        print(f"Avg MoE/MLP time per layer  : {avg_moe*1000:.2f} ms")
    total_hits = loader.gpu_hits + loader.ram_hits + loader.ssd_hits
    if total_hits > 0:
        gpu_pct = (loader.gpu_hits / total_hits) * 100
        ram_pct = (loader.ram_hits / total_hits) * 100
        ssd_pct = (loader.ssd_hits / total_hits) * 100
    else:
        gpu_pct, ram_pct, ssd_pct = 0.0, 0.0, 0.0

    print(f"GPU hits:\n{gpu_pct:.0f}%\n")
    print(f"RAM hits:\n{ram_pct:.0f}%\n")
    print(f"SSD hits:\n{ssd_pct:.0f}%")
    print("=" * 60)

    loader.close()
    return generated_text


@torch.no_grad()
def main():
    print("=" * 60)
    print("PHASE 2.2 — AUTOREGRESSIVE GENERATION WITH KV CACHE")
    print("=" * 60)

    prompt = "Choose any topic and write an informative article of approximately 500 words. Structure it with an introduction, explanation, examples, analysis, and conclusion. Make it engaging, factual, and coherent."
    generate(SNAPSHOT_PATH, prompt, max_new_tokens=500)


if __name__ == "__main__":
    main()
