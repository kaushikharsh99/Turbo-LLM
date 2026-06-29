import torch
import time
import os
import gc
import psutil
from transformers import AutoTokenizer

class TurboEngine:

    def __init__(self, adapter):
        self.adapter = adapter
        self.loader = adapter.loader
        
        if self.adapter.capabilities["is_moe"]:
            from execution.router import RouterExecutor
            from execution.moe import MoEExecutor
            from execution.layer_executor import LayerExecutor

            self.router = RouterExecutor(self.loader, self.adapter.num_layers)
            self.moe = MoEExecutor(self.loader)
            self.layer = LayerExecutor(self.adapter, self.loader, self.router, self.moe)
        else:
            self.router = None
            self.moe = None
            self.layer = None

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=50,
        config=None,
        chat=False,
        system_prompt=None,
        collector=None,
    ):
        
        self.layer.collector = collector        
        DEVICE = self.loader.DEVICE
        
        model_name_or_path = self.loader.snapshot_path
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        # Apply chat template if --chat mode
        if chat:
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt,
                })
            messages.append({
                "role": "user",
                "content": prompt,
            })
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            print(f"Prompt (chat mode): {prompt}")
        else:
            formatted_prompt = prompt
            print(f"Prompt: {prompt}")

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
        input_ids = inputs.input_ids
        prompt_len = input_ids.shape[1]

        kv_cache = self.adapter.create_cache(
            prompt_len + max_new_tokens
        )
        
        print(f"Generating {max_new_tokens} tokens...")
        
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.time()
        next_token_id = None
        generated_text = ""
        ttft = None
        decode_start = None
        
        for step in range(max_new_tokens):
            step_start_time = time.time()
            
            # Prefill vs Decode setup
            if step == 0:
                input_to_model = input_ids
                position_ids = torch.arange(prompt_len, device=DEVICE).unsqueeze(0)
            else:
                input_to_model = next_token_id.view(1, 1)
                position_ids = torch.tensor([[prompt_len + step - 1]], device=DEVICE)
                
            # 5. Embeddings Step
            hidden_states = self.adapter.embed(input_to_model)
            
            # 6. Rotary Embeddings & Attention Mask Setup
            position_embeddings = self.adapter.rotary_embeddings(hidden_states, position_ids)
            causal_mask = self.adapter.create_attention_mask(
                hidden=hidden_states,
                kv_cache=kv_cache,
                position_ids=position_ids
            )
            if collector is not None:
                collector.begin_token(
                    token_id=-1,
                    token_text="",
                    position=position_ids[0, -1].item() + 1,
                    generation_step=step + 1,
                )
            # 7. Layer-by-Layer Execution
            if self.adapter.capabilities["is_moe"]:
                for layer_id in range(self.adapter.num_layers):
                    hidden_states, _ = self.layer.execute_layer(
                        layer_id=layer_id,
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        kv_cache=kv_cache
                    )
            else:
                for layer_id in range(self.adapter.num_layers):
                    hidden_states, _ = self.adapter.forward_layer(
                        layer_id=layer_id,
                        hidden=hidden_states,
                        kv_cache=kv_cache,
                        position_ids=position_ids,
                        attention_mask=causal_mask
                    )

            # 8. Final LayerNorm
            hidden_states = self.adapter.final_norm(hidden_states)

            # 9. LM Head Projection
            logits = self.adapter.lm_head(hidden_states)

            # 10. Argmax & Get Token
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            next_token = tokenizer.decode([next_token_id.item()])

            generated_text += next_token
            
            if collector is not None:
                collector.current_token["token_id"] = next_token_id.item()
                collector.current_token["token_text"] = next_token
                collector.finish_token()

            # Stop if EOS token is generated
            if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                print("\nEOS token generated. Stopping generation.")
                break
            
            if step == 0:
                ttft = time.time() - start_time
                decode_start = time.time()
            
            # Memory tracking
            if DEVICE == "cuda":
                vram_allocated = torch.cuda.memory_allocated()
            else:
                vram_allocated = 0
                
            max_vram_bytes = 5.8 * 1024**3
            if config and "memory" in config and "max_vram_mb" in config["memory"]:
                max_vram_bytes = config["memory"]["max_vram_mb"] * 1024**2
            assert vram_allocated < max_vram_bytes, f"VRAM allocation {vram_allocated / 1024**3:.2f} GB exceeds limit!"
            
            kv_bytes = kv_cache.get_memory_bytes()
            kv_mb = kv_bytes / 1024**2
            
            peak_vram_step = (
                torch.cuda.max_memory_allocated() / 1024**2
                if DEVICE == "cuda"
                else 0
            )
            step_duration = time.time() - step_start_time
            
            if self.adapter.capabilities["is_moe"]:
                exp_cache_count = len(self.loader.expert_cache)
                exp_cache_limit = self.loader.cache_limit
                print(f"Step {step:02d} | Token: {repr(next_token):<10} (ID: {next_token_id.item():<5}) | "
                      f"Peak VRAM: {peak_vram_step:.2f} MB | Cache VRAM: {kv_mb:.2f} MB | "
                      f"Expert Cache: {exp_cache_count}/{exp_cache_limit} | Time: {step_duration:.2f}s")
            else:
                print(f"Step {step:02d} | Token: {repr(next_token):<10} (ID: {next_token_id.item():<5}) | "
                      f"Peak VRAM: {peak_vram_step:.2f} MB | Cache VRAM: {kv_mb:.2f} MB | Time: {step_duration:.2f}s")

        total_duration = time.time() - start_time

        decode_duration = (
            total_duration - ttft
            if ttft is not None
            else total_duration
        )

        decode_tokens = max(max_new_tokens - 1, 0)

        decode_tokens_per_sec = (
            decode_tokens / decode_duration
            if decode_duration > 0 and decode_tokens > 0
            else 0.0
        )

        overall_tokens_per_sec = (
            max_new_tokens / total_duration
            if total_duration > 0
            else 0.0
        )
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        print(f"Prompt            : {prompt}")
        print(f"Generated text    : {generated_text}")
        print(f"Generated tokens  : {max_new_tokens}")

        print("\nLatency")
        print("-" * 60)
        print(f"Time to First Token : {ttft:.2f} s")

        print("\nDecode")
        print("-" * 60)
        print(f"Decode Tokens       : {decode_tokens}")
        print(f"Decode Time         : {decode_duration:.2f} s")
        print(f"Decode Speed        : {decode_tokens_per_sec:.2f} tok/s")

        print("\nOverall")
        print("-" * 60)
        print(f"Total Runtime       : {total_duration:.2f} s")
        print(f"End-to-End Speed    : {overall_tokens_per_sec:.2f} tok/s")
        
        if DEVICE == "cuda":
            print(f"Peak VRAM used    : {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        else:
            print("Peak VRAM used    : N/A (non-CUDA device)")
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Current System RAM: {ram_mb:.2f} MB")
        
        if self.adapter.capabilities["is_moe"] and hasattr(self.layer, "attn_times") and len(self.layer.attn_times) > 0:
            avg_attn = sum(self.layer.attn_times) / len(self.layer.attn_times)
            avg_moe = sum(self.layer.moe_times) / len(self.layer.moe_times)
            print(f"Avg Attention time per layer: {avg_attn*1000:.2f} ms")
            print(f"Avg MoE/MLP time per layer  : {avg_moe*1000:.2f} ms")
            
        if self.adapter.capabilities["is_moe"]:
            total_hits = self.loader.gpu_hits + self.loader.ram_hits + self.loader.ssd_hits
            if total_hits > 0:
                gpu_pct = (self.loader.gpu_hits / total_hits) * 100
                ram_pct = (self.loader.ram_hits / total_hits) * 100
                ssd_pct = (self.loader.ssd_hits / total_hits) * 100
            else:
                gpu_pct, ram_pct, ssd_pct = 0.0, 0.0, 0.0

            print(f"GPU hits:\n{gpu_pct:.0f}%\n")
            print(f"RAM hits:\n{ram_pct:.0f}%\n")
            print(f"SSD hits:\n{ssd_pct:.0f}%")
            print("=" * 60)

        self.loader.close()
        return generated_text
