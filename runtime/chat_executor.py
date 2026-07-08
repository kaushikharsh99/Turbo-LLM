import time
import torch
from transformers import AutoTokenizer
from runtime.executor import Executor
from sampling.sampler import Sampler
from attension.kv_cache import KVCache
from typing import Generator, Optional, Any

class ChatExecutor:
    """Wrapper around Executor to support autoregressive chat generation (streaming)."""

    def __init__(
        self,
        executor: Executor,
        model_path: str,
        think_mode: bool = True,
        profiler: Optional[Any] = None
    ):
        self.executor = executor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.think_mode = think_mode
        self.profiler = profiler
        self.device_manager = executor.device_manager
        self.device = self.device_manager.device
        
        self.sampler = Sampler(
            temperature=0.0,
            top_p=0.95,
            top_k=50,
        )

    def generate(self, prompt: str, max_new_tokens: int = 128) -> Generator[str, None, None]:
        """
        Runs the prefill phase and then streams generated tokens autoregressively.
        """
        if self.profiler and self.profiler.enabled:
            self.profiler.start()
            self.profiler.prompt_tokens = 0
            self.profiler.generated_tokens = 0
            self.profiler.decode_times = []

        # 1. Format user prompt using tokenizer's chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.think_mode,
        )
        
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        seq_len = input_ids.shape[1]

        # Initialize KV Cache
        kv_cache = KVCache(self.device_manager)

        # 2. Prefill Phase
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
        
        start_prefill = time.perf_counter()
        logits = self.executor.forward(
            input_ids,
            kv_cache,
            position_ids,
            attention_mask=None,
            profiler=self.profiler
        )
        
        if self.profiler and self.profiler.enabled:
            self.device_manager.synchronize()
            self.profiler.prefill_time = time.perf_counter() - start_prefill
            self.profiler.prompt_tokens = seq_len
            self.profiler.record_step_stats(batch_size=1, active_reqs=1)

        # Sample the first new token
        next_token = self.sampler.sample(logits)
        yield self.tokenizer.decode([next_token])
        if self.profiler and self.profiler.enabled:
            self.profiler.generated_tokens += 1

        # 3. Autoregressive Generation Phase
        current_len = seq_len
        input_token = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens - 1):
            # Check for EOS token
            if next_token in (self.tokenizer.eos_token_id, getattr(self.tokenizer, "pad_token_id", None)):
                break

            position_ids = torch.tensor([[current_len]], dtype=torch.long, device=self.device)
            
            start_decode = time.perf_counter()
            logits = self.executor.forward(
                input_token,
                kv_cache,
                position_ids,
                attention_mask=None,
                profiler=self.profiler
            )
            
            if self.profiler and self.profiler.enabled:
                self.device_manager.synchronize()
                self.profiler.decode_times.append(time.perf_counter() - start_decode)

            next_token = self.sampler.sample(logits)
            yield self.tokenizer.decode([next_token])
            
            if self.profiler and self.profiler.enabled:
                self.profiler.generated_tokens += 1
                self.profiler.record_step_stats(batch_size=1, active_reqs=1)

            input_token = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            current_len += 1

        kv_cache.clear()
        
        if self.profiler and self.profiler.enabled:
            self.profiler.stop(self.profiler.prompt_tokens, self.profiler.generated_tokens)
            self.profiler.print_summary(mode="Chat", batch_size_limit=1)
