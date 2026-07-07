import time
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from attension.kv_cache import KVCache
from sampling.sampler import Sampler

class Request:
    """Represents an active generation request in the scheduler."""
    def __init__(
        self,
        request_id: int,
        prompt: str,
        input_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        metadata: Dict[str, Any]
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.input_ids = input_ids
        self.tokens: List[int] = []
        self.finished = False
        self.prompt_length = len(input_ids)
        self.current_length = len(input_ids)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.metadata = metadata

class BatchScheduler:
    """
    Manages batched inference requests, orchestrating prefill and decode loops
    with dynamic masking and sequence tracking.
    """

    def __init__(
        self,
        executor,
        tokenizer,
        sampler: Sampler,
        max_new_tokens: int = 128,
        think_mode: bool = True,
        profiler: Optional[Any] = None
    ):
        self.executor = executor
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.max_new_tokens = max_new_tokens
        self.think_mode = think_mode
        self.profiler = profiler

    def generate(self, batch_requests: List[Request]) -> List[Request]:
        """
        Runs batched generation for the given requests.

        Prefill is run per-sequence (B=1) to avoid padding corruption in
        linear attention layers (recurrent state is corrupted by pad tokens
        because dt_bias and sigmoid(0) produce non-trivial state updates).

        After prefill, KV caches are merged for batched decode:
        - Full attention K/V: left-padded along seq dim, concatenated along batch dim
        - Linear attention conv/recurrent states: concatenated along batch dim (fixed size)
        """
        B = len(batch_requests)
        if B == 0:
            return batch_requests

        max_prompt_len = max(req.prompt_length for req in batch_requests)

        # ── 1. Per-Sequence Prefill (no padding → no linear attention corruption) ──

        per_seq_caches: List[KVCache] = []
        all_first_logits = []

        start_prefill = 0.0
        if self.profiler and self.profiler.enabled:
            torch.cuda.synchronize()
            start_prefill = time.perf_counter()

        for i, req in enumerate(batch_requests):
            seq_cache = KVCache()
            input_ids = torch.tensor([req.input_ids], dtype=torch.long, device="cuda")  # (1, seq_len)
            position_ids = torch.arange(req.prompt_length, dtype=torch.long, device="cuda").unsqueeze(0)  # (1, seq_len)

            logits = self.executor.forward(
                input_ids,
                seq_cache,
                position_ids,
                attention_mask=None,  # No padding → standard causal mask inside attention
                profiler=self.profiler,
            )

            # Keep only the last token's logits for sampling
            all_first_logits.append(logits[0, -1, :])  # (vocab_size,)
            per_seq_caches.append(seq_cache)

            if self.profiler and self.profiler.enabled:
                self.profiler.prompt_tokens += req.prompt_length

        if self.profiler and self.profiler.enabled:
            torch.cuda.synchronize()
            self.profiler.prefill_time += time.perf_counter() - start_prefill

        # ── 2. Merge KV Caches ──

        merged_cache = KVCache()

        # Collect all layer IDs from the per-sequence caches
        all_k_layers = set()
        all_conv_layers = set()
        all_recurrent_layers = set()
        for sc in per_seq_caches:
            all_k_layers.update(sc.k_caches.keys())
            all_conv_layers.update(sc.conv_states.keys())
            all_recurrent_layers.update(sc.recurrent_states.keys())

        # Merge full-attention KV caches: left-pad shorter sequences along seq dim
        for layer_id in all_k_layers:
            k_list = []
            v_list = []
            for sc in per_seq_caches:
                k = sc.k_caches[layer_id]  # (1, seq_len_i, num_heads, head_dim)
                v = sc.v_caches[layer_id]
                seq_len_i = k.shape[1]
                pad_len = max_prompt_len - seq_len_i
                if pad_len > 0:
                    # Left-pad with zeros along seq dim (dim=1)
                    k = F.pad(k, (0, 0, 0, 0, pad_len, 0))  # pad seq dim on left
                    v = F.pad(v, (0, 0, 0, 0, pad_len, 0))
                k_list.append(k)
                v_list.append(v)
            merged_cache.k_caches[layer_id] = torch.cat(k_list, dim=0)  # (B, max_prompt_len, ...)
            merged_cache.v_caches[layer_id] = torch.cat(v_list, dim=0)

        # Merge linear-attention conv states: fixed size, just concat along batch dim
        for layer_id in all_conv_layers:
            merged_cache.conv_states[layer_id] = torch.cat(
                [sc.conv_states[layer_id] for sc in per_seq_caches], dim=0
            )

        # Merge linear-attention recurrent states: fixed size, just concat along batch dim
        for layer_id in all_recurrent_layers:
            merged_cache.recurrent_states[layer_id] = torch.cat(
                [sc.recurrent_states[layer_id] for sc in per_seq_caches], dim=0
            )

        # Free individual caches
        for sc in per_seq_caches:
            sc.k_caches.clear()
            sc.v_caches.clear()
            sc.conv_states.clear()
            sc.recurrent_states.clear()

        # ── 3. Sample First Token ──

        for i, req in enumerate(batch_requests):
            self.sampler.temperature = req.temperature
            self.sampler.top_p = req.top_p
            tok = self.sampler.sample(all_first_logits[i].unsqueeze(0))
            req.tokens.append(tok)
            req.current_length += 1

        # ── 4. Decode Loop (batched) ──

        step = 0
        while True:
            active_reqs = [req for req in batch_requests if not req.finished]
            if not active_reqs:
                break

            if self.profiler and self.profiler.enabled:
                self.profiler.record_step_stats(batch_size=B, active_reqs=len(active_reqs))

            # Build decode inputs
            input_tokens_list = []
            decode_pos_list = []
            for i, req in enumerate(batch_requests):
                input_tokens_list.append(req.tokens[-1])
                decode_pos_list.append(req.current_length - 1)

            decode_input_ids = torch.tensor(input_tokens_list, dtype=torch.long, device="cuda").unsqueeze(1)  # (B, 1)
            decode_position_ids = torch.tensor(decode_pos_list, dtype=torch.long, device="cuda").unsqueeze(1)  # (B, 1)

            # 4D attention mask for decode: (B, 1, 1, kv_seq_len)
            # KV cache has max_prompt_len (with left-pad zeros) + step+1 generated tokens
            kv_seq_len = max_prompt_len + step + 1
            decode_mask_list = []
            for i, req in enumerate(batch_requests):
                pad_len = max_prompt_len - req.prompt_length
                mask = [0.0] * pad_len + [1.0] * (req.prompt_length + step + 1)
                decode_mask_list.append(mask)

            decode_attention_mask = torch.tensor(decode_mask_list, dtype=torch.bfloat16, device="cuda")
            decode_mask = torch.zeros((B, 1, 1, kv_seq_len), device="cuda", dtype=torch.bfloat16)
            decode_mask = decode_mask.masked_fill(decode_attention_mask.unsqueeze(1).unsqueeze(2) == 0.0, float("-inf"))

            start_decode = 0.0
            if self.profiler and self.profiler.enabled:
                torch.cuda.synchronize()
                start_decode = time.perf_counter()

            logits = self.executor.forward(
                decode_input_ids,
                merged_cache,
                decode_position_ids,
                decode_mask,
                self.profiler,
            )

            if self.profiler and self.profiler.enabled:
                torch.cuda.synchronize()
                self.profiler.decode_times.append(time.perf_counter() - start_decode)

            # Sample next tokens for unfinished sequences
            for i, req in enumerate(batch_requests):
                if req.finished:
                    continue

                self.sampler.temperature = req.temperature
                self.sampler.top_p = req.top_p
                tok = self.sampler.sample(logits[i, -1, :].unsqueeze(0))
                req.tokens.append(tok)
                req.current_length += 1

                if self.profiler and self.profiler.enabled:
                    self.profiler.generated_tokens += 1

                if tok == self.tokenizer.eos_token_id or len(req.tokens) >= req.max_new_tokens:
                    req.finished = True

            step += 1

        merged_cache.clear()
        return batch_requests
