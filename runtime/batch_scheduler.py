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
        profiler: Optional[Any] = None,
        k_bits: Optional[int] = None,
        v_bits: Optional[int] = None,
        group_size: int = 32,
        residual_length: int = 32
    ):
        self.executor = executor
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.max_new_tokens = max_new_tokens
        self.think_mode = think_mode
        self.profiler = profiler
        self.device_manager = executor.device_manager
        self.device = self.device_manager.device
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length

    def generate(self, batch_requests: List[Request]) -> List[Request]:
        
        B = len(batch_requests)
        if B == 0:
            return batch_requests

        max_prompt_len = max(req.prompt_length for req in batch_requests)

        merged_cache = KVCache(
            self.device_manager,
            k_bits=self.k_bits,
            v_bits=self.v_bits,
            group_size=self.group_size,
            residual_length=self.residual_length
        )

        input_ids = []
        position_ids = []
        attention_masks = []

        for req in batch_requests:

            pad = max_prompt_len - req.prompt_length

            input_ids.append(
                req.input_ids + [self.tokenizer.pad_token_id] * pad
            )

            position_ids.append(
                list(range(req.prompt_length)) +
                [0] * pad
            )

            attention_masks.append(
                [1] * req.prompt_length +
                [0] * pad
            )

        input_ids = torch.tensor(
            input_ids,
            device=self.device,
            dtype=torch.long,
        )

        position_ids = torch.tensor(
            position_ids,
            device=self.device,
            dtype=torch.long,
        )

        attention_masks = torch.tensor(
            attention_masks,
            device=self.device,
            dtype=torch.bool,
        )

        # ---------------------------------------------------------
        # Build 4D attention mask for batched prefill
        # Shape: (B, 1, L, L)
        # ---------------------------------------------------------

        L = max_prompt_len

        # Standard causal mask
        causal_mask = torch.triu(
            torch.full(
                (L, L),
                float("-inf"),
                device=self.device,
                dtype=torch.bfloat16,
            ),
            diagonal=1,
        )

        # Expand to every batch element
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(
            B,
            1,
            L,
            L,
        ).clone()

        # Padding mask
        padding_mask = (~attention_masks).unsqueeze(1).unsqueeze(2)

        # Mask padded key positions
        attention_mask.masked_fill_(
            padding_mask,
            float("-inf"),
        )

        start_prefill = 0.0

        if self.profiler and self.profiler.enabled:
            self.device_manager.synchronize()
            start_prefill = time.perf_counter()

        logits = self.executor.forward(
            input_ids=input_ids,
            kv_cache=merged_cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
            profiler=self.profiler,
        )

        if self.profiler and self.profiler.enabled:
            self.device_manager.synchronize()
            self.profiler.prefill_time += time.perf_counter() - start_prefill
            self.profiler.prompt_tokens += attention_masks.sum().item()

            
        for i, req in enumerate(batch_requests):

            self.sampler.temperature = req.temperature
            self.sampler.top_p = req.top_p

            last_prompt_token = req.prompt_length - 1

            tok = self.sampler.sample(
                logits[i, last_prompt_token].unsqueeze(0)
            )

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

            decode_input_ids = torch.tensor(input_tokens_list, dtype=torch.long, device=self.device).unsqueeze(1)  # (B, 1)
            decode_position_ids = torch.tensor(decode_pos_list, dtype=torch.long, device=self.device).unsqueeze(1)  # (B, 1)

            # 4D attention mask for decode: (B, 1, 1, kv_seq_len)
            # KV cache has max_prompt_len (with left-pad zeros) + step+1 generated tokens
            kv_seq_len = max_prompt_len + step + 1
            decode_mask_list = []
            for i, req in enumerate(batch_requests):
                pad_len = max_prompt_len - req.prompt_length
                mask = [0.0] * pad_len + [1.0] * (req.prompt_length + step + 1)
                decode_mask_list.append(mask)

            decode_attention_mask = torch.tensor(decode_mask_list, dtype=torch.bfloat16, device=self.device)
            decode_mask = torch.zeros((B, 1, 1, kv_seq_len), device=self.device, dtype=torch.bfloat16)
            decode_mask = decode_mask.masked_fill(decode_attention_mask.unsqueeze(1).unsqueeze(2) == 0.0, float("-inf"))

            start_decode = 0.0
            if self.profiler and self.profiler.enabled:
                self.device_manager.synchronize()
                start_decode = time.perf_counter()

            logits = self.executor.forward(
                decode_input_ids,
                merged_cache,
                decode_position_ids,
                decode_mask,
                self.profiler,
            )

            if self.profiler and self.profiler.enabled:
                self.device_manager.synchronize()
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
