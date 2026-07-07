import json
import time
import torch
from transformers import AutoTokenizer
from runtime.executor import Executor
from sampling.sampler import Sampler
from runtime.batch_scheduler import BatchScheduler, Request
from typing import List, Dict, Any, Optional

class BatchExecutor:
    """A batch executor that executes generation for batches of prompts in parallel using BatchScheduler."""

    def __init__(
        self,
        executor: Executor,
        model_path: str,
        batch_size: int = 1,
        max_new_tokens: int = 128,
        think_mode: bool = True,
        profiler: Optional[Any] = None
    ):
        self.executor = executor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.think_mode = think_mode
        self.profiler = profiler

        # Batch greedy decoding by default (temperature=0.0)
        self.sampler = Sampler(
            temperature=0.0,
            top_p=0.95,
            top_k=50,
        )

        self.scheduler = BatchScheduler(
            executor=self.executor,
            tokenizer=self.tokenizer,
            sampler=self.sampler,
            max_new_tokens=max_new_tokens,
            think_mode=think_mode,
            profiler=profiler
        )

    def generate_from_jsonl(self, input_jsonl_path: str, output_jsonl_path: str = "output.jsonl") -> List[Dict[str, Any]]:
        """
        Reads input prompts from a JSONL file, batches them, executes generation,
        and writes results to output JSONL.
        """
        # Load JSONL inputs
        requests_data = []
        with open(input_jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    requests_data.append(json.loads(line))

        if self.profiler and self.profiler.enabled:
            self.profiler.start()
            self.profiler.prompt_tokens = 0
            self.profiler.generated_tokens = 0
            self.profiler.decode_times = []

        # Construct requests
        requests: List[Request] = []
        for req_data in requests_data:
            prompt = req_data.get("prompt", "")
            req_id = req_data.get("id", 0)

            # Check for request-specific think_mode override
            req_think = req_data.get("think", self.think_mode)

            # Apply chat template
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=req_think,
            )

            input_ids = self.tokenizer.encode(formatted_prompt)

            max_tokens = req_data.get("max_tokens", self.max_new_tokens)
            temp = req_data.get("temperature", 0.0)
            top_p = req_data.get("top_p", 0.95)

            requests.append(Request(
                request_id=req_id,
                prompt=prompt,
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=top_p,
                metadata=req_data
            ))

        # Run batch execution
        results = []
        num_requests = len(requests)
        
        for i in range(0, num_requests, self.batch_size):
            batch = requests[i : i + self.batch_size]
            
            start_batch = time.perf_counter()
            self.scheduler.generate(batch)
            batch_time = time.perf_counter() - start_batch

            # Write outputs for batch
            for req in batch:
                response_text = self.tokenizer.decode(req.tokens, skip_special_tokens=True)
                
                # Preserve original metadata fields
                out_data = dict(req.metadata)
                out_data["response"] = response_text

                if self.profiler and self.profiler.enabled:
                    out_data["tokens"] = len(req.tokens)
                    out_data["time"] = round(batch_time, 2)
                    out_data["tok_per_sec"] = round(len(req.tokens) / max(batch_time, 1e-6), 2)
                
                results.append(out_data)

        # Write output JSONL
        with open(output_jsonl_path, "w") as f:
            for out_data in results:
                f.write(json.dumps(out_data) + "\n")

        if self.profiler and self.profiler.enabled:
            self.profiler.stop(self.profiler.prompt_tokens, self.profiler.generated_tokens)
            self.profiler.print_summary(mode="Batch", batch_size_limit=self.batch_size)

        return results
