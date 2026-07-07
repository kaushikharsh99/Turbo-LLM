import torch

class Sampler:
    """Implements sampling methods: Greedy, Temperature, Top-K, and Top-P (nucleus)."""

    def __init__(self, temperature: float = 0.7, top_p: float = 0.95, top_k: int = 50):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def sample(self, logits: torch.Tensor) -> int:
        """
        Samples a single token from the model logits.
        logits shape: (vocab_size,) or (1, vocab_size) or (1, 1, vocab_size)
        """
        # Squeeze dimensions to get 1D logits tensor
        if logits.dim() == 3:
            logits = logits[0, -1, :]  # shape: (vocab_size,)
        elif logits.dim() == 2:
            logits = logits[-1, :]     # shape: (vocab_size,)

        # Greedy fallback
        if self.temperature <= 0.0:
            return int(torch.argmax(logits).item())

        # 1. Temperature scaling
        logits = logits / self.temperature

        # 2. Top-K filtering
        if self.top_k > 0 and self.top_k < logits.shape[-1]:
            val, _ = torch.topk(logits, self.top_k)
            min_val = val[-1]
            logits[logits < min_val] = float("-inf")

        # 3. Top-P (Nucleus) filtering
        if 0.0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Mask out elements with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Keep at least the first/highest probability token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Map the sorted indices to remove back to original indices
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")

        # 4. Softmax and multinomial sampling
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return int(next_token.item())
