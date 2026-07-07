import torch

class MoEMerge:
    def __init__(self, config):
        self.config = config

    def forward(
        self,
        routed_output: torch.Tensor,
        shared_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sums the outputs of the routed experts and the shared expert.
        routed_output shape: (batch_size, seq_len, hidden_size)
        shared_output shape: (batch_size, seq_len, hidden_size)
        Returns:
            output shape: (batch_size, seq_len, hidden_size)
        """
        return routed_output + shared_output
