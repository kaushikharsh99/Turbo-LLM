#include "grouped_gemm.h"

torch::Tensor grouped_gemm_moe(
    torch::Tensor hidden_states,
    std::vector<torch::Tensor> gate_weights,
    std::vector<torch::Tensor> up_weights,
    std::vector<torch::Tensor> down_weights,
    torch::Tensor top_k_weights
) {
    // 1. Stack the weights of the top-k experts into contiguous batched tensors
    auto gate_weights_stacked = torch::stack(gate_weights, 0); // [top_k, intermediate, hidden]
    auto up_weights_stacked = torch::stack(up_weights, 0);     // [top_k, intermediate, hidden]
    auto down_weights_stacked = torch::stack(down_weights, 0); // [top_k, hidden, intermediate]

    int64_t top_k = top_k_weights.size(1);
    int64_t hidden_dim = hidden_states.size(1);

    // 2. Expand the hidden states to batch dimension (zero-copy stride trick)
    auto hidden_expanded = hidden_states.expand({top_k, 1, hidden_dim}); // [top_k, 1, hidden]

    // 3. Batched GEMMs for Gate and Up projections
    auto gate_out = torch::bmm(hidden_expanded, gate_weights_stacked.transpose(1, 2)); // [top_k, 1, intermediate]
    auto up_out = torch::bmm(hidden_expanded, up_weights_stacked.transpose(1, 2));     // [top_k, 1, intermediate]

    // 4. In-place fused SwiGLU activation
    torch::silu_(gate_out);
    gate_out.mul_(up_out); // gate_out now stores the intermediate states: [top_k, 1, intermediate]

    // 5. Batched GEMM for Down projection
    auto down_out = torch::bmm(gate_out, down_weights_stacked.transpose(1, 2)); // [top_k, 1, hidden]

    // 6. Scale each expert output by routing weight and sum
    auto weights_reshaped = top_k_weights.view({top_k, 1, 1});
    auto final_output = (down_out * weights_reshaped).sum(0); // [1, hidden]

    return final_output;
}
