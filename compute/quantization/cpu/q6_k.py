import torch
import numpy as np
from compute.quantization.cpu.base import CPUDequantizer
from compute.quantization.base import QuantizationType

# Q6_K block layout (QK_K = 256):
#
#   struct block_q6_K {
#       uint8_t ql[128];            // 128 bytes: Quantized values, lower 4 bits
#       uint8_t qh[64];             // 64 bytes: Quantized values, upper 2 bits
#       int8_t  scales[16];         // 16 bytes: Scales, quantized with 8 bits
#       ggml_fp16_t d;              // 2 bytes: Super-block scale
#   };
#   Total: 128 + 64 + 16 + 2 = 210 bytes per super-block of 256 elements


class Q6KDequantizer(CPUDequantizer):
    """
    CPU reference dequantizer for GGUF Q6_K format.
    Fully vectorized NumPy implementation — no Python loops.
    """
    @property
    def type(self) -> str:
        return QuantizationType.Q6_K

    @property
    def block_size(self) -> int:
        return 256  # QK_K

    @property
    def bytes_per_block(self) -> int:
        return 210  # 128 + 64 + 16 + 2

    def _dequantize_blocks(
        self,
        raw_tensor: torch.Tensor,
        out: torch.Tensor,
        num_blocks: int,
        dtype: torch.dtype
    ):
        """
        Vectorized dequantization of Q6_K blocks.
        Port of dequantize_row_q6_K() from ggml-quants.c.
        """
        raw = raw_tensor.numpy().view(np.uint8)
        
        # Reshape into [num_blocks, 210]
        blocks = raw[:num_blocks * 210].reshape(num_blocks, 210)
        
        # Extract fields
        # ql: bytes [0:128] -> [num_blocks, 128]
        ql = blocks[:, 0:128]
        
        # qh: bytes [128:192] -> [num_blocks, 64]
        qh = blocks[:, 128:192]
        
        # scales: bytes [192:208] signed int8 -> [num_blocks, 16]
        scales = blocks[:, 192:208].copy().view(np.int8).astype(np.float32)
        
        # d: bytes [208:210] float16 -> [num_blocks]
        d = blocks[:, 208:210].copy().view(np.float16).astype(np.float32).reshape(-1)
        
        # Allocate output [num_blocks, 256]
        result = np.empty((num_blocks, 256), dtype=np.float32)
        
        # Index array for scale selection: length 32
        is_idx = np.array([0]*16 + [1]*16, dtype=np.int32)
        d_col = d[:, None]
        
        # Process 2 groups of 128 elements (g=0, 1)
        for g in range(2):
            ql_g = ql[:, g * 64 : (g + 1) * 64]
            qh_g = qh[:, g * 32 : (g + 1) * 32]
            sc_g = scales[:, g * 8 : (g + 1) * 8]
            
            ql_g_first = ql_g[:, :32].astype(np.int32)
            ql_g_second = ql_g[:, 32:].astype(np.int32)
            qh_g_cast = qh_g.astype(np.int32)
            
            # Extract 6-bit values and subtract 32 offset
            q1 = np.bitwise_or(np.bitwise_and(ql_g_first, 0xF), np.left_shift(np.bitwise_and(np.right_shift(qh_g_cast, 0), 3), 4)) - 32
            q2 = np.bitwise_or(np.bitwise_and(ql_g_second, 0xF), np.left_shift(np.bitwise_and(np.right_shift(qh_g_cast, 2), 3), 4)) - 32
            q3 = np.bitwise_or(np.right_shift(ql_g_first, 4), np.left_shift(np.bitwise_and(np.right_shift(qh_g_cast, 4), 3), 4)) - 32
            q4 = np.bitwise_or(np.right_shift(ql_g_second, 4), np.left_shift(np.bitwise_and(np.right_shift(qh_g_cast, 6), 3), 4)) - 32
            
            # Compute scales for each subgroup of 32
            s1 = d_col * sc_g[:, is_idx + 0]
            s2 = d_col * sc_g[:, is_idx + 2]
            s3 = d_col * sc_g[:, is_idx + 4]
            s4 = d_col * sc_g[:, is_idx + 6]
            
            # Write to output
            out_offset = g * 128
            result[:, out_offset + 0  : out_offset + 32]  = s1 * q1.astype(np.float32)
            result[:, out_offset + 32 : out_offset + 64]  = s2 * q2.astype(np.float32)
            result[:, out_offset + 64 : out_offset + 96]  = s3 * q3.astype(np.float32)
            result[:, out_offset + 96 : out_offset + 128] = s4 * q4.astype(np.float32)
            
        out.copy_(torch.from_numpy(result.reshape(-1)).to(dtype=dtype))
