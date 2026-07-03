import torch
import numpy as np
from compute.quantization.cpu.base import CPUDequantizer
from compute.quantization.base import QuantizationType

# Q5_K block layout (QK_K = 256):
#
#   struct block_q5_K {
#       ggml_fp16_t d;              // 2 bytes: Super-block scale
#       ggml_fp16_t dmin;           // 2 bytes: Super-block min
#       uint8_t scales[12];         // 12 bytes: Scales and mins
#       uint8_t qh[32];             // 32 bytes: High bits
#       uint8_t qs[128];            // 128 bytes: Low bits
#   };
#   Total: 2 + 2 + 12 + 32 + 128 = 176 bytes per super-block of 256 elements


def _vectorized_get_scales_mins(scales_raw: np.ndarray) -> tuple:
    """
    Vectorized extraction of all 8 sub-block scales and 8 sub-block mins
    from the 12-byte packed scales array.
    """
    nb = scales_raw.shape[0]
    sc = np.zeros((nb, 8), dtype=np.float32)
    m = np.zeros((nb, 8), dtype=np.float32)
    
    # j = 0..3: sc[j] = scales[j] & 63, m[j] = scales[j+4] & 63
    sc[:, 0] = scales_raw[:, 0] & 63
    sc[:, 1] = scales_raw[:, 1] & 63
    sc[:, 2] = scales_raw[:, 2] & 63
    sc[:, 3] = scales_raw[:, 3] & 63
    m[:, 0] = scales_raw[:, 4] & 63
    m[:, 1] = scales_raw[:, 5] & 63
    m[:, 2] = scales_raw[:, 6] & 63
    m[:, 3] = scales_raw[:, 7] & 63
    
    # j = 4..7: 
    #   sc[j] = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
    #   m[j]  = (scales[j+4] >> 4)  | ((scales[j]   >> 6) << 4)
    sc[:, 4] = (scales_raw[:, 8]  & 0xF) | ((scales_raw[:, 0] >> 6) << 4)
    sc[:, 5] = (scales_raw[:, 9]  & 0xF) | ((scales_raw[:, 1] >> 6) << 4)
    sc[:, 6] = (scales_raw[:, 10] & 0xF) | ((scales_raw[:, 2] >> 6) << 4)
    sc[:, 7] = (scales_raw[:, 11] & 0xF) | ((scales_raw[:, 3] >> 6) << 4)
    m[:, 4]  = (scales_raw[:, 8]  >> 4)  | ((scales_raw[:, 4] >> 6) << 4)
    m[:, 5]  = (scales_raw[:, 9]  >> 4)  | ((scales_raw[:, 5] >> 6) << 4)
    m[:, 6]  = (scales_raw[:, 10] >> 4)  | ((scales_raw[:, 6] >> 6) << 4)
    m[:, 7]  = (scales_raw[:, 11] >> 4)  | ((scales_raw[:, 7] >> 6) << 4)
    
    return sc, m


class Q5KDequantizer(CPUDequantizer):
    """
    CPU reference dequantizer for GGUF Q5_K format.
    Fully vectorized NumPy implementation — no Python loops.
    """
    @property
    def type(self) -> str:
        return QuantizationType.Q5_K

    @property
    def block_size(self) -> int:
        return 256  # QK_K

    @property
    def bytes_per_block(self) -> int:
        return 176  # 2 + 2 + 12 + 32 + 128

    def _dequantize_blocks(
        self,
        raw_tensor: torch.Tensor,
        out: torch.Tensor,
        num_blocks: int,
        dtype: torch.dtype
    ):
        """
        Vectorized dequantization of Q5_K blocks.
        """
        raw = raw_tensor.numpy().view(np.uint8)
        
        # Reshape into [num_blocks, 176]
        blocks = raw[:num_blocks * 176].reshape(num_blocks, 176)
        
        # Extract fields
        d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
        dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1)
        scales_raw = blocks[:, 4:16]
        qh = blocks[:, 16:48]
        qs = blocks[:, 48:176]
        
        # Unpack scales/mins
        sc, m = _vectorized_get_scales_mins(scales_raw)
        
        # Allocate output [num_blocks, 256]
        result = np.empty((num_blocks, 256), dtype=np.float32)
        
        qh_cast = qh.astype(np.int32)
        
        # Process 4 steps of 64 elements (each step does 2 sub-blocks of 32)
        for s in range(4):
            ql_step = qs[:, s * 32 : (s + 1) * 32].astype(np.int32)
            
            # Extract high bits (bit 2*s and 2*s + 1)
            high1 = np.bitwise_and(np.right_shift(qh_cast, 2 * s), 1) * 16
            high2 = np.bitwise_and(np.right_shift(qh_cast, 2 * s + 1), 1) * 16
            
            # Combine low and high bits
            val1 = np.bitwise_and(ql_step, 0xF) + high1
            val2 = np.right_shift(ql_step, 4) + high2
            
            # Scale and min multipliers
            sc1 = sc[:, 2 * s]
            m1 = m[:, 2 * s]
            d_sc1 = d * sc1
            min_m1 = dmin * m1
            
            sc2 = sc[:, 2 * s + 1]
            m2 = m[:, 2 * s + 1]
            d_sc2 = d * sc2
            min_m2 = dmin * m2
            
            # Compute dequantized values
            out_offset = s * 64
            result[:, out_offset : out_offset + 32] = d_sc1[:, None] * val1.astype(np.float32) - min_m1[:, None]
            result[:, out_offset + 32 : out_offset + 64] = d_sc2[:, None] * val2.astype(np.float32) - min_m2[:, None]
            
        out.copy_(torch.from_numpy(result.reshape(-1)).to(dtype=dtype))
