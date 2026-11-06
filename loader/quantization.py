"""Block FP8 decoding, independent of checkpoint naming and cache placement."""
import torch
import torch.nn.functional as F
from runtime.registry import CompatibilityError


def dequantize_block_fp8(weight, scale, dtype=torch.float16, block_size=128):
    values = weight.to(dtype=dtype)
    if scale is None:
        return values
    if values.ndim != 2 or scale.ndim != 2:
        raise CompatibilityError('Block FP8 requires a matrix and a 2D inverse-scale grid')
    rows, columns = values.shape
    blocks_r = (rows + block_size - 1) // block_size
    blocks_c = (columns + block_size - 1) // block_size
    if tuple(scale.shape) != (blocks_r, blocks_c):
        raise CompatibilityError(f'FP8 scale grid {tuple(scale.shape)} does not match weight {tuple(values.shape)} with {block_size}x{block_size} blocks')
    padded_r, padded_c = blocks_r * block_size, blocks_c * block_size
    if (padded_r, padded_c) != (rows, columns):
        values = F.pad(values, (0, padded_c - columns, 0, padded_r - rows))
    scales = scale.to(device=values.device, dtype=dtype)
    result = (values.view(blocks_r, block_size, blocks_c, block_size)
              * scales.view(blocks_r, 1, blocks_c, 1)).view(padded_r, padded_c)
    return result[:rows, :columns]
