import ctypes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gguf

# 1. Load compiled C++ ctypes library
lib = ctypes.CDLL("./build/libturbo_cuda.so")

lib.launch_dequant_c.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p
]
lib.launch_dequant_c.restype = None

# Helper to dequantize GGUF tensors
def dequantize(tensor):
    qtype = tensor.tensor_type
    shape = tensor.shape
    shape_pt = list(reversed(shape))
    elements = 1
    for s in shape:
        elements *= s
    out = np.zeros(elements, dtype=np.float32)
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    in_arr = np.frombuffer(tensor.data, dtype=np.uint8)
    in_ptr = in_arr.ctypes.data_as(ctypes.c_void_p)
    lib.launch_dequant_c(out_ptr, in_ptr, qtype, elements, None)
    return torch.tensor(out, dtype=torch.float32).reshape(shape_pt)

# 2. Load GGUF model reader
model_path = "/home/harsh/.turbollm/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
reader = gguf.GGUFReader(model_path)
tensors = {t.name: t for t in reader.tensors}

# 3. Setup RMSNorm helper
def rmsnorm(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

# 4. SwiGLU helper
def swiglu(gate, up):
    return F.silu(gate) * up

# 5. Delta rule helper (from modeling_qwen3_5_moe.py)
def l2norm(x, dim=-1, eps=1e-6):
    return x / (x.norm(2, dim=dim, keepdim=True) + eps)

def recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None):
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    
    # transpose to [batch_size, num_heads, seq_len, head_dim]
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous() for x in (query, key, value, beta, g)
    ]
    
    batch_size, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale
    
    core_attn_out = torch.zeros(batch_size, num_heads, seq_len, v_dim, dtype=value.dtype)
    state = torch.zeros(batch_size, num_heads, k_dim, v_dim, dtype=value.dtype) if initial_state is None else initial_state
    
    for i in range(seq_len):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        
    return core_attn_out.transpose(1, 2).contiguous()

# 6. Setup input hidden states
num_tokens = 4
hidden_size = 2048
input_x = torch.zeros(num_tokens, hidden_size)
input_x[0, 0] = 0.98  # Match C++ validation test input initialization!

# Save reference intermediate tensors dictionary
ref_tensors = {}

# Compute Layer 0 (SSM layer)
print("Executing reference Layer 0...", flush=True)

# Load weights for Layer 0
attn_norm = dequantize(tensors["blk.0.attn_norm.weight"])
attn_qkv = dequantize(tensors["blk.0.attn_qkv.weight"])
attn_gate = dequantize(tensors["blk.0.attn_gate.weight"])
ssm_a = dequantize(tensors["blk.0.ssm_a"])
ssm_alpha = dequantize(tensors["blk.0.ssm_alpha.weight"])
ssm_beta = dequantize(tensors["blk.0.ssm_beta.weight"])
ssm_conv1d = dequantize(tensors["blk.0.ssm_conv1d.weight"])
ssm_dt_bias = dequantize(tensors["blk.0.ssm_dt.bias"])
ssm_norm = dequantize(tensors["blk.0.ssm_norm.weight"])
ssm_out = dequantize(tensors["blk.0.ssm_out.weight"])

# RMSNorm 1
norm_out_1 = rmsnorm(input_x, attn_norm)
ref_tensors["blk.0.norm_out_1"] = norm_out_1.numpy()

# SSM Gate
z = F.linear(norm_out_1, attn_gate)
ref_tensors["blk.0.z"] = z.numpy()

# SSM Input Projection
mixed_qkv = F.linear(norm_out_1, attn_qkv) # shape: [num_tokens, 8192]
ref_tensors["blk.0.mixed_qkv"] = mixed_qkv.numpy()

# Split Q, K, V (since we don't have causal conv1d context for multi-token step here, we mock conv1d as identity/silu)
# Actually, the model does: mixed_qkv = F.silu(conv1d(mixed_qkv))
# Let's perform standard Conv1d:
# Shape of mixed_qkv for conv1d: [batch, channels, seq_len] = [1, 8192, num_tokens]
mixed_qkv_conv = mixed_qkv.T.unsqueeze(0) # [1, 8192, 4]
# Perform groups-based 1D convolution (grouped by channels, kernel size = 4)
# Weight shape: [8192, 1, 4]
conv_weight = ssm_conv1d.view(8192, 1, 4)
# Padding = kernel_size - 1 = 3 (causal padding)
conv_padded = F.pad(mixed_qkv_conv, (3, 0))
conv_out = F.conv1d(conv_padded, conv_weight, groups=8192) # [1, 8192, 4]
mixed_qkv_act = F.silu(conv_out).squeeze(0).T # [4, 8192]
ref_tensors["blk.0.mixed_qkv_act"] = mixed_qkv_act.numpy()

query, key, value = torch.split(mixed_qkv_act, [2048, 2048, 4096], dim=-1)
# Reshape query/key/value for DeltaRule: [batch_size, seq_len, num_heads, head_dim]
# head_k_dim = 256, num_k_heads = 8, head_v_dim = 256, num_v_heads = 16
# Wait, key_dim = 2048, value_dim = 4096!
query = query.view(1, num_tokens, 16, 128)
key = key.view(1, num_tokens, 16, 128)
value = value.view(1, num_tokens, 32, 128)

b = F.linear(norm_out_1, ssm_beta) # [num_tokens, 32]
a = F.linear(norm_out_1, ssm_alpha) # [num_tokens, 32]
beta = torch.sigmoid(b).view(1, num_tokens, 32)
# Gated scan scale
g = -torch.exp(ssm_a).view(1, 1, 32) * F.softplus(a.view(1, num_tokens, 32) + ssm_dt_bias)
ref_tensors["blk.0.beta"] = beta.numpy()
ref_tensors["blk.0.g"] = g.numpy()

# Run DeltaRule
# Repeat Q/K to match V head count (16 heads vs 8 heads)
query = query.repeat_interleave(2, dim=2)
key = key.repeat_interleave(2, dim=2)

core_attn_out = recurrent_gated_delta_rule(query, key, value, g, beta) # [1, 4, 16, 256]
core_attn_out = core_attn_out.squeeze(0).view(num_tokens, 4096)
ref_tensors["blk.0.core_attn_out"] = core_attn_out.numpy()

# RMSNormGated
def norm_gated(h, gate_z, norm_weight):
    # h and gate_z: [num_tokens, 32, 128]
    h = h.view(num_tokens, 32, 128)
    gate_z = gate_z.view(num_tokens, 32, 128)
    # apply gate to z
    gated = gate_z * torch.sigmoid(gate_z)
    # RMSNorm on head dim (128)
    variance = h.pow(2).mean(-1, keepdim=True)
    normed = h * torch.rsqrt(variance + 1e-6) * norm_weight.view(1, 1, 128)
    # multiply by gated
    return (normed * gated).view(num_tokens, 4096)

# Apply Gated Norm
ssm_norm_out = norm_gated(core_attn_out, z, ssm_norm)
ref_tensors["blk.0.ssm_norm_out"] = ssm_norm_out.numpy()

# Out Projection
attn_final_out = F.linear(ssm_norm_out, ssm_out)
ref_tensors["blk.0.attn_final_out"] = attn_final_out.numpy()

# Save all ref tensors
np.savez("/tmp/ref_tensors.npz", **ref_tensors)
print("Saved reference tensors to /tmp/ref_tensors.npz", flush=True)

import os
os.makedirs("/tmp/turbo_ref", exist_ok=True)
for name, val in ref_tensors.items():
    with open(f"/tmp/turbo_ref/{name}.bin", "wb") as f:
        f.write(val.tobytes())
print("Saved raw binary reference tensors to /tmp/turbo_ref/", flush=True)
