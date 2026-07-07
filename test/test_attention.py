import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from loader.model_loader import load_model
from memory.memory_manager import MemoryManager
from attension.attension import Attention
from attension.kv_cache import KVCache

MODEL_DIR = "/home/harsh/.turbollm/models/Qwen_Qwen3.6-35B-A3B-FP8"

print("Loading model...")
model = load_model(MODEL_DIR)

print("\nInitializing MemoryManager and Attention Subsystem...")
# CPU RAM limit: 3 GB, GPU VRAM limit: 1 GB
manager = MemoryManager(model, 3 * 1024**3, 1024**3)
attention = Attention(model.config)
kv_cache = KVCache()

# We test with Layer 3 (which has full attention)
layer = model.layers[3]

# 1. Load weights for Layer 3 attention to GPU
print("\nLoading Layer 3 attention weights to GPU...")
manager.move_to_gpu(layer.attention_norm.name)
manager.move_to_gpu(layer.attention.q_proj.name)
manager.move_to_gpu(layer.attention.k_proj.name)
manager.move_to_gpu(layer.attention.v_proj.name)
manager.move_to_gpu(layer.attention.o_proj.name)
if layer.attention.q_scale:
    manager.move_to_gpu(layer.attention.q_scale.name)
if layer.attention.k_scale:
    manager.move_to_gpu(layer.attention.k_scale.name)
if layer.attention.v_scale:
    manager.move_to_gpu(layer.attention.v_scale.name)
if layer.attention.o_scale:
    manager.move_to_gpu(layer.attention.o_scale.name)

# Make sure weights are correctly loaded
dtype = layer.attention_norm.data.dtype
device = layer.attention_norm.data.device
print(f"Weights loaded onto device: {device}, dtype: {dtype}")

# 2. Run initial forward pass (prefill phase, seq_len = 4)
print("\nRunning initial forward pass (prefill, seq_len = 4)...")
hidden = torch.randn(1, 4, model.config.hidden_size, device=device, dtype=dtype)
position_ids = torch.arange(4, device=device).unsqueeze(0)  # shape (1, 4)

output = attention.forward(hidden, layer, kv_cache, position_ids)
print(f"Input shape: {hidden.shape}")
print(f"Output shape: {output.shape}")
print(f"KV Cache seq_len for Layer 3: {kv_cache.sequence_length(layer.layer_id)}")

# 3. Run autoregressive forward pass (generation phase, seq_len = 1)
print("\nRunning generation step (seq_len = 1)...")
hidden_new = torch.randn(1, 1, model.config.hidden_size, device=device, dtype=dtype)
position_ids_new = torch.tensor([[4]], device=device)  # shape (1, 1)

output_new = attention.forward(hidden_new, layer, kv_cache, position_ids_new)
print(f"New Input shape: {hidden_new.shape}")
print(f"New Output shape: {output_new.shape}")
print(f"KV Cache seq_len for Layer 3: {kv_cache.sequence_length(layer.layer_id)}")

print("\nSUCCESS!")
