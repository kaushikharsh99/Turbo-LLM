import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from loader.model_loader import load_model
from memory.memory_manager import MemoryManager
from attension.attension import Attention
from attension.kv_cache import KVCache
from runtime.device import DeviceManager

MODEL_DIR = "/home/harsh/.turbollm/models/Qwen_Qwen3.6-35B-A3B-FP8"

def test_kivi_mode(device_name="cuda", k_bits=4, v_bits=4, group_size=32, residual_length=32):
    print(f"\n--- Testing KIVI (device={device_name}, k_bits={k_bits}, v_bits={v_bits}, group_size={group_size}, residual_len={residual_length}) ---")
    
    device_manager = DeviceManager(device_name)
    model = load_model(MODEL_DIR)
    manager = MemoryManager(model, 3 * 1024**3, 1024**3, device_manager=device_manager)
    attention = Attention(model.config)
    
    # Initialize KV Cache with KIVI settings
    kv_cache = KVCache(
        device_manager=device_manager,
        k_bits=k_bits,
        v_bits=v_bits,
        group_size=group_size,
        residual_length=residual_length
    )
    
    layer = model.layers[3]
    manager.move_to_gpu(layer.attention_norm.name)
    manager.move_to_gpu(layer.attention.q_proj.name)
    manager.move_to_gpu(layer.attention.k_proj.name)
    manager.move_to_gpu(layer.attention.v_proj.name)
    manager.move_to_gpu(layer.attention.o_proj.name)
    
    dtype = layer.attention_norm.data.dtype
    device = layer.attention_norm.data.device
    print(f"Weights loaded onto: {device}, dtype: {dtype}")

    # 1. Prefill Phase (length = 80, to exceed residual_length = 32 and trigger quantization)
    print("Running initial forward pass (prefill, seq_len = 80)...")
    hidden = torch.randn(1, 80, model.config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(80, device=device).unsqueeze(0)
    
    output = attention.forward(hidden, layer, kv_cache, position_ids)
    print(f"Input shape: {hidden.shape}, Output shape: {output.shape}")
    print(f"KV Cache seq_len for Layer 3: {kv_cache.sequence_length(layer.layer_id)}")
    
    # Verify components are allocated
    if layer.layer_id in kv_cache.k_codes:
        print(f"Quantized K code shape: {kv_cache.k_codes[layer.layer_id].shape}")
        print(f"Quantized K scale shape: {kv_cache.k_scales[layer.layer_id].shape}")
        print(f"Residual K shape: {kv_cache.k_residuals[layer.layer_id].shape}")
    if layer.layer_id in kv_cache.v_codes:
        print(f"Quantized V code shape: {kv_cache.v_codes[layer.layer_id].shape}")
        print(f"Quantized V scale shape: {kv_cache.v_scales[layer.layer_id].shape}")
        print(f"Residual V shape: {kv_cache.v_residuals[layer.layer_id].shape}")
        
    # 2. Decoding Step (seq_len = 1)
    print("Running generation step (decode, seq_len = 1)...")
    hidden_new = torch.randn(1, 1, model.config.hidden_size, device=device, dtype=dtype)
    position_ids_new = torch.tensor([[80]], device=device)
    
    output_new = attention.forward(hidden_new, layer, kv_cache, position_ids_new)
    print(f"New Output shape: {output_new.shape}")
    print(f"KV Cache seq_len for Layer 3: {kv_cache.sequence_length(layer.layer_id)}")
    print(f"New Residual K shape: {kv_cache.k_residuals[layer.layer_id].shape}")
    
    print("Success for this config!")

if __name__ == "__main__":
    # Test 4-bit KIVI on CUDA (Triton)
    try:
        test_kivi_mode(device_name="cuda", k_bits=4, v_bits=4)
    except Exception as e:
        print(f"4-bit CUDA KIVI failed: {e}")
        
    # Test 2-bit KIVI on CUDA (Triton)
    try:
        test_kivi_mode(device_name="cuda", k_bits=2, v_bits=2)
    except Exception as e:
        print(f"2-bit CUDA KIVI failed: {e}")

    # Test 4-bit KIVI on CPU (PyTorch fallback)
    try:
        test_kivi_mode(device_name="cpu", k_bits=4, v_bits=4)
    except Exception as e:
        print(f"4-bit CPU KIVI failed: {e}")

    print("\nALL TESTS COMPLETED!")
