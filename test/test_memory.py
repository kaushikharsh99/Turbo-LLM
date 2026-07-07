import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from loader.model_loader import load_model
from memory.memory_manager import MemoryManager

MODEL_DIR = "/home/harsh/.turbollm/models/Qwen_Qwen3.6-35B-A3B-FP8"

print("Loading model...")
model = load_model(MODEL_DIR)

print("\nInitializing MemoryManager...")
# CPU RAM limit: 3 GB
# GPU VRAM limit: 150 MB (to easily trigger eviction with layer weights)
max_cpu = 3 * 1024 * 1024 * 1024
max_gpu = 150 * 1024 * 1024

manager = MemoryManager(model, max_cpu, max_gpu)

print(f"Total registered tensors: {len(manager.tensor_registry)}")

# Test 1: Load embedding
print("\nTest 1: Load embedding to CPU...")
embed_name = model.embedding.name
embed_data = manager.load_tensor(embed_name)
print(f"Embedding shape: {embed_data.shape}, device: {embed_data.device}")
print(f"Embedding metadata: loaded={model.embedding.loaded}, device={model.embedding.device}")

# Test 2: Move embedding to GPU
print("\nTest 2: Move embedding to GPU...")
embed_gpu = manager.move_to_gpu(embed_name)
print(f"Embedding data device: {embed_gpu.device}")
print(f"Embedding metadata after move: loaded={model.embedding.loaded}, device={model.embedding.device}")

# Test 3: Load layer projections to trigger eviction
print("\nTest 3: Moving layer attention projections to GPU to trigger eviction...")
proj_names = [
    model.layers[3].attention.q_proj.name,
    model.layers[3].attention.k_proj.name,
    model.layers[3].attention.v_proj.name,
    model.layers[3].attention.o_proj.name,
    model.layers[7].attention.q_proj.name,
    model.layers[7].attention.k_proj.name,
    model.layers[7].attention.v_proj.name,
]

print("Moving layer 3 projections to GPU (Q, K, V, O)...")
for name in proj_names[:4]:
    print(f"Moving {name} to GPU (size: {manager.tensor_registry[name].size_bytes / (1024*1024):.2f} MB)...")
    manager.move_to_gpu(name)

print(f"GPU Memory usage: {manager.gpu_memory.current_bytes / (1024*1024):.2f} MB")
print(f"Layer 3 Q Proj Metadata: loaded={model.layers[3].attention.q_proj.loaded}, device={model.layers[3].attention.q_proj.device}")

print("\nMoving layer 7 projections to GPU to exceed 150 MB limit...")
for name in proj_names[4:]:
    print(f"Moving {name} to GPU (size: {manager.tensor_registry[name].size_bytes / (1024*1024):.2f} MB)...")
    manager.move_to_gpu(name)

# Now load layers 11, 15, 19 to trigger clear eviction of layer 3 projections
extra_proj_names = [
    model.layers[11].attention.q_proj.name,
    model.layers[11].attention.k_proj.name,
    model.layers[11].attention.v_proj.name,
    model.layers[11].attention.o_proj.name,
    model.layers[15].attention.q_proj.name,
]

print("\nMoving additional layers (11 and 15) to GPU to force eviction of layer 3...")
for name in extra_proj_names:
    print(f"Moving {name} to GPU (size: {manager.tensor_registry[name].size_bytes / (1024*1024):.2f} MB)...")
    manager.move_to_gpu(name)

print(f"GPU Memory usage: {manager.gpu_memory.current_bytes / (1024*1024):.2f} MB")
# Layer 3 Q Proj should be evicted by now
print(f"Layer 3 Q Proj Metadata: loaded={model.layers[3].attention.q_proj.loaded}, device={model.layers[3].attention.q_proj.device}")

print("\nSUCCESS!")
