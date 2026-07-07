import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from loader.model_loader import load_model

MODEL_DIR = "/home/harsh/.turbollm/models/Qwen_Qwen3.6-35B-A3B-FP8"

print("Loading model...")
model = load_model(MODEL_DIR)

print("\nModel Config:")
print(f"Architecture: {model.config.architecture}")
print(f"Num layers: {model.config.num_layers}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Intermediate size: {model.config.intermediate_size}")
print(f"Vocab size: {model.config.vocab_size}")
print(f"Num experts: {model.config.num_experts}")
print(f"Experts per token: {model.config.experts_per_token}")

print("\nTensors loaded:")
print(f"Embedding name: {model.embedding.name}, shape: {model.embedding.shape}, dtype: {model.embedding.dtype}, size: {model.embedding.size_bytes} bytes")
print(f"Final Norm name: {model.final_norm.name}, shape: {model.final_norm.shape}, dtype: {model.final_norm.dtype}, size: {model.final_norm.size_bytes} bytes")
print(f"LM Head name: {model.lm_head.name}, shape: {model.lm_head.shape}, dtype: {model.lm_head.dtype}, size: {model.lm_head.size_bytes} bytes")

print("\nLayer 0 (Linear Attention) Metadata:")
l0 = model.layers[0]
print(f"Attention Norm: {l0.attention_norm.name}, shape: {l0.attention_norm.shape}")
print(f"FFN Norm: {l0.ffn_norm.name}, shape: {l0.ffn_norm.shape}")
if l0.attention and l0.attention.q_proj:
    print(f"Q proj: {l0.attention.q_proj.name}, shape: {l0.attention.q_proj.shape}")
else:
    print("Q proj: None (Linear Attention Layer)")
print(f"Num experts: {len(l0.moe.experts)}")
if len(l0.moe.experts) > 0:
    print(f"Expert 0 Gate Proj: {l0.moe.experts[0].gate_proj.name}, shape: {l0.moe.experts[0].gate_proj.shape}")
if l0.moe.shared_expert:
    print(f"Shared Expert Gate Proj: {l0.moe.shared_expert.gate_proj.name}, shape: {l0.moe.shared_expert.gate_proj.shape}")

print("\nLayer 3 (Full Attention) Metadata:")
l3 = model.layers[3]
print(f"Attention Norm: {l3.attention_norm.name}, shape: {l3.attention_norm.shape}")
print(f"FFN Norm: {l3.ffn_norm.name}, shape: {l3.ffn_norm.shape}")
if l3.attention and l3.attention.q_proj:
    print(f"Q proj: {l3.attention.q_proj.name}, shape: {l3.attention.q_proj.shape}")
else:
    print("Q proj: None")
print(f"Num experts: {len(l3.moe.experts)}")
if len(l0.moe.experts) > 0:
    print(f"Expert 0 Gate Proj: {l0.moe.experts[0].gate_proj.name}, shape: {l0.moe.experts[0].gate_proj.shape}")
if l0.moe.shared_expert:
    print(f"Shared Expert Gate Proj: {l0.moe.shared_expert.gate_proj.name}, shape: {l0.moe.shared_expert.gate_proj.shape}")

print("\nSUCCESS!")
