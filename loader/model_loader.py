import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

from model.model import Model, ModelConfig
from model.tensor import Tensor
from model.layer import Layer, Attention, Router, Expert, SharedExpert, MoE
from loader.safetensor_loader import SafeTensorLoader
from loader.layer_loader import LayerLoader


def get_config_val(cfg: dict, key: str, default: Any = None) -> Any:
    """Helper to extract a config value, checking inside 'text_config' first."""
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        if key in cfg["text_config"]:
            return cfg["text_config"][key]
    return cfg.get(key, default)


def make_tensor_metadata(name: str, tensor_loader: SafeTensorLoader, model_dir: Path) -> Optional[Tensor]:
    """Creates a metadata-only Tensor instance for a given tensor name."""
    if name not in tensor_loader:
        return None
    shape = tensor_loader.get_shape(name)
    dtype = tensor_loader.get_dtype(name)
    filename = tensor_loader.tensor_file(name)
    file_path = model_dir / filename

    # Estimate size in bytes
    numel = 1
    for dim in shape:
        numel *= dim
    try:
        element_size = torch.tensor([], dtype=dtype).element_size()
    except Exception:
        element_size = 2  # default fallback (e.g. for bfloat16/float16)
    size_bytes = numel * element_size

    return Tensor(
        name=name,
        shape=shape,
        dtype=dtype,
        file_path=file_path,
        file_offset=0,
        size_bytes=size_bytes,
        device="disk",
        loaded=False,
        pinned=False,
        data=None,
    )


def find_tensor_name(loader: SafeTensorLoader, candidates: List[str]) -> str:
    """Helper to find the first candidate tensor name that exists in the loader."""
    for candidate in candidates:
        if candidate in loader:
            return candidate
    raise KeyError(f"None of the candidate tensors {candidates} found in weight map.")


def load_model(model_dir: str | Path) -> Model:
    """Loads model configuration and index to build a Model metadata structure."""
    model_dir = Path(model_dir)

    # 1. Open config
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
    
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # 2. Open index
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
    else:
        # Fallback: scan all .safetensors files in the directory to map tensor names to files
        weight_map = {}
        from safetensors import safe_open
        for file in sorted(model_dir.glob("*.safetensors")):
            try:
                with safe_open(file, framework="pt") as f:
                    for key in f.keys():
                        weight_map[key] = file.name
            except Exception as e:
                # Log or handle corrupted safetensors file
                pass

    if not weight_map:
        raise ValueError(f"No safetensors files or weights found in {model_dir}")

    # 3. Create SafeTensorLoader
    tensor_loader = SafeTensorLoader(model_dir, weight_map)

    # 4. Create LayerLoader
    layer_loader = LayerLoader(tensor_loader)

    # 5. Populate ModelConfig
    architecture = cfg.get("architectures", [""])[0]
    num_layers = get_config_val(cfg, "num_hidden_layers")
    hidden_size = get_config_val(cfg, "hidden_size")

    # Handle intermediate size for MoE/dense
    intermediate_size = get_config_val(cfg, "intermediate_size")
    if intermediate_size is None:
        intermediate_size = get_config_val(cfg, "moe_intermediate_size")
    if intermediate_size is None:
        intermediate_size = get_config_val(cfg, "shared_expert_intermediate_size")
    if intermediate_size is None:
        intermediate_size = hidden_size * 4  # standard fallback

    vocab_size = get_config_val(cfg, "vocab_size")
    max_position_embeddings = get_config_val(cfg, "max_position_embeddings")
    num_attention_heads = get_config_val(cfg, "num_attention_heads")
    num_key_value_heads = get_config_val(cfg, "num_key_value_heads")

    # Extract rope_theta
    rope_theta = get_config_val(cfg, "rope_theta")
    if rope_theta is None:
        rope_params = get_config_val(cfg, "rope_parameters")
        if isinstance(rope_params, dict):
            rope_theta = rope_params.get("rope_theta")
    if rope_theta is None:
        rope_theta = 10000.0

    rms_norm_eps = get_config_val(cfg, "rms_norm_eps", 1e-6)
    num_experts = get_config_val(cfg, "num_experts", 0)
    experts_per_token = get_config_val(cfg, "num_experts_per_tok", 0)

    # Parse head_dim and partial_rotary_factor
    q_proj_name = None
    for k in tensor_loader.weight_map.keys():
        if "self_attn.q_proj.weight" in k:
            q_proj_name = k
            break
    if q_proj_name:
        q_shape = tensor_loader.get_shape(q_proj_name)
        head_dim = q_shape[0] // num_attention_heads
    else:
        head_dim = get_config_val(cfg, "head_dim")
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads

    partial_rotary_factor = get_config_val(cfg, "partial_rotary_factor")
    if partial_rotary_factor is None:
        rope_params = get_config_val(cfg, "rope_parameters")
        if isinstance(rope_params, dict):
            partial_rotary_factor = rope_params.get("partial_rotary_factor")
    if partial_rotary_factor is None:
        partial_rotary_factor = 1.0

    model_config = ModelConfig(
        architecture=architecture,
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        rope_theta=float(rope_theta),
        rms_norm_eps=float(rms_norm_eps),
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        head_dim=int(head_dim),
        partial_rotary_factor=float(partial_rotary_factor),
    )

    # 6. Build embedding, final_norm, lm_head metadata Tensors
    embed_name = find_tensor_name(
        tensor_loader,
        ["model.language_model.embed_tokens.weight", "model.embed_tokens.weight"],
    )
    norm_name = find_tensor_name(
        tensor_loader,
        ["model.language_model.norm.weight", "model.norm.weight"],
    )
    lm_head_name = find_tensor_name(
        tensor_loader,
        ["lm_head.weight"],
    )

    embedding_tensor = make_tensor_metadata(embed_name, tensor_loader, model_dir)
    final_norm_tensor = make_tensor_metadata(norm_name, tensor_loader, model_dir)
    lm_head_tensor = make_tensor_metadata(lm_head_name, tensor_loader, model_dir)

    # 7. Build Layers metadata
    layers = []
    
    # Detect layer prefix dynamically
    if "model.language_model.layers.0.input_layernorm.weight" in tensor_loader:
        prefix_pattern = "model.language_model.layers.{layer_id}"
    elif "model.layers.0.input_layernorm.weight" in tensor_loader:
        prefix_pattern = "model.layers.{layer_id}"
    else:
        # Fallback to language_model style
        prefix_pattern = "model.language_model.layers.{layer_id}"

    for layer_id in range(num_layers):
        prefix = prefix_pattern.format(layer_id=layer_id)

        # Norms
        attn_norm = make_tensor_metadata(
            f"{prefix}.input_layernorm.weight", tensor_loader, model_dir
        )
        ffn_norm = make_tensor_metadata(
            f"{prefix}.post_attention_layernorm.weight", tensor_loader, model_dir
        )

        # Attention
        attn = Attention(
            q_proj=make_tensor_metadata(f"{prefix}.self_attn.q_proj.weight", tensor_loader, model_dir),
            k_proj=make_tensor_metadata(f"{prefix}.self_attn.k_proj.weight", tensor_loader, model_dir),
            v_proj=make_tensor_metadata(f"{prefix}.self_attn.v_proj.weight", tensor_loader, model_dir),
            o_proj=make_tensor_metadata(f"{prefix}.self_attn.o_proj.weight", tensor_loader, model_dir),
            q_scale=make_tensor_metadata(f"{prefix}.self_attn.q_proj.weight_scale_inv", tensor_loader, model_dir),
            k_scale=make_tensor_metadata(f"{prefix}.self_attn.k_proj.weight_scale_inv", tensor_loader, model_dir),
            v_scale=make_tensor_metadata(f"{prefix}.self_attn.v_proj.weight_scale_inv", tensor_loader, model_dir),
            o_scale=make_tensor_metadata(f"{prefix}.self_attn.o_proj.weight_scale_inv", tensor_loader, model_dir),
        )

        # MoE
        router_gate_name = f"{prefix}.mlp.gate.weight"
        router = None
        if router_gate_name in tensor_loader:
            router = Router(gate=make_tensor_metadata(router_gate_name, tensor_loader, model_dir))

        # Experts
        experts = []
        for expert_id in range(num_experts):
            expert_gate = f"{prefix}.mlp.experts.{expert_id}.gate_proj.weight"
            if expert_gate not in tensor_loader:
                break
            
            expert = Expert(
                gate_proj=make_tensor_metadata(expert_gate, tensor_loader, model_dir),
                up_proj=make_tensor_metadata(f"{prefix}.mlp.experts.{expert_id}.up_proj.weight", tensor_loader, model_dir),
                down_proj=make_tensor_metadata(f"{prefix}.mlp.experts.{expert_id}.down_proj.weight", tensor_loader, model_dir),
            )
            experts.append(expert)

        # Shared Expert
        shared_expert = None
        shared_gate = f"{prefix}.mlp.shared_expert.gate_proj.weight"
        if shared_gate in tensor_loader:
            shared_expert = SharedExpert(
                gate_proj=make_tensor_metadata(shared_gate, tensor_loader, model_dir),
                up_proj=make_tensor_metadata(f"{prefix}.mlp.shared_expert.up_proj.weight", tensor_loader, model_dir),
                down_proj=make_tensor_metadata(f"{prefix}.mlp.shared_expert.down_proj.weight", tensor_loader, model_dir),
            )

        moe = MoE(
            router=router,
            experts=experts,
            shared_expert=shared_expert,
        )

        layer = Layer(
            layer_id=layer_id,
            attention_norm=attn_norm,
            ffn_norm=ffn_norm,
            attention=attn,
            moe=moe,
        )
        layers.append(layer)

    # 8. Create Model object
    model = Model(
        config=model_config,
        embedding=embedding_tensor,
        layers=layers,
        final_norm=final_norm_tensor,
        lm_head=lm_head_tensor,
        loaded=False,
        current_device="disk",
        tensor_loader=tensor_loader,
        layer_loader=layer_loader,
    )

    return model
