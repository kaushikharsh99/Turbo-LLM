"""
Transformer layer definitions.

A Layer is a passive description of one transformer block.
It contains no execution logic.

Execution is handled entirely by runtime/executor.py
"""

from dataclasses import dataclass, field

from model.tensor import Tensor

@dataclass(slots=True)
class Attention:

    q_proj: Tensor
    k_proj: Tensor
    v_proj: Tensor
    o_proj: Tensor

@dataclass(slots=True)
class Expert:

    gate_proj: Tensor

    up_proj: Tensor

    down_proj: Tensor

@dataclass(slots=True)
class SharedExpert:

    gate_proj: Tensor

    up_proj: Tensor

    down_proj: Tensor

@dataclass(slots=True)
class Router:

    gate: Tensor

@dataclass(slots=True)
class MoE:

    router: Router

    experts: list[Expert] = field(default_factory=list)

    shared_expert: SharedExpert | None = None

@dataclass(slots=True)
class Layer:

    layer_id: int

    attention_norm: Tensor

    ffn_norm: Tensor

    attention: Attention

    moe: MoE