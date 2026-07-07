"""
Transformer layer definitions.

A Layer is a passive description of one transformer block.
It contains no execution logic.

Execution is handled entirely by runtime/executor.py
"""

from dataclasses import dataclass, field
from typing import Optional

from model.tensor import Tensor

@dataclass(slots=True)
class Attention:

    q_proj: Optional[Tensor] = None
    k_proj: Optional[Tensor] = None
    v_proj: Optional[Tensor] = None
    o_proj: Optional[Tensor] = None
    q_scale: Optional[Tensor] = None
    k_scale: Optional[Tensor] = None
    v_scale: Optional[Tensor] = None
    o_scale: Optional[Tensor] = None

@dataclass(slots=True)
class Expert:

    gate_proj: Optional[Tensor] = None
    up_proj: Optional[Tensor] = None
    down_proj: Optional[Tensor] = None

@dataclass(slots=True)
class SharedExpert:

    gate_proj: Optional[Tensor] = None
    up_proj: Optional[Tensor] = None
    down_proj: Optional[Tensor] = None

@dataclass(slots=True)
class Router:

    gate: Optional[Tensor] = None

@dataclass(slots=True)
class MoE:

    router: Optional[Router] = None
    experts: list[Expert] = field(default_factory=list)
    shared_expert: Optional[SharedExpert] = None

@dataclass(slots=True)
class Layer:

    layer_id: int
    attention_norm: Optional[Tensor] = None
    ffn_norm: Optional[Tensor] = None
    attention: Optional[Attention] = None
    moe: Optional[MoE] = None