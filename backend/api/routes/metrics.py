from fastapi import APIRouter, HTTPException
from backend.api.engine_manager import engine_manager
import psutil
import torch
import os

router = APIRouter()

@router.get("/v1/stats")
async def get_stats():
    # If stats loop has run, return last system stats
    if hasattr(engine_manager, "last_system_stats"):
        return engine_manager.last_system_stats
    
    # Fallback
    ram = psutil.virtual_memory()
    return {
        "status": engine_manager.status,
        "active_model": engine_manager.active_model_id,
        "ram_usage_percent": ram.percent,
        "ram_used_mb": ram.used / (1024**2),
        "ram_total_mb": ram.total / (1024**2),
        "cpu_percent": psutil.cpu_percent(),
        "vram_allocated_mb": torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0,
    }

@router.get("/v1/metrics")
async def get_metrics():
    return engine_manager.last_metrics

@router.get("/v1/gpu")
async def get_gpu():
    is_cuda = torch.cuda.is_available()
    vram_alloc = torch.cuda.memory_allocated() / (1024**2) if is_cuda else 0.0
    vram_reserved = torch.cuda.memory_reserved() / (1024**2) if is_cuda else 0.0
    
    return {
        "device": getattr(engine_manager.loader, "DEVICE", "cpu"),
        "cuda_available": is_cuda,
        "vram_allocated_mb": vram_alloc,
        "vram_reserved_mb": vram_reserved,
        "peak_vram_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2) if is_cuda else 0.0,
        "device_name": torch.cuda.get_device_name(0) if is_cuda else "CPU"
    }

@router.get("/v1/storage")
async def get_storage():
    ram_cache_len = 0
    ram_cache_hits = 0
    ssd_hits = 0
    if engine_manager.loader:
        ram_cache_len = len(engine_manager.loader.ram_cache.cache) if hasattr(engine_manager.loader, "ram_cache") else 0
        ram_cache_hits = engine_manager.loader.ram_hits
        ssd_hits = engine_manager.loader.ssd_hits
        
    return {
        "storage_backend": "SafeTensors",
        "mmap_enabled": True,
        "ram_cache_items": ram_cache_len,
        "ram_cache_hits": ram_cache_hits,
        "ssd_reads": ssd_hits,
        "read_ahead": True
    }

@router.get("/v1/cache")
async def get_cache():
    kv_allocated = 0.0
    context_length = 0
    if engine_manager.loader:
        # Estimated KV cache
        kv_allocated = 256.0  # mock MB
        context_length = 4096
    return {
        "context_length": context_length,
        "sliding_window": 1024,
        "paged_attention": True,
        "block_size": 16,
        "kv_allocated_mb": kv_allocated
    }

@router.get("/v1/moe")
async def get_moe():
    num_experts = 0
    active_experts = 0
    if engine_manager.adapter:
        num_experts = getattr(engine_manager.adapter, "num_experts", 0) or 64 # fallback
        active_experts = getattr(engine_manager.loader, "cache_limit", 128)
    return {
        "total_experts": num_experts,
        "active_experts": active_experts,
        "routing_strategy": "Top-K",
        "expert_fusion": True
    }

@router.get("/v1/experts")
async def get_experts():
    # Construct expert metadata
    expert_stats = []
    if engine_manager.loader:
        # Loop through static expert cache and extract metadata
        metadata = engine_manager.loader.expert_metadata
        for slot, meta in metadata.items():
            key = meta["key"] # (layer_id, expert_id)
            expert_stats.append({
                "layer_id": key[0],
                "expert_id": key[1],
                "hits": meta["hits"],
                "load_ms": round(meta["load_ms"], 2),
                "vram_cost_kb": round(meta["vram_cost"] / 1024, 2)
            })
    return {"experts": expert_stats}

@router.get("/v1/pipeline")
async def get_pipeline():
    return {
        "status": engine_manager.status,
        "stage": "Decode" if engine_manager.status == "Generating" else "Idle",
        "layer_count": engine_manager.adapter.num_layers if engine_manager.adapter else 0,
        "current_layer": getattr(engine_manager, "current_layer", 0)
    }

@router.get("/v1/layers")
async def get_layers():
    layers = []
    if engine_manager.adapter:
        num_layers = engine_manager.adapter.num_layers
        for idx in range(num_layers):
            layers.append({
                "layer_id": idx,
                "attention_heads": 32,
                "experts_count": 64,
                "tensor_size_mb": 96.0, # typical MoE layer size for FP8
                "load_time_ms": 0.0,
                "compute_time_ms": 0.0
            })
    return {"layers": layers}
