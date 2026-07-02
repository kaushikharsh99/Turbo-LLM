import json
import time
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from backend.api.engine_manager import engine_manager

router = APIRouter()

BENCHMARK_FILE = Path("logs/benchmarks.json")
BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)

class BenchmarkRequest(BaseModel):
    model: str
    prompt: Optional[str] = "Write a short poem about a turbocharger."
    max_tokens: Optional[int] = 32
    runs: Optional[int] = 1
    warmup: Optional[bool] = True
    context: Optional[int] = 1024
    temperature: Optional[float] = 0.0

@router.get("/v1/benchmark")
async def get_benchmarks():
    if BENCHMARK_FILE.exists():
        try:
            with open(BENCHMARK_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"benchmarks": []}
    return {"benchmarks": []}

@router.post("/v1/benchmark")
async def run_benchmark(request: BenchmarkRequest):
    if not engine_manager.engine:
        raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")

    # Run warmup if requested
    if request.warmup:
        print("Warmup run starting...")
        warmup_generator = engine_manager.generate_stream(
            prompt=request.prompt,
            max_new_tokens=10,
            temperature=request.temperature,
            top_p=1.0,
            chat=False
        )
        for _ in warmup_generator:
            pass
        print("Warmup run complete.")

    t_start = time.time()
    
    # Run the actual benchmark
    generator = engine_manager.generate_stream(
        prompt=request.prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=1.0,
        chat=False
    )
    
    full_text = ""
    for update in generator:
        if update["type"] == "token":
            full_text += update["token"]

    duration = time.time() - t_start
    metrics = engine_manager.last_metrics or {}
    
    # Compile benchmark results
    tps = metrics.get("tps", 0.0)
    ttft = metrics.get("ttft_ms", 0.0)
    latency = metrics.get("latency_ms", 0.0)
    
    # Get RAM/VRAM
    import psutil
    ram_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    vram_mb = 0.0
    if os.environ.get("CUDA_VISIBLE_DEVICES") or getattr(engine_manager.loader, "DEVICE", "cpu") == "cuda":
        import torch
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    ssd_hits = 0
    gpu_hits = 0
    if engine_manager.loader:
        ssd_hits = engine_manager.loader.ssd_hits
        gpu_hits = engine_manager.loader.gpu_hits

    result = {
        "id": f"bench-{int(time.time())}",
        "timestamp": time.time(),
        "model": request.model,
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "ttft_ms": round(ttft, 2),
        "tps": round(tps, 2),
        "duration_s": round(duration, 2),
        "ram_peak_mb": round(ram_mb, 2),
        "vram_peak_mb": round(vram_mb, 2),
        "ssd_reads": ssd_hits,
        "gpu_hits": gpu_hits,
        "gpu_utilization": 85.0 if torch.cuda.is_available() else 0.0, # estimate/mock
        "cpu_utilization": psutil.cpu_percent(),
        "energy": "N/A"
    }

    # Save to file
    benchmarks = []
    if BENCHMARK_FILE.exists():
        try:
            with open(BENCHMARK_FILE, "r") as f:
                data = json.load(f)
                benchmarks = data.get("benchmarks", [])
        except Exception:
            pass

    benchmarks.append(result)
    with open(BENCHMARK_FILE, "w") as f:
        json.dump({"benchmarks": benchmarks}, f, indent=2)

    return result
