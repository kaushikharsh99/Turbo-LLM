from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
from backend.api.engine_manager import engine_manager
from loader.model_manager import get_model

router = APIRouter()

class LoadModelRequest(BaseModel):
    model: str
    config: Optional[Dict[str, Any]] = None

class DownloadModelRequest(BaseModel):
    model: str

@router.get("/v1/models")
async def get_models():
    # Fetch running model & installed models
    installed = engine_manager.get_installed_models()
    
    # Recommended list of models for Turbo-LLM
    recommended = [
        {"id": "Qwen/Qwen3.6-35B-A3B-FP8", "name": "Qwen 3.6 35B A3B FP8 (Recommended)", "size": "35B", "architecture": "MoE"},
        {"id": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "name": "Qwen 3 30B A3B Instruct FP8", "size": "30B", "architecture": "MoE"},
    ]
    
    # Map installed model IDs
    installed_ids = {m["id"] for m in installed}
    
    models_list = []
    # Add installed
    for m in installed:
        models_list.append({
            "id": m["id"],
            "object": "model",
            "owned_by": "turbo-llm",
            "status": "Running" if engine_manager.active_model_id == m["id"] else "Installed",
            "path": m["path"],
            "size_gb": round(m["size_gb"], 2) if m["size_gb"] else 0.0,
            "architecture": "MoE"
        })
        
    # Add recommended if not installed
    for r in recommended:
        if r["id"] not in installed_ids:
            models_list.append({
                "id": r["id"],
                "object": "model",
                "owned_by": "huggingface",
                "status": "Available",
                "path": None,
                "size_gb": 0.0,
                "architecture": r["architecture"]
            })
            
    return {"object": "list", "data": models_list}

@router.post("/v1/load")
async def load_model(request: LoadModelRequest):
    try:
        engine_manager.load_model(request.model, request.config)
        return {"status": "Loading started", "model": request.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/unload")
async def unload_model():
    try:
        engine_manager.unload_model()
        return {"status": "Model unloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/download")
async def download_model(request: DownloadModelRequest, background_tasks: BackgroundTasks):
    def _download():
        try:
            print(f"Background download started for model: {request.model}")
            get_model(request.model)
            print(f"Background download completed for model: {request.model}")
        except Exception as e:
            print(f"Error in background download of {request.model}: {e}")

    background_tasks.add_task(_download)
    return {"status": "Download started in background", "model": request.model}
