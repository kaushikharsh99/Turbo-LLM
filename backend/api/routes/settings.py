from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from backend.api.engine_manager import engine_manager

router = APIRouter()

class UpdateSettingsRequest(BaseModel):
    settings: Dict[str, Any]

@router.get("/v1/settings")
async def get_settings():
    return {
        "status": "success",
        "settings": engine_manager.config
    }

@router.post("/v1/settings")
async def update_settings(request: UpdateSettingsRequest):
    try:
        # Merge settings
        for section, values in request.settings.items():
            if isinstance(values, dict) and section in engine_manager.config:
                engine_manager.config[section].update(values)
            else:
                engine_manager.config[section] = values
        
        # If loader is loaded, we can dynamically adjust cached limit
        if engine_manager.loader:
            if "cache" in engine_manager.config and "expert_limit" in engine_manager.config["cache"]:
                limit = engine_manager.config["cache"]["expert_limit"]
                if limit != "auto":
                    engine_manager.loader.cache_limit = int(limit)
                else:
                    engine_manager.loader.adjust_cache_limit()
                    
        return {"status": "success", "settings": engine_manager.config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/engine")
async def update_engine_settings(settings: Dict[str, Any]):
    return await update_settings(UpdateSettingsRequest(settings={"runtime": settings}))

@router.post("/v1/quantization")
async def update_quantization_settings(settings: Dict[str, Any]):
    return await update_settings(UpdateSettingsRequest(settings={"execution": settings}))

@router.post("/v1/cache")
async def update_cache_settings(settings: Dict[str, Any]):
    return await update_settings(UpdateSettingsRequest(settings={"cache": settings}))

@router.post("/v1/storage")
async def update_storage_settings(settings: Dict[str, Any]):
    return await update_settings(UpdateSettingsRequest(settings={"memory": settings}))
