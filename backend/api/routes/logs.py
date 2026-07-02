from fastapi import APIRouter
from backend.api.engine_manager import engine_manager

router = APIRouter()

@router.get("/v1/logs")
async def get_logs():
    # If LogInterceptor has history, return it
    logs = []
    if hasattr(engine_manager.log_interceptor, "history"):
        with engine_manager.log_interceptor.lock:
            logs = list(engine_manager.log_interceptor.history)
            
    # Parse logs into sections if client wants, or return raw strings
    return {
        "status": "success",
        "logs": logs
    }
