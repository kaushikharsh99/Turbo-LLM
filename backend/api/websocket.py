from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import queue
from backend.api.engine_manager import engine_manager

router = APIRouter()

# Active connections lists
class ConnectionManager:
    def __init__(self):
        self.stats_connections = []
        self.pipeline_connections = []
        self.experts_connections = []
        self.metrics_connections = []
        self.tokens_connections = []

    async def connect_stats(self, websocket: WebSocket):
        await websocket.accept()
        self.stats_connections.append(websocket)

    def disconnect_stats(self, websocket: WebSocket):
        if websocket in self.stats_connections:
            self.stats_connections.remove(websocket)

    async def connect_pipeline(self, websocket: WebSocket):
        await websocket.accept()
        self.pipeline_connections.append(websocket)

    def disconnect_pipeline(self, websocket: WebSocket):
        if websocket in self.pipeline_connections:
            self.pipeline_connections.remove(websocket)

    async def broadcast_pipeline(self, message: dict):
        closed = []
        for ws in self.pipeline_connections:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                closed.append(ws)
        for ws in closed:
            self.disconnect_pipeline(ws)

    async def broadcast_stats(self, message: dict):
        closed = []
        for ws in self.stats_connections:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                closed.append(ws)
        for ws in closed:
            self.disconnect_stats(ws)


manager = ConnectionManager()

# Hook into engine_manager methods to broadcast to WebSocket
def patch_engine_manager_broadcasts():
    # Update EngineManager methods to send to WebSocket manager
    loop = asyncio.get_event_loop()
    
    def broadcast_pipeline_update_ws(payload):
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(
                manager.broadcast_pipeline(payload),
                loop
            )
            
    def broadcast_stats_update_ws(stats_payload):
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(
                manager.broadcast_stats({
                    "type": "token_stats",
                    "stats": stats_payload
                }),
                loop
            )
            
    engine_manager.broadcast_pipeline_update = broadcast_pipeline_update_ws
    engine_manager.broadcast_stats_update = broadcast_stats_update_ws

# Apply the patches
patch_engine_manager_broadcasts()


@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    log_queue = queue.Queue(maxsize=1000)
    engine_manager.log_interceptor.register_queue(log_queue)
    
    # Send existing history first
    if hasattr(engine_manager.log_interceptor, "history"):
        with engine_manager.log_interceptor.lock:
            for log_line in list(engine_manager.log_interceptor.history):
                try:
                    await websocket.send_text(json.dumps({"type": "log", "message": log_line}))
                except Exception:
                    break

    try:
        while True:
            # We must be careful not to block the async loop.
            # Read from thread-safe queue. Use a small timeout so we yield control.
            try:
                log_line = log_queue.get_nowait()
                await websocket.send_text(json.dumps({"type": "log", "message": log_line}))
            except queue.Empty:
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        engine_manager.log_interceptor.unregister_queue(log_queue)


@router.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    await manager.connect_stats(websocket)
    try:
        while True:
            # Send latest system stats every 1 second
            if hasattr(engine_manager, "last_system_stats"):
                await websocket.send_text(json.dumps(engine_manager.last_system_stats))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        manager.disconnect_stats(websocket)
    except Exception:
        manager.disconnect_stats(websocket)


@router.websocket("/ws/pipeline")
@router.websocket("/ws/layers")
async def websocket_pipeline(websocket: WebSocket):
    await manager.connect_pipeline(websocket)
    try:
        # Keep connection open, broadcasts happen asynchronously via engine_manager.broadcast_pipeline_update
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_pipeline(websocket)
    except Exception:
        manager.disconnect_pipeline(websocket)


@router.websocket("/ws/experts")
async def websocket_experts(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send expert cache stats every 2 seconds
            if engine_manager.loader:
                metadata = engine_manager.loader.expert_metadata
                expert_stats = []
                for slot, meta in list(metadata.items()):
                    key = meta["key"]
                    expert_stats.append({
                        "layer_id": key[0],
                        "expert_id": key[1],
                        "hits": meta["hits"],
                        "load_ms": round(meta["load_ms"], 2),
                        "vram_cost_kb": round(meta["vram_cost"] / 1024, 2)
                    })
                await websocket.send_text(json.dumps({
                    "type": "experts",
                    "experts": expert_stats
                }))
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
