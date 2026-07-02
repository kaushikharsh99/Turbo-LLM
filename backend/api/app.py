from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import chat, models, settings, benchmark, metrics, logs
from backend.api import websocket

app = FastAPI(
    title="Turbo-LLM API Server",
    description="API Server for Turbo-LLM, exposing OpenAI compatible and runtime visualization endpoints.",
    version="1.0.0"
)

# CORS configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat.router, tags=["Chat"])
app.include_router(models.router, tags=["Models"])
app.include_router(settings.router, tags=["Settings"])
app.include_router(benchmark.router, tags=["Benchmark"])
app.include_router(metrics.router, tags=["Metrics"])
app.include_router(logs.router, tags=["Logs"])

# Include WebSocket routes
app.include_router(websocket.router, tags=["WebSockets"])

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Turbo-LLM API Server is running. Use /v1/chat/completions or /v1/models."
    }
