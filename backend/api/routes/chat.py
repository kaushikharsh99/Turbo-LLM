import json
import time
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from backend.api.engine_manager import engine_manager

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 128
    stream: Optional[bool] = False
    thinking: Optional[str] = "off"  # "on" or "off"

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 128
    stream: Optional[bool] = False

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    chat: Optional[bool] = False
    system_prompt: Optional[str] = None
    thinking: Optional[str] = "off"

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not engine_manager.engine:
        raise HTTPException(status_code=400, detail="No model loaded. Please load a model first using /v1/load.")

    # Extract user prompt and system prompt if any
    system_prompt = None
    user_prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_prompt = msg.content

    # Check if streaming
    if request.stream:
        async def event_generator():
            request_id = f"chatcmpl-{uuid.uuid4()}"
            created_time = int(time.time())
            
            for update in engine_manager.generate_stream(
                prompt=user_prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                chat=True,
                system_prompt=system_prompt,
                thinking=request.thinking
            ):
                if update["type"] == "token":
                    delta = {"content": update["token"]}
                    # OpenAI chunk format
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": delta,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif update["type"] == "error":
                    yield f"data: {json.dumps({'error': update['error']})}\n\n"
                elif update["type"] == "done":
                    # Send finish chunk
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        # Non-streaming OpenAI format
        tokens_generator = engine_manager.generate_stream(
            prompt=user_prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            chat=True,
            system_prompt=system_prompt,
            thinking=request.thinking
        )
        
        full_text = ""
        last_metrics = None
        for update in tokens_generator:
            if update["type"] == "token":
                full_text += update["token"]
            elif update["type"] == "error":
                raise HTTPException(status_code=500, detail=update["error"])
            elif update["type"] == "done":
                last_metrics = engine_manager.last_metrics

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,  # placeholder
                "completion_tokens": last_metrics.get("step", 0) if last_metrics else 0,
                "total_tokens": last_metrics.get("step", 0) if last_metrics else 0
            },
            "metrics": last_metrics
        }

@router.post("/v1/completions")
async def completions(request: CompletionRequest):
    if not engine_manager.engine:
        raise HTTPException(status_code=400, detail="No model loaded.")

    if request.stream:
        async def event_generator():
            request_id = f"cmpl-{uuid.uuid4()}"
            created_time = int(time.time())
            
            for update in engine_manager.generate_stream(
                prompt=request.prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                chat=False
            ):
                if update["type"] == "token":
                    chunk = {
                        "id": request_id,
                        "object": "text_completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [{
                            "text": update["token"],
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif update["type"] == "done":
                    chunk = {
                        "id": request_id,
                        "object": "text_completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [{
                            "text": "",
                            "index": 0,
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        tokens_generator = engine_manager.generate_stream(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            chat=False
        )
        full_text = ""
        for update in tokens_generator:
            if update["type"] == "token":
                full_text += update["token"]
                
        return {
            "id": f"cmpl-{uuid.uuid4()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "text": full_text,
                "index": 0,
                "finish_reason": "stop"
            }]
        }

@router.post("/v1/generate")
async def generate(request: GenerateRequest):
    if not engine_manager.engine:
        raise HTTPException(status_code=400, detail="No model loaded.")
    
    # Custom Turbo-LLM endpoint (returns raw streaming)
    async def raw_generator():
        for update in engine_manager.generate_stream(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            chat=request.chat,
            system_prompt=request.system_prompt,
            thinking=request.thinking
        ):
            yield json.dumps(update) + "\n"

    return StreamingResponse(raw_generator(), media_type="application/x-ndjson")
