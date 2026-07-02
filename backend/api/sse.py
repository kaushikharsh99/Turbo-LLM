# Server-Sent Events utilities for Turbo-LLM
import json

def format_sse(data: dict, event: str = None) -> str:
    """Format data into a server-sent event string."""
    formatted = f"data: {json.dumps(data)}\n\n"
    if event:
        formatted = f"event: {event}\n" + formatted
    return formatted
