"""OpenAI-compatible /v1/chat/completions endpoint."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from orchestrator.protocol.messages import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Usage,
)

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    """Handle B2B inference requests. OpenAI-compatible API."""
    registry = request.app.state.registry
    session_mgr = request.app.state.session_manager
    pipeline_router = request.app.state.router

    # Create a session
    session = session_mgr.create(
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    # Try to find a pipeline route
    pipeline = pipeline_router.find_route(session)

    if not pipeline:
        session_mgr.remove(session.id)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": f"No complete pipeline available. "
                    f"{registry.ready_count} nodes ready but cannot cover all layers.",
                    "type": "server_error",
                    "code": "no_pipeline",
                }
            },
        )

    # TODO Phase 2: Actually route through the pipeline
    # For now, return a placeholder showing the pipeline we found
    pipeline_desc = " -> ".join(
        f"Node({n.node_id[:8]}, L{n.layers[0]}-{n.layers[1]})"
        for n in pipeline
    )

    response = ChatCompletionResponse(
        id=f"chatcmpl-{session.id}",
        model=req.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(
                    role="assistant",
                    content=(
                        f"[POC] Pipeline found: {pipeline_desc}. "
                        f"Actual inference coming in Phase 2."
                    ),
                )
            )
        ],
        usage=Usage(
            prompt_tokens=sum(len(m.content.split()) for m in req.messages),
        ),
    )

    session_mgr.remove(session.id)
    return response


@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "llama-3-8b", "object": "model", "owned_by": "distributed-llm"},
        ],
    }
