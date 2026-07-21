"""OpenAI-compatible /v1/chat/completions endpoint, served by whole-model nodes."""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from orchestrator.config import settings
from orchestrator.protocol import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    MessageType,
    SessionFailureCode,
    Usage,
)
from orchestrator.session import SessionFailure

logger = logging.getLogger("orchestrator.completions")

router = APIRouter()


def _error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "server_error", "code": code}},
    )


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    """Handle inference requests. OpenAI-compatible API."""
    session_mgr = request.app.state.session_manager
    node_router = request.app.state.router

    node = node_router.pick_node(req.model)
    if node is None:
        # Distinguish an empty fleet from a fleet serving something else -
        # substituting a different model would be worse than refusing.
        served = request.app.state.registry.served_models()
        if req.model and served:
            message = (
                f"No node is serving '{req.model}'. "
                f"Currently served: {', '.join(served)}."
            )
        else:
            message = (
                "No compute node is currently serving a model. "
                "Start one with: fleetlm up"
            )
        return _error(503, message, SessionFailureCode.NO_CAPACITY)

    session = session_mgr.create(
        model=node.model_id or req.model,
        node_id=node.node_id,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    node.active_sessions += 1

    try:
        await node.ws.send_json({
            "type": MessageType.GENERATE_REQUEST,
            "session_id": session.id,
            "messages": [m.model_dump() for m in req.messages],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        })
    except Exception as e:
        node.active_sessions -= 1
        session_mgr.remove(session.id)
        logger.error(f"Failed to dispatch to node {node.node_id[:8]}: {e}")
        return _error(502, "Failed to reach the serving node.", SessionFailureCode.NODE_ERROR)

    if req.stream:
        return EventSourceResponse(
            _stream_response(session, node, session_mgr, request.app.state.metrics),
            media_type="text/event-stream",
        )

    metrics = request.app.state.metrics
    try:
        parts: list[str] = []
        async for chunk in session.stream():
            parts.append(chunk)
        metrics.session_completed(
            node.node_id, session.prompt_tokens, session.completion_tokens
        )
    except SessionFailure as e:
        metrics.session_failed(node.node_id)
        return _error(502, f"Generation failed: {e}", SessionFailureCode.NODE_ERROR)
    finally:
        node.active_sessions = max(0, node.active_sessions - 1)
        session_mgr.remove(session.id)

    return ChatCompletionResponse(
        id=f"chatcmpl-{session.id}",
        model=session.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content="".join(parts)),
                finish_reason=session.finish_reason or "stop",
            )
        ],
        usage=Usage(
            prompt_tokens=session.prompt_tokens,
            completion_tokens=session.completion_tokens,
            total_tokens=session.prompt_tokens + session.completion_tokens,
        ),
    )


async def _stream_response(session, node, session_mgr, metrics):
    """Yield OpenAI-style chat.completion.chunk SSE events."""
    completion_id = f"chatcmpl-{session.id}"
    created = int(time.time())

    def chunk_payload(delta: dict, finish_reason: str | None = None) -> str:
        return json.dumps({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": session.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        })

    try:
        yield chunk_payload({"role": "assistant", "content": ""})
        async for text in session.stream():
            yield chunk_payload({"content": text})
        yield chunk_payload({}, finish_reason=session.finish_reason or "stop")
        metrics.session_completed(
            node.node_id, session.prompt_tokens, session.completion_tokens
        )
    except SessionFailure as e:
        metrics.session_failed(node.node_id)
        yield json.dumps({"error": {"message": str(e), "code": SessionFailureCode.NODE_ERROR}})
    finally:
        node.active_sessions = max(0, node.active_sessions - 1)
        session_mgr.remove(session.id)
    yield "[DONE]"


@router.get("/v1/models")
async def list_models(request: Request):
    registry = request.app.state.registry
    served = registry.served_models()
    if not served:
        served = [settings.default_model]
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "owned_by": "fleetlm"}
            for model in served
        ],
    }
