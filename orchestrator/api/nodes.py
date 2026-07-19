"""The one WebSocket a compute node opens, and everything that flows over it.

The node always dials out, so contributors need no open ports, no static IP,
and no firewall changes. Control, interactive generations, batch work units,
and heartbeats all share this single connection.

    node  -> register {node_id, model_id, gpu_name, gpu_vram_mb, join_token}
    orch  -> serve_model {model_id}
    node  -> model_loaded {model_id}                     ... now ready
    node  -> heartbeat {...} every 5s

  interactive:
    orch  -> generate_request {session_id, messages, ...}
    node  -> generate_chunk {session_id, text} ...
    node  -> generate_complete {session_id, finish_reason, usage}

  batch (node-paced pull):
    node  -> work_request {capacity}
    orch  -> work_assignment {units: [...]}
    node  -> work_result {unit_id, text, usage} | work_failed {unit_id, message}
"""

from __future__ import annotations

import hmac
import json
import logging

from fastapi import APIRouter, WebSocket

from orchestrator.config import settings
from orchestrator.fleet.registry import ConnectedNode
from orchestrator.protocol import MessageType

router = APIRouter()
logger = logging.getLogger("orchestrator.nodes")

MAX_LEASE_PER_REQUEST = 64


@router.websocket("/nodes/ws")
async def node_websocket(ws: WebSocket):
    registry = ws.app.state.registry
    session_mgr = ws.app.state.session_manager
    batch_store = ws.app.state.batch_store
    metrics = ws.app.state.metrics
    await ws.accept()
    node_id = None

    try:
        msg = json.loads(await ws.receive_text())

        if msg.get("type") != MessageType.REGISTER:
            await ws.send_json({"error": "First message must be type: register"})
            await ws.close()
            return

        # Constant-time compare, so the token can't be recovered by timing
        # repeated join attempts.
        if settings.join_token and not hmac.compare_digest(
            str(msg.get("join_token", "")), settings.join_token
        ):
            logger.warning(f"Rejected join from {msg.get('node_id', '?')[:8]}: bad token")
            await ws.send_json({"error": "invalid join token"})
            await ws.close(code=4401)
            return

        node_id = msg["node_id"]
        node = ConnectedNode(
            node_id=node_id,
            ws=ws,
            gpu_name=msg.get("gpu_name", "unknown"),
            gpu_vram_mb=msg.get("gpu_vram_mb", 0),
            runtime=msg.get("runtime", "native"),
            model_id=msg.get("model_id"),
        )
        await registry.add(node)
        metrics.node_joined(node_id, node.gpu_name)

        await ws.send_json({
            "type": MessageType.SERVE_MODEL,
            "model_id": node.model_id or settings.default_model,
        })

        while True:
            frame = await ws.receive()
            if frame["type"] == "websocket.disconnect":
                break
            if frame.get("text") is not None:
                await _handle(
                    json.loads(frame["text"]),
                    node_id, registry, session_mgr, batch_store, metrics, ws,
                )
            elif frame.get("bytes") is not None:
                logger.debug(
                    f"Ignoring unexpected binary frame "
                    f"({len(frame['bytes'])} bytes) from {node_id[:8]}"
                )

    except Exception as e:
        logger.error(f"Node {node_id[:8] if node_id else '?'} error: {e}")
    finally:
        if node_id:
            logger.info(f"Node {node_id[:8]} disconnected")
            session_mgr.fail_sessions_for_node(node_id)
            await batch_store.release_node(node_id)
            await registry.remove(node_id)
            metrics.node_left(node_id)


async def _handle(msg, node_id, registry, session_mgr, batch_store, metrics, ws):
    """Dispatch one message from a node."""
    msg_type = msg.get("type")

    if msg_type == MessageType.HEARTBEAT:
        await registry.update_heartbeat(
            node_id,
            cpu=msg.get("cpu_usage", 0),
            gpu=msg.get("gpu_usage", 0),
            sessions=msg.get("active_sessions", 0),
        )

    elif msg_type == MessageType.MODEL_LOADED:
        await registry.set_model_loaded(node_id, msg["model_id"])
        metrics.node_ready(node_id)

    elif msg_type == MessageType.GENERATE_CHUNK:
        session = session_mgr.get(msg.get("session_id"))
        if session:
            session.push_chunk(msg.get("text", ""))

    elif msg_type == MessageType.GENERATE_COMPLETE:
        session = session_mgr.get(msg.get("session_id"))
        if session:
            session.complete(
                finish_reason=msg.get("finish_reason", "stop"),
                prompt_tokens=msg.get("prompt_tokens", 0),
                completion_tokens=msg.get("completion_tokens", 0),
            )

    elif msg_type == MessageType.GENERATE_ERROR:
        session = session_mgr.get(msg.get("session_id"))
        if session:
            logger.error(f"Node {node_id[:8]} generation error: {msg.get('message')}")
            session.fail(msg.get("message", "node generation error"))

    elif msg_type == MessageType.WORK_REQUEST:
        # Nodes pull: each asks for as much as it has room for.
        capacity = max(0, min(int(msg.get("capacity", 1)), MAX_LEASE_PER_REQUEST))
        units = await batch_store.lease(node_id, capacity) if capacity else []
        await ws.send_json({
            "type": MessageType.WORK_ASSIGNMENT,
            "units": [u.payload() for u in units],
        })

    elif msg_type == MessageType.WORK_RESULT:
        await batch_store.complete(
            msg["unit_id"],
            node_id,
            msg.get("text", ""),
            prompt_tokens=msg.get("prompt_tokens", 0),
            completion_tokens=msg.get("completion_tokens", 0),
            generation_sec=msg.get("generation_sec", 0.0),
        )

    elif msg_type == MessageType.WORK_FAILED:
        await batch_store.fail(
            msg["unit_id"], node_id, msg.get("message", "node reported failure")
        )

    elif msg_type == MessageType.ERROR:
        logger.error(f"Node {node_id[:8]} reported error: {msg.get('message')}")

    else:
        logger.warning(f"Unknown message type from {node_id[:8]}: {msg_type}")
