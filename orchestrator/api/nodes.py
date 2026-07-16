"""WebSocket endpoint for compute nodes (native agents or browser tabs)."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket

from orchestrator.node_manager.assigner import compute_layer_assignment
from orchestrator.node_manager.registry import ConnectedNode
from orchestrator.protocol.messages import MessageType, NodeMode

router = APIRouter()
logger = logging.getLogger("orchestrator.nodes")


@router.websocket("/nodes/ws")
async def node_websocket(ws: WebSocket):
    """WebSocket connection for compute nodes.

    Protocol (whole_model mode — Phase 1):
    1. Node connects and sends { type: "register", node_id, mode: "whole_model",
       model_id, gpu_name, gpu_vram_mb, runtime }
    2. Orchestrator replies { type: "serve_model", model_id }
    3. Node loads the model, then sends { type: "model_loaded", model_id }
    4. Node is ready; heartbeats every 5s
    5. Orchestrator sends { type: "generate_request", session_id, messages, ... }
    6. Node streams { type: "generate_chunk", session_id, text } messages,
       then { type: "generate_complete", session_id, finish_reason, usage... }

    Protocol (layer_shard mode — legacy browser path, pipeline inference not
    yet implemented): registration is answered with a layer_assignment and the
    node is tracked, but no inference is routed to it.
    """
    registry = ws.app.state.registry
    session_mgr = ws.app.state.session_manager
    await ws.accept()
    node_id = None

    try:
        # Step 1: registration message
        raw = await ws.receive_text()
        msg = json.loads(raw)

        if msg.get("type") != MessageType.REGISTER:
            await ws.send_json({"error": "First message must be type: register"})
            await ws.close()
            return

        node_id = msg["node_id"]
        node = ConnectedNode(
            node_id=node_id,
            ws=ws,
            gpu_name=msg.get("gpu_name", "unknown"),
            gpu_vram_mb=msg.get("gpu_vram_mb", 0),
            runtime=msg.get("runtime", "webgpu"),
            mode=msg.get("mode", NodeMode.LAYER_SHARD),
            model_id=msg.get("model_id"),
        )
        await registry.add(node)

        # Step 2: assignment
        if node.mode == NodeMode.WHOLE_MODEL:
            await ws.send_json({
                "type": MessageType.SERVE_MODEL,
                "model_id": node.model_id or "default",
            })
        else:
            start, end = compute_layer_assignment(node, registry)
            await ws.send_json({
                "type": MessageType.LAYER_ASSIGNMENT,
                "model_id": "llama-3-8b",
                "start_layer": start,
                "end_layer": end,
                "weight_shard_urls": [],  # Phase 5: real shard URLs
            })

        # Step 3: main message loop (text frames are JSON; binary frames are
        # activation tensors on the layer-shard path, ignored in Phase 1)
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text") is not None:
                msg = json.loads(message["text"])
                await _handle_node_message(node_id, msg, registry, session_mgr)
            elif message.get("bytes") is not None:
                logger.debug(
                    f"Ignoring binary frame ({len(message['bytes'])} bytes) "
                    f"from node {node_id[:8]}"
                )

    except Exception as e:
        logger.error(f"Node {node_id[:8] if node_id else '?'} error: {e}")
    finally:
        if node_id:
            logger.info(f"Node {node_id[:8]} disconnected")
            session_mgr.fail_sessions_for_node(node_id)
            await registry.remove(node_id)


async def _handle_node_message(node_id: str, msg: dict, registry, session_mgr):
    """Dispatch incoming node messages."""
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

    elif msg_type == MessageType.LAYERS_LOADED:
        await registry.set_layers(node_id, msg["start_layer"], msg["end_layer"])

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

    elif msg_type == MessageType.ACTIVATION_RESULT:
        logger.debug(f"Activation result from {node_id[:8]} (layer-shard path, ignored)")

    elif msg_type == MessageType.ERROR:
        logger.error(f"Node {node_id[:8]} reported error: {msg.get('message')}")
