"""WebSocket endpoint for compute nodes (browser tabs or native agents)."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request

from orchestrator.node_manager.assigner import compute_layer_assignment
from orchestrator.node_manager.registry import ConnectedNode
from orchestrator.protocol.messages import MessageType

router = APIRouter()
logger = logging.getLogger("orchestrator.nodes")


@router.websocket("/nodes/ws")
async def node_websocket(ws: WebSocket, request: Request):
    """WebSocket connection for compute nodes.

    Protocol:
    1. Node connects
    2. Node sends JSON: { type: "register", node_id, gpu_name, gpu_vram_mb, runtime }
    3. Orchestrator sends: { type: "layer_assignment", model_id, start_layer, end_layer }
    4. Node downloads weights, loads into GPU
    5. Node sends: { type: "layers_loaded", start_layer, end_layer }
    6. Node enters ready state, sends heartbeats every 5s
    7. Orchestrator sends prefill/decode requests when inference is needed
    """
    registry = request.app.state.registry
    await ws.accept()
    node_id = None

    try:
        # Step 1: Wait for registration message
        raw = await ws.receive_text()
        msg = json.loads(raw)

        if msg.get("type") != MessageType.REGISTER:
            await ws.send_json({"error": "First message must be type: register"})
            await ws.close()
            return

        node_id = msg["node_id"]

        # Step 2: Register node
        node = ConnectedNode(
            node_id=node_id,
            ws=ws,
            gpu_name=msg.get("gpu_name", "unknown"),
            gpu_vram_mb=msg.get("gpu_vram_mb", 0),
            runtime=msg.get("runtime", "webgpu"),
        )
        await registry.add(node)

        # Step 3: Assign layers
        start, end = compute_layer_assignment(node, registry)
        await ws.send_json({
            "type": MessageType.LAYER_ASSIGNMENT,
            "model_id": "llama-3-8b",
            "start_layer": start,
            "end_layer": end,
            "weight_shard_urls": [],  # TODO: real URLs
        })

        # Step 4: Main message loop
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            await _handle_node_message(node_id, msg, registry, request.app.state.session_manager)

    except WebSocketDisconnect:
        logger.info(f"Node {node_id[:8] if node_id else '?'} disconnected")
    except Exception as e:
        logger.error(f"Node {node_id[:8] if node_id else '?'} error: {e}")
    finally:
        if node_id:
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

    elif msg_type == MessageType.LAYERS_LOADED:
        start = msg["start_layer"]
        end = msg["end_layer"]
        await registry.set_layers(node_id, start, end)

    elif msg_type == MessageType.ACTIVATION_RESULT:
        session_id = msg.get("session_id")
        session = session_mgr.get(session_id)
        if session:
            # TODO: receive actual binary data and forward to next pipeline stage
            logger.debug(f"Activation from {node_id[:8]} for session {session_id}")

    elif msg_type == MessageType.ERROR:
        logger.error(f"Node {node_id[:8]} reported error: {msg.get('message')}")
