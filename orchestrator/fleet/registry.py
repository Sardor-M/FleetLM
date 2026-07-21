"""Who is on the fleet right now.

Soft state only: every node here is one that currently holds an open outbound
WebSocket. A node that vanishes is simply forgotten - its in-flight work is
recovered by the batch store's lease accounting, not by anything here.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from fastapi import WebSocket

from orchestrator.config import settings
from orchestrator.protocol import NodeStatus

logger = logging.getLogger("orchestrator.registry")


@dataclass
class ConnectedNode:
    node_id: str
    ws: WebSocket
    gpu_name: str = "unknown"
    gpu_vram_mb: int = 0
    runtime: str = "native"
    model_id: str | None = None
    status: NodeStatus = NodeStatus.REGISTERING
    last_heartbeat: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    active_sessions: int = 0


class NodeRegistry:
    def __init__(self):
        self.nodes: dict[str, ConnectedNode] = {}
        self._lock = asyncio.Lock()

    async def add(self, node: ConnectedNode) -> None:
        async with self._lock:
            self.nodes[node.node_id] = node
        logger.info(
            f"Node registered: {node.node_id[:8]} "
            f"(gpu={node.gpu_name}, mem={node.gpu_vram_mb}MB, runtime={node.runtime})"
        )

    async def remove(self, node_id: str) -> None:
        async with self._lock:
            self.nodes.pop(node_id, None)
        logger.info(f"Node removed: {node_id[:8]}")

    async def update_heartbeat(
        self, node_id: str, cpu: float, gpu: float, sessions: int
    ) -> None:
        node = self.nodes.get(node_id)
        if node:
            node.last_heartbeat = time.time()
            node.cpu_usage = cpu
            node.gpu_usage = gpu
            node.active_sessions = sessions

    async def set_model_loaded(self, node_id: str, model_id: str) -> None:
        node = self.nodes.get(node_id)
        if node:
            node.model_id = model_id
            node.status = NodeStatus.READY
            logger.info(f"Node {node_id[:8]} ready, serving {model_id}")

    def get_node(self, node_id: str) -> ConnectedNode | None:
        return self.nodes.get(node_id)

    def get_ready_nodes(self, model: str | None = None) -> list[ConnectedNode]:
        """Ready nodes serving `model`, or every ready node if none is named.

        Strict on purpose. This used to fall back to any ready node when no
        node served the requested model, which silently answered a request for
        one model with another model's weights: same response shape, different
        results, no error anywhere to notice. Callers that genuinely do not
        care which model runs simply pass none.
        """
        ready = [n for n in self.nodes.values() if n.status == NodeStatus.READY]
        if model:
            return [n for n in ready if n.model_id == model]
        return ready

    def get_stale_nodes(self, timeout: int | None = None) -> list[str]:
        """Node IDs that have missed heartbeats for longer than the timeout."""
        timeout = timeout or settings.heartbeat_timeout_sec
        now = time.time()
        return [
            nid for nid, node in self.nodes.items()
            if now - node.last_heartbeat > timeout
        ]

    def served_models(self) -> list[str]:
        return sorted({n.model_id for n in self.get_ready_nodes() if n.model_id})

    @property
    def count(self) -> int:
        return len(self.nodes)

    @property
    def ready_count(self) -> int:
        return len(self.get_ready_nodes())

    def summary(self) -> dict:
        return {
            "total_nodes": self.count,
            "ready_nodes": self.ready_count,
            "nodes": [
                {
                    "id": n.node_id[:8],
                    "gpu": n.gpu_name,
                    "vram_mb": n.gpu_vram_mb,
                    "runtime": n.runtime,
                    "model": n.model_id,
                    "status": n.status.value,
                    "cpu": round(n.cpu_usage, 1),
                    "gpu_usage": round(n.gpu_usage, 1),
                    "active": n.active_sessions,
                }
                for n in self.nodes.values()
            ],
        }
