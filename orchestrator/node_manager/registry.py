"""Tracks all connected compute nodes and their state."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from fastapi import WebSocket

from orchestrator.config import settings
from orchestrator.protocol.messages import NodeStatus

logger = logging.getLogger("orchestrator.registry")


@dataclass
class ConnectedNode:
    node_id: str
    ws: WebSocket
    gpu_name: str = "unknown"
    gpu_vram_mb: int = 0
    runtime: str = "webgpu"
    status: NodeStatus = NodeStatus.REGISTERING
    layers: tuple[int, int] | None = None
    last_heartbeat: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    active_sessions: int = 0


class NodeRegistry:
    """Thread-safe registry of all connected compute nodes."""

    def __init__(self):
        self.nodes: dict[str, ConnectedNode] = {}
        self._lock = asyncio.Lock()

    async def add(self, node: ConnectedNode) -> None:
        async with self._lock:
            self.nodes[node.node_id] = node
            logger.info(
                f"Node registered: {node.node_id} "
                f"(gpu={node.gpu_name}, vram={node.gpu_vram_mb}MB, runtime={node.runtime})"
            )

    async def remove(self, node_id: str) -> None:
        async with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Node removed: {node_id}")

    async def update_heartbeat(self, node_id: str, cpu: float, gpu: float, sessions: int) -> None:
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.last_heartbeat = time.time()
            node.cpu_usage = cpu
            node.gpu_usage = gpu
            node.active_sessions = sessions

    async def set_layers(self, node_id: str, start: int, end: int) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].layers = (start, end)
            self.nodes[node_id].status = NodeStatus.READY
            logger.info(f"Node {node_id} ready with layers {start}-{end}")

    def get_ready_nodes(self) -> list[ConnectedNode]:
        return [n for n in self.nodes.values() if n.status == NodeStatus.READY]

    def get_node(self, node_id: str) -> ConnectedNode | None:
        return self.nodes.get(node_id)

    def find_pipeline(self, total_layers: int = 32) -> list[ConnectedNode] | None:
        """Find a complete set of nodes that covers all layers in order.

        Returns ordered list of nodes forming a pipeline, or None if no
        complete coverage exists.
        """
        ready = self.get_ready_nodes()
        if not ready:
            return None

        # Sort by start layer
        with_layers = [n for n in ready if n.layers is not None]
        with_layers.sort(key=lambda n: n.layers[0])

        # Greedy: find a chain that covers [0, total_layers)
        pipeline = []
        covered_up_to = 0

        for node in with_layers:
            start, end = node.layers
            if start <= covered_up_to and end > covered_up_to:
                pipeline.append(node)
                covered_up_to = end + 1
                if covered_up_to >= total_layers:
                    return pipeline

        return None  # incomplete coverage

    def get_stale_nodes(self, timeout: int | None = None) -> list[str]:
        """Return node IDs that haven't sent a heartbeat within timeout."""
        timeout = timeout or settings.heartbeat_timeout_sec
        now = time.time()
        return [
            nid for nid, node in self.nodes.items()
            if now - node.last_heartbeat > timeout
        ]

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
                    "status": n.status.value,
                    "layers": n.layers,
                    "cpu": round(n.cpu_usage, 1),
                    "gpu_usage": round(n.gpu_usage, 1),
                }
                for n in self.nodes.values()
            ],
        }
