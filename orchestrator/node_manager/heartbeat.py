"""Background task that monitors node health via heartbeats."""

from __future__ import annotations

import asyncio
import logging

from orchestrator.config import settings
from orchestrator.node_manager.registry import NodeRegistry

logger = logging.getLogger("orchestrator.heartbeat")


async def heartbeat_monitor(registry: NodeRegistry) -> None:
    """Runs forever, checking for stale nodes every few seconds."""
    logger.info("Heartbeat monitor started")

    while True:
        await asyncio.sleep(settings.heartbeat_timeout_sec // 2 or 5)

        stale = registry.get_stale_nodes()
        for node_id in stale:
            node = registry.get_node(node_id)
            if node:
                logger.warning(f"Node {node_id[:8]} missed heartbeat, marking offline")
                try:
                    await node.ws.close()
                except Exception:
                    pass
                await registry.remove(node_id)
