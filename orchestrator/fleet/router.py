"""Picking a node for an interactive request.

Every ready node holds the whole model, so nodes are interchangeable and
routing is just load balancing. Batch work does not come through here: nodes
pull work units themselves, at a rate they choose (see orchestrator/batch.py).
"""

from __future__ import annotations

import logging

from orchestrator.fleet.registry import ConnectedNode, NodeRegistry

logger = logging.getLogger("orchestrator.router")


class Router:
    def __init__(self, registry: NodeRegistry):
        self.registry = registry

    def pick_node(self, model: str | None = None) -> ConnectedNode | None:
        """Least-loaded ready node, preferring an exact model match."""
        candidates = self.registry.get_ready_nodes(model)
        if not candidates:
            return None
        node = min(candidates, key=lambda n: n.active_sessions)
        logger.info(
            f"Routing to {node.node_id[:8]} "
            f"(model={node.model_id}, active={node.active_sessions})"
        )
        return node
