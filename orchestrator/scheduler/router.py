"""Routes inference requests to the best available pipeline."""

from __future__ import annotations

import logging

from orchestrator.node_manager.registry import ConnectedNode, NodeRegistry
from orchestrator.session.manager import InferenceSession

logger = logging.getLogger("orchestrator.router")


class PipelineRouter:
    def __init__(self, registry: NodeRegistry):
        self.registry = registry

    def find_generation_node(self, model: str | None = None) -> ConnectedNode | None:
        """Pick the least-loaded ready whole-model node (preferring exact model match)."""
        candidates = self.registry.get_generation_nodes(model)
        if not candidates:
            return None
        node = min(candidates, key=lambda n: n.active_sessions)
        logger.info(
            f"Routing to node {node.node_id[:8]} "
            f"(model={node.model_id}, active={node.active_sessions})"
        )
        return node

    def find_route(self, session: InferenceSession) -> list[ConnectedNode] | None:
        """Find a pipeline of layer-shard nodes covering all layers (Phase 5 path).

        Returns an ordered list of nodes covering all layers, or None.
        """
        pipeline = self.registry.find_pipeline()

        if pipeline:
            node_ids = [n.node_id[:8] for n in pipeline]
            layers = [n.layers for n in pipeline]
            logger.info(f"Route found for session {session.id}: {list(zip(node_ids, layers))}")

        return pipeline
