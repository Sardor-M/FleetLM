"""Routes inference requests to the best available pipeline."""

from __future__ import annotations

import logging

from orchestrator.node_manager.registry import ConnectedNode, NodeRegistry
from orchestrator.session.manager import InferenceSession

logger = logging.getLogger("orchestrator.router")


class PipelineRouter:
    def __init__(self, registry: NodeRegistry):
        self.registry = registry

    def find_route(self, session: InferenceSession) -> list[ConnectedNode] | None:
        """Find the best pipeline of nodes to handle this session.

        Returns an ordered list of nodes covering all layers, or None.
        """
        pipeline = self.registry.find_pipeline()

        if pipeline:
            node_ids = [n.node_id[:8] for n in pipeline]
            layers = [n.layers for n in pipeline]
            logger.info(f"Route found for session {session.id}: {list(zip(node_ids, layers))}")
            session.pipeline_node_ids = [n.node_id for n in pipeline]

        return pipeline
