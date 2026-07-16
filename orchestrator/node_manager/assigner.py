"""Assigns transformer layers to nodes based on their GPU capabilities."""

from __future__ import annotations

import logging

from orchestrator.config import settings
from orchestrator.node_manager.registry import ConnectedNode, NodeRegistry

logger = logging.getLogger("orchestrator.assigner")


def compute_layer_assignment(
    node: ConnectedNode,
    registry: NodeRegistry,
    total_layers: int | None = None,
) -> tuple[int, int]:
    """Determine which layers a newly registered node should serve.

    Strategy (simple for POC):
    - Find the first gap in layer coverage
    - Assign a chunk proportional to the node's VRAM
    - Default: split evenly among expected node count
    """
    total_layers = total_layers or settings.total_layers

    # Find which layers are already covered
    covered = set()
    for n in registry.get_ready_nodes():
        if n.layers:
            for l in range(n.layers[0], n.layers[1] + 1):
                covered.add(l)

    # Find first uncovered layer
    start = 0
    for l in range(total_layers):
        if l not in covered:
            start = l
            break
    else:
        # All layers covered, assign the second half as a redundant backup
        start = total_layers // 2

    # How many layers to assign based on VRAM
    # Rough estimate: ~160MB per layer for 8B Q4 model
    mb_per_layer = 160
    if node.gpu_vram_mb > 0:
        max_layers = max(4, node.gpu_vram_mb // mb_per_layer)
    else:
        # CPU-only node, give it fewer layers
        max_layers = 8

    end = min(start + max_layers - 1, total_layers - 1)

    logger.info(
        f"Assigning layers {start}-{end} to node {node.node_id[:8]} "
        f"(vram={node.gpu_vram_mb}MB, max_layers={max_layers})"
    )
    return (start, end)
