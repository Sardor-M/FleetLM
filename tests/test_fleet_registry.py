"""Fleet membership and routing.

The registry is soft state: it knows only which nodes currently hold an open
socket, and forgets anything that leaves. Routing on top of it is pure load
balancing, because every ready node holds the whole model and any of them can
serve any request.
"""

import time

from orchestrator.fleet.registry import ConnectedNode, NodeRegistry
from orchestrator.fleet.router import Router
from orchestrator.protocol import NodeStatus


def _node(node_id, model="tiny-test-model", status=NodeStatus.READY, active=0, hb=None):
    node = ConnectedNode(node_id=node_id, ws=None, model_id=model)
    node.status = status
    node.active_sessions = active
    if hb is not None:
        node.last_heartbeat = hb
    return node


# ── membership ──────────────────────────────────────────────────────────

async def test_node_is_servable_only_after_its_model_loads():
    registry = NodeRegistry()
    await registry.add(_node("n1", status=NodeStatus.REGISTERING))
    assert registry.count == 1
    assert registry.ready_count == 0

    await registry.set_model_loaded("n1", "tiny-test-model")

    assert registry.ready_count == 1
    assert registry.get_node("n1").status == NodeStatus.READY


async def test_a_removed_node_is_forgotten_entirely():
    registry = NodeRegistry()
    await registry.add(_node("n1"))
    await registry.remove("n1")

    assert registry.count == 0
    assert registry.get_node("n1") is None


async def test_removing_an_unknown_node_is_harmless():
    """Disconnect cleanup runs even for a node that never finished registering."""
    registry = NodeRegistry()
    await registry.remove("never-existed")
    assert registry.count == 0


async def test_stale_nodes_are_those_past_the_heartbeat_timeout():
    registry = NodeRegistry()
    await registry.add(_node("fresh", hb=time.time()))
    await registry.add(_node("silent", hb=time.time() - 999))

    assert registry.get_stale_nodes(timeout=30) == ["silent"]


async def test_heartbeat_refreshes_the_load_figures_routing_reads():
    registry = NodeRegistry()
    await registry.add(_node("n1"))
    await registry.update_heartbeat("n1", cpu=42.0, gpu=7.0, sessions=3)

    node = registry.get_node("n1")
    assert node.cpu_usage == 42.0
    assert node.active_sessions == 3


# ── which nodes can serve what ──────────────────────────────────────────

async def test_an_exact_model_match_narrows_the_candidates():
    registry = NodeRegistry()
    await registry.add(_node("llama", model="llama"))
    await registry.add(_node("qwen", model="qwen"))

    assert [n.node_id for n in registry.get_ready_nodes("qwen")] == ["qwen"]


async def test_an_unserved_model_falls_back_to_the_whole_fleet():
    """Serving from a different model beats refusing the request outright."""
    registry = NodeRegistry()
    await registry.add(_node("llama", model="llama"))

    assert len(registry.get_ready_nodes("a-model-nobody-serves")) == 1


async def test_a_node_still_loading_is_never_a_candidate():
    registry = NodeRegistry()
    await registry.add(_node("warming", status=NodeStatus.REGISTERING))

    assert registry.get_ready_nodes() == []


async def test_served_models_are_deduped_and_sorted():
    registry = NodeRegistry()
    await registry.add(_node("a", model="zeta"))
    await registry.add(_node("b", model="alpha"))
    await registry.add(_node("c", model="alpha"))

    assert registry.served_models() == ["alpha", "zeta"]


# ── routing ─────────────────────────────────────────────────────────────

async def test_router_picks_the_least_loaded_node():
    registry = NodeRegistry()
    await registry.add(_node("busy", active=5))
    await registry.add(_node("idle", active=0))

    assert Router(registry).pick_node().node_id == "idle"


async def test_model_match_outranks_a_lighter_node():
    """Load only breaks ties inside the set that already serves the model."""
    registry = NodeRegistry()
    await registry.add(_node("idle-wrong-model", model="llama", active=0))
    await registry.add(_node("busy-right-model", model="qwen", active=9))

    assert Router(registry).pick_node("qwen").node_id == "busy-right-model"


async def test_router_returns_nothing_when_no_node_is_ready():
    registry = NodeRegistry()
    await registry.add(_node("warming", status=NodeStatus.REGISTERING))

    assert Router(registry).pick_node() is None
