"""A request for one model must never be answered by another.

With a fleet the operator configured by hand this was a harmless convenience.
Once nodes choose their own models - which is exactly what easy model
management encourages - substituting a different model returns a
plausible-looking answer computed by the wrong weights, with nothing in the
response saying so. These pin both halves: interactive routing, and which
units a node is allowed to lease.
"""

import pytest
from fastapi.testclient import TestClient

from orchestrator.batch import BatchStore
from orchestrator.fleet.registry import ConnectedNode, NodeRegistry
from orchestrator.fleet.router import Router
from orchestrator.main import app
from orchestrator.protocol import NodeStatus


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _node(node_id, model, active=0):
    node = ConnectedNode(node_id=node_id, ws=None, model_id=model)
    node.status = NodeStatus.READY
    node.active_sessions = active
    return node


async def _registry(*nodes):
    registry = NodeRegistry()
    for n in nodes:
        await registry.add(n)
    return registry


# ── interactive routing ─────────────────────────────────────────────────

async def test_a_request_is_never_routed_to_a_node_serving_another_model():
    """The regression: this used to fall back and answer with the wrong weights."""
    registry = await _registry(_node("n1", "phi4-mini"))

    assert Router(registry).pick_node("llama3.2") is None


async def test_routing_still_finds_a_node_that_does_serve_the_model():
    registry = await _registry(_node("wrong", "phi4-mini"), _node("right", "llama3.2"))

    assert Router(registry).pick_node("llama3.2").node_id == "right"


async def test_a_busy_matching_node_beats_an_idle_mismatched_one():
    registry = await _registry(
        _node("idle-wrong", "phi4-mini", active=0),
        _node("busy-right", "llama3.2", active=9),
    )

    assert Router(registry).pick_node("llama3.2").node_id == "busy-right"


async def test_naming_no_model_still_accepts_any_ready_node():
    registry = await _registry(_node("n1", "phi4-mini"))

    assert Router(registry).pick_node(None) is not None


def test_asking_for_an_unserved_model_says_what_is_served(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "no_capacity"


# ── batch leasing ───────────────────────────────────────────────────────

async def _batch(store, model, n=2):
    return await store.create_batch(
        [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(n)], model
    )


async def test_a_node_cannot_lease_units_for_a_model_it_does_not_serve():
    """Worse than the routing half: leasing had no model check at all."""
    store = BatchStore()
    await _batch(store, "llama3.2", n=2)

    assert await store.lease("n1", 5, node_model="phi4-mini") == []


async def test_a_node_leases_only_the_units_it_can_serve():
    store = BatchStore()
    await _batch(store, "llama3.2", n=2)
    await _batch(store, "phi4-mini", n=3)

    leased = await store.lease("n1", 10, node_model="phi4-mini")

    assert len(leased) == 3
    assert {u.model for u in leased} == {"phi4-mini"}


async def test_units_a_node_skips_stay_queued_for_someone_else():
    """Skipped units must keep their place, not be dropped off the queue."""
    store = BatchStore()
    await _batch(store, "llama3.2", n=2)

    assert await store.lease("wrong", 10, node_model="phi4-mini") == []
    leased = await store.lease("right", 10, node_model="llama3.2")

    assert len(leased) == 2


async def test_a_batch_with_no_model_can_run_anywhere():
    store = BatchStore()
    await _batch(store, None, n=2)

    assert len(await store.lease("n1", 10, node_model="phi4-mini")) == 2


async def test_a_node_reporting_no_model_is_not_blocked():
    """Older nodes predate model reporting; they must not deadlock the queue."""
    store = BatchStore()
    await _batch(store, "llama3.2", n=2)

    assert len(await store.lease("n1", 10, node_model=None)) == 2


async def test_fifo_order_survives_a_skip():
    store = BatchStore()
    await _batch(store, "llama3.2", n=3)

    await store.lease("wrong", 10, node_model="phi4-mini")  # skips all three
    leased = await store.lease("right", 10, node_model="llama3.2")

    assert [u.index for u in leased] == [0, 1, 2]


# ── provenance on the result ────────────────────────────────────────────

async def test_a_result_records_the_model_that_actually_ran():
    store = BatchStore()
    batch = await _batch(store, "llama3.2", n=1)
    unit = (await store.lease("n1", 1, node_model="llama3.2"))[0]

    await store.complete(unit.id, "n1", "hi", served_by="llama3.2:latest")

    record = store.results(batch.id)[0]
    assert record["model"] == "llama3.2:latest"


async def test_a_node_that_reports_no_model_falls_back_to_what_was_asked():
    store = BatchStore()
    batch = await _batch(store, "llama3.2", n=1)
    unit = (await store.lease("n1", 1))[0]

    await store.complete(unit.id, "n1", "hi")

    assert store.results(batch.id)[0]["model"] == "llama3.2"


async def test_a_unit_inherits_the_batch_model_when_the_key_is_present_but_null():
    """The API model_dumps a request whose `model` key exists and is None.

    A .get default never fires in that case, so units silently lost the
    batch's model and every model filter downstream became a no-op. Unit
    tests that hand-build dicts without the key cannot catch this.
    """
    store = BatchStore()
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": "q"}], "model": None,
          "max_tokens": 256, "temperature": 0.7}],
        "llama3.2",
    )

    unit = store.units[batch.unit_ids[0]]
    assert unit.model == "llama3.2"
    assert await store.lease("n1", 1, node_model="phi4-mini") == []
