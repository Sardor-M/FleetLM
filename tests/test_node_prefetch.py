"""Keeping the next lease in flight while the current batch decodes.

A node used to ask for more work only after a batch finished, so it stood idle
for a full orchestrator round-trip between every batch with the GPU doing
nothing. It now asks before decoding. Two bounds keep that safe: only one
request outstanding at a time, and never more queued than one batch's worth.
"""

import asyncio
import contextlib
import time

from node_agent.__main__ import NodeAgent
from node_agent.engine import MockEngine

MODEL = "tiny-test-model"


def _agent(batch_size=4, engine=None):
    engine = engine or MockEngine()
    engine.load(MODEL)
    agent = NodeAgent("ws://test/nodes/ws", engine, MODEL, batch_size=batch_size)
    agent.model_loaded = True
    return agent


def _drain(agent):
    return [agent.outbox.get_nowait() for _ in range(agent.outbox.qsize())]


def _unit(i):
    return {
        "unit_id": f"unit-{i}",
        "messages": [{"role": "user", "content": f"q{i}"}],
        "max_tokens": 8,
        "temperature": 0.0,
    }


# ── how much a node may ask for ─────────────────────────────────────────

def test_a_full_backlog_asks_for_nothing():
    agent = _agent(batch_size=2)
    agent.work_queue.put_nowait(_unit(0))
    agent.work_queue.put_nowait(_unit(1))

    agent._request_work()

    assert _drain(agent) == []


def test_capacity_asked_for_is_the_room_left_in_the_backlog():
    agent = _agent(batch_size=4)
    agent.work_queue.put_nowait(_unit(0))

    agent._request_work()

    assert _drain(agent)[0]["capacity"] == 3


def test_a_node_with_no_model_loaded_asks_for_nothing():
    agent = _agent()
    agent.model_loaded = False

    agent._request_work()

    assert _drain(agent) == []


# ── only one request in flight ──────────────────────────────────────────

def test_requests_do_not_stack():
    """Spurious calls are expected; they must not multiply the leases held."""
    agent = _agent(batch_size=4)

    agent._request_work()
    agent._request_work()
    agent._request_work()

    assert len(_drain(agent)) == 1


def test_an_assignment_clears_the_way_for_the_next_request():
    agent = _agent(batch_size=4)
    agent._request_work()
    _drain(agent)

    agent.work_request_inflight = False  # what receiving an assignment does
    agent._request_work()

    assert len(_drain(agent)) == 1


def test_a_disconnect_clears_the_outstanding_request():
    """Its assignment died with the socket, so the node must be free to re-ask."""
    agent = _agent()
    agent._request_work()
    assert agent.work_request_inflight is True

    agent._drain_work_queue()

    assert agent.work_request_inflight is False


# ── the prefetch itself ─────────────────────────────────────────────────

class SlowEngine(MockEngine):
    """Decodes slowly enough that the prefetch is observable mid-batch."""

    def generate_batch(self, items):
        time.sleep(0.25)
        return super().generate_batch(items)


async def test_the_next_lease_is_requested_before_the_batch_finishes():
    agent = _agent(batch_size=4, engine=SlowEngine())
    agent.work_queue.put_nowait(_unit(0))

    loop = asyncio.create_task(agent._work_loop())
    await asyncio.sleep(0.05)
    mid_batch = [m["type"] for m in _drain(agent)]

    loop.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await loop

    assert "work_request" in mid_batch, "the next lease was not requested during decode"
    assert "work_result" not in mid_batch, "the batch finished before the check ran"
