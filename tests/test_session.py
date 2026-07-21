"""Interactive session lifecycle.

A session is the orchestrator's view of one in-flight generation. What matters
is that it ends exactly once, and that every way a node can let it down (an
error, silence, a disconnect) surfaces as a failure the API layer can turn into
a response rather than a hang.
"""

import pytest

from orchestrator.config import settings
from orchestrator.session import (
    InferenceSession,
    SessionFailure,
    SessionManager,
    SessionState,
)


async def _drain(session):
    return [chunk async for chunk in session.stream()]


# ── streaming ───────────────────────────────────────────────────────────

async def test_chunks_arrive_in_order_and_usage_is_recorded():
    session = InferenceSession()
    session.push_chunk("Hello")
    session.push_chunk(" world")
    session.complete(finish_reason="stop", prompt_tokens=4, completion_tokens=2)

    assert await _drain(session) == ["Hello", " world"]
    assert session.state == SessionState.COMPLETE
    assert session.finish_reason == "stop"
    assert session.prompt_tokens == 4
    assert session.completion_tokens == 2


async def test_a_node_error_surfaces_as_a_session_failure():
    session = InferenceSession()
    session.push_chunk("partial")
    session.fail("node exploded")

    with pytest.raises(SessionFailure, match="node exploded"):
        await _drain(session)


async def test_a_silent_node_times_out_instead_of_hanging(monkeypatch):
    """No chunk ever arrives. The client must get an error, not wait forever."""
    monkeypatch.setattr(settings, "chunk_timeout_sec", 0.05)
    monkeypatch.setattr(settings, "generation_timeout_sec", 5)

    session = InferenceSession()
    with pytest.raises(SessionFailure, match="no output"):
        await _drain(session)
    assert session.state == SessionState.FAILED


async def test_an_overrunning_generation_is_cut_off(monkeypatch):
    """Chunks keep coming but the overall budget is spent."""
    monkeypatch.setattr(settings, "generation_timeout_sec", 0)

    session = InferenceSession()
    session.push_chunk("still going")
    with pytest.raises(SessionFailure, match="exceeded"):
        await _drain(session)


# ── terminal states ─────────────────────────────────────────────────────

def test_a_completed_session_cannot_be_failed_afterwards():
    """A late error must not rewrite a result the client already received."""
    session = InferenceSession()
    session.complete(finish_reason="stop", prompt_tokens=1, completion_tokens=1)
    session.fail("too late")

    assert session.state == SessionState.COMPLETE


def test_failing_twice_keeps_the_first_reason():
    session = InferenceSession()
    session.fail("first")
    session.fail("second")

    assert session.state == SessionState.FAILED


# ── manager ─────────────────────────────────────────────────────────────

def test_only_the_dead_node_s_sessions_are_failed():
    manager = SessionManager()
    doomed = manager.create(node_id="dead")
    survivor = manager.create(node_id="alive")

    manager.fail_sessions_for_node("dead")

    assert doomed.state == SessionState.FAILED
    assert survivor.state == SessionState.PENDING


def test_active_count_ignores_finished_sessions():
    manager = SessionManager()
    manager.create(node_id="n1")
    done = manager.create(node_id="n1")
    done.complete(finish_reason="stop", prompt_tokens=0, completion_tokens=0)

    assert manager.active_count == 1


def test_removed_sessions_are_no_longer_addressable():
    manager = SessionManager()
    session = manager.create(node_id="n1")
    manager.remove(session.id)

    assert manager.get(session.id) is None
    assert manager.active_count == 0
