"""Wire-protocol edges on the node socket.

These are the frames a real node eventually sends by accident: a bad first
message, a binary frame, a type nobody implemented, a capacity of zero, a
result for a unit that no longer exists. None of them may take the orchestrator
down or strand a lease.
"""

import json
import time

import pytest
from fastapi.testclient import TestClient

from orchestrator.api.nodes import MAX_LEASE_PER_REQUEST
from orchestrator.main import app

NODE = "protonode"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register(ws, client, node_id=NODE, model="tiny-test-model"):
    ws.send_text(json.dumps({
        "type": "register", "node_id": node_id, "model_id": model,
    }))
    assert ws.receive_json()["type"] == "serve_model"
    ws.send_text(json.dumps({
        "type": "model_loaded", "node_id": node_id, "model_id": model,
    }))
    for _ in range(50):
        if client.get("/health").json()["nodes"]["ready_nodes"] >= 1:
            return
        time.sleep(0.05)
    raise AssertionError("node never became ready")


def _submit(client, n=1):
    return client.post("/v1/batches", json={
        "requests": [
            {"messages": [{"role": "user", "content": f"q{i}"}], "max_tokens": 8}
            for i in range(n)
        ],
    })


def _ask_for_work(ws, capacity):
    ws.send_text(json.dumps({
        "type": "work_request", "node_id": NODE, "capacity": capacity,
    }))
    return ws.receive_json()


# ── handshake ───────────────────────────────────────────────────────────

def test_the_first_message_must_be_register(client):
    with client.websocket_connect("/nodes/ws") as ws:
        ws.send_text(json.dumps({"type": "heartbeat", "node_id": "impostor"}))
        assert "register" in ws.receive_json()["error"]


def test_a_node_that_never_loads_a_model_is_not_servable(client):
    with client.websocket_connect("/nodes/ws") as ws:
        ws.send_text(json.dumps({
            "type": "register", "node_id": "warming", "model_id": "tiny-test-model",
        }))
        assert ws.receive_json()["type"] == "serve_model"

        assert client.get("/health").json()["nodes"]["ready_nodes"] == 0
        assert client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
        }).status_code == 503


# ── frames that must not kill the connection ────────────────────────────

def test_an_unknown_message_type_is_survivable(client):
    with client.websocket_connect("/nodes/ws") as ws:
        _register(ws, client)
        ws.send_text(json.dumps({"type": "no_such_message", "node_id": NODE}))

        assert _ask_for_work(ws, 1)["type"] == "work_assignment"


def test_a_binary_frame_is_ignored(client):
    """Nodes only speak JSON text; stray bytes must not desynchronise the socket."""
    with client.websocket_connect("/nodes/ws") as ws:
        _register(ws, client)
        ws.send_bytes(b"\x00\x01\x02")

        assert _ask_for_work(ws, 1)["type"] == "work_assignment"


def test_a_result_for_an_unknown_unit_is_dropped(client):
    with client.websocket_connect("/nodes/ws") as ws:
        _register(ws, client)
        ws.send_text(json.dumps({
            "type": "work_result", "node_id": NODE,
            "unit_id": "unit_does_not_exist", "text": "orphan",
        }))

        assert _ask_for_work(ws, 1)["type"] == "work_assignment"


# ── leasing bounds ──────────────────────────────────────────────────────

def test_asking_for_no_work_leases_nothing(client):
    with client.websocket_connect("/nodes/ws") as ws:
        _register(ws, client)
        _submit(client, n=1)
        assert ws.receive_json()["type"] == "work_available"

        assert _ask_for_work(ws, 0)["units"] == []


def test_capacity_is_clamped_to_the_per_request_cap(client):
    """A node asking for everything still gets a bounded slice."""
    with client.websocket_connect("/nodes/ws") as ws:
        _register(ws, client)
        _submit(client, n=MAX_LEASE_PER_REQUEST + 5)
        assert ws.receive_json()["type"] == "work_available"

        assert len(_ask_for_work(ws, 10_000)["units"]) == MAX_LEASE_PER_REQUEST


def test_a_negative_capacity_leases_nothing(client):
    with client.websocket_connect("/nodes/ws") as ws:
        _register(ws, client)
        _submit(client, n=2)
        assert ws.receive_json()["type"] == "work_available"

        assert _ask_for_work(ws, -5)["units"] == []
