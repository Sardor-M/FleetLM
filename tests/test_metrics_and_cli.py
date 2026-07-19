"""Fleet metrics, join-token auth, and the contributor CLI."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from node_agent.cli import to_ws_url
from orchestrator.config import settings
from orchestrator.main import app
from orchestrator.metrics import FleetMetrics


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── URL handling (what contributors paste) ──────────────────────────────

@pytest.mark.parametrize("given,expected", [
    ("https://fleet.example.com", "wss://fleet.example.com/nodes/ws"),
    ("http://localhost:8080", "ws://localhost:8080/nodes/ws"),
    ("localhost:8080", "ws://localhost:8080/nodes/ws"),
    ("http://localhost:8080/", "ws://localhost:8080/nodes/ws"),
    ("ws://localhost:8080/nodes/ws", "ws://localhost:8080/nodes/ws"),
    ("wss://fleet.example.com/nodes/ws", "wss://fleet.example.com/nodes/ws"),
])
def test_to_ws_url(given, expected):
    assert to_ws_url(given) == expected


# ── Metrics accounting ──────────────────────────────────────────────────

def test_metrics_track_throughput_and_join_time():
    m = FleetMetrics()
    m.node_joined("node-a", "apple-silicon")
    m.node_ready("node-a")
    m.unit_completed("node-a", prompt_tokens=10, completion_tokens=90, seconds=3.0)
    m.unit_completed("node-a", prompt_tokens=10, completion_tokens=30, seconds=1.0)
    m.unit_failed("node-a")

    snap = m.snapshot()
    assert snap["nodes_live"] == 1
    assert snap["units_completed"] == 2
    assert snap["units_failed"] == 1
    assert snap["unit_success_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert snap["completion_tokens"] == 120
    # 120 completion tokens over 4.0 s of generation
    assert snap["fleet_tokens_per_sec_generating"] == pytest.approx(30.0)
    assert snap["median_join_to_ready_sec"] is not None
    assert snap["nodes"][0]["tokens_per_sec"] == pytest.approx(30.0)


def test_metrics_retain_departed_node_contribution():
    m = FleetMetrics()
    m.node_joined("gone", "apple-silicon")
    m.node_ready("gone")
    m.unit_completed("gone", 5, 45, 1.5)
    m.node_left("gone")

    snap = m.snapshot()
    assert snap["nodes_live"] == 0
    assert snap["nodes_departed"] == 1
    # Work done by a node that has since left still counts toward the fleet
    assert snap["completion_tokens"] == 45
    assert snap["units_completed"] == 1


def test_metrics_ignore_events_for_unknown_nodes():
    m = FleetMetrics()
    m.unit_completed("ghost", 10, 10, 1.0)
    m.session_failed("ghost")
    assert m.snapshot()["units_completed"] == 0


def test_metrics_endpoint_and_lease_counter(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("fleet_uptime_sec", "nodes_live", "units_completed", "nodes"):
        assert key in body

    with client.websocket_connect("/nodes/ws") as ws:
        ws.send_text(json.dumps({
            "type": "register", "node_id": "metricnode", "gpu_name": "test-gpu",
            "gpu_vram_mb": 8000, "runtime": "native",
            "model_id": "tiny-test-model",
        }))
        assert ws.receive_json()["type"] == "serve_model"
        ws.send_text(json.dumps({
            "type": "model_loaded", "node_id": "metricnode", "model_id": "tiny-test-model",
        }))
        for _ in range(50):
            if client.get("/metrics").json()["nodes_live"] == 1:
                break
            time.sleep(0.05)

        snap = client.get("/metrics").json()
        assert snap["nodes_live"] == 1
        assert snap["nodes"][0]["join_to_ready_sec"] is not None

        # Lease two units, then die holding them
        client.post("/v1/batches", json={
            "model": "tiny-test-model",
            "requests": [{"messages": [{"role": "user", "content": f"q{i}"}]}
                         for i in range(2)],
        })
        ws.receive_json()  # work_available
        ws.send_text(json.dumps({
            "type": "work_request", "node_id": "metricnode", "capacity": 4,
        }))
        assert len(ws.receive_json()["units"]) == 2
        ws.close()

    for _ in range(50):
        snap = client.get("/metrics").json()
        if snap["leases_reclaimed"] >= 2:
            break
        time.sleep(0.05)
    assert snap["leases_reclaimed"] >= 2
    assert snap["nodes_departed"] >= 1


# ── Join token ──────────────────────────────────────────────────────────

def test_join_token_rejects_wrong_token(client, monkeypatch):
    monkeypatch.setattr(settings, "join_token", "s3cret")
    with client.websocket_connect("/nodes/ws") as ws:
        ws.send_text(json.dumps({
            "type": "register", "node_id": "intruder",
            "model_id": "m", "join_token": "wrong",
        }))
        assert "invalid" in ws.receive_json()["error"]
    assert client.get("/health").json()["nodes"]["total_nodes"] == 0


def test_join_token_accepts_correct_token(client, monkeypatch):
    monkeypatch.setattr(settings, "join_token", "s3cret")
    with client.websocket_connect("/nodes/ws") as ws:
        ws.send_text(json.dumps({
            "type": "register", "node_id": "invited",
            "model_id": "m", "join_token": "s3cret",
        }))
        assert ws.receive_json()["type"] == "serve_model"
