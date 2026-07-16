"""Basic tests for the orchestrator."""

import pytest
from fastapi.testclient import TestClient

from orchestrator.main import app


@pytest.fixture
def client():
    # Use context manager so lifespan (startup/shutdown) runs
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "nodes" in data


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    # With no nodes connected, the configured default model is advertised
    assert len(data["data"]) == 1
    assert data["data"][0]["object"] == "model"


def test_chat_completions_no_nodes(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "any-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    # Should return 503 since no nodes are available
    assert resp.status_code == 503
    data = resp.json()
    assert data["error"]["code"] == "no_capacity"


def test_dashboard(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text


def test_compute_page(client):
    resp = client.get("/compute")
    assert resp.status_code == 200
    assert "WebGPU" in resp.text
