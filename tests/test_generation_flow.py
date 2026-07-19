"""End-to-end tests for the whole-model generation flow.

A fake node is driven over the real WebSocket protocol while API calls run in
a background thread, exercising register -> serve_model -> model_loaded ->
generate_request -> chunks -> completion.
"""

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from orchestrator.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register_node(ws, client, node_id="fakenode1", model="tiny-test-model"):
    ws.send_text(json.dumps({
        "type": "register",
        "node_id": node_id,
        "gpu_name": "test-gpu",
        "gpu_vram_mb": 8000,
        "runtime": "native",
        "model_id": model,
    }))
    assignment = ws.receive_json()
    assert assignment["type"] == "serve_model"
    assert assignment["model_id"] == model

    ws.send_text(json.dumps({
        "type": "model_loaded",
        "node_id": node_id,
        "model_id": model,
    }))
    # Wait for the orchestrator to process model_loaded
    for _ in range(50):
        if client.get("/health").json()["nodes"]["ready_nodes"] == 1:
            return
        time.sleep(0.05)
    raise AssertionError("node never became ready")


def test_node_registration_and_models_list(client):
    with client.websocket_connect("/nodes/ws") as ws:
        _register_node(ws, client)

        models = client.get("/v1/models").json()
        assert models["data"][0]["id"] == "tiny-test-model"

        health = client.get("/health").json()
        node = health["nodes"]["nodes"][0]
        assert node["model"] == "tiny-test-model"


def test_full_generation_roundtrip(client):
    with client.websocket_connect("/nodes/ws") as ws:
        _register_node(ws, client)

        result = {}

        def post_completion():
            result["resp"] = client.post("/v1/chat/completions", json={
                "model": "tiny-test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            })

        t = threading.Thread(target=post_completion)
        t.start()

        # Act as the node: receive the generate request, stream a reply
        gen = ws.receive_json()
        assert gen["type"] == "generate_request"
        assert gen["messages"] == [{"role": "user", "content": "Hello"}]
        sid = gen["session_id"]

        ws.send_text(json.dumps({"type": "generate_chunk", "session_id": sid, "text": "Hello "}))
        ws.send_text(json.dumps({"type": "generate_chunk", "session_id": sid, "text": "world"}))
        ws.send_text(json.dumps({
            "type": "generate_complete",
            "session_id": sid,
            "finish_reason": "stop",
            "prompt_tokens": 3,
            "completion_tokens": 2,
        }))

        t.join(timeout=10)
        resp = result["resp"]
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello world"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 3
        assert data["usage"]["completion_tokens"] == 2
        assert data["usage"]["total_tokens"] == 5
        assert data["model"] == "tiny-test-model"

        # Session cleaned up
        assert client.get("/health").json()["active_sessions"] == 0


def test_generation_error_from_node(client):
    with client.websocket_connect("/nodes/ws") as ws:
        _register_node(ws, client)

        result = {}

        def post_completion():
            result["resp"] = client.post("/v1/chat/completions", json={
                "model": "tiny-test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            })

        t = threading.Thread(target=post_completion)
        t.start()

        gen = ws.receive_json()
        sid = gen["session_id"]
        ws.send_text(json.dumps({
            "type": "generate_error",
            "session_id": sid,
            "message": "out of memory",
        }))

        t.join(timeout=10)
        resp = result["resp"]
        assert resp.status_code == 502
        assert "out of memory" in resp.json()["error"]["message"]


def test_node_disconnect_fails_inflight_session(client):
    with client.websocket_connect("/nodes/ws") as ws:
        _register_node(ws, client)

        result = {}

        def post_completion():
            result["resp"] = client.post("/v1/chat/completions", json={
                "model": "tiny-test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            })

        t = threading.Thread(target=post_completion)
        t.start()

        gen = ws.receive_json()
        assert gen["type"] == "generate_request"
        # Node dies mid-generation: close the socket without replying
        ws.close()

        t.join(timeout=15)
        resp = result["resp"]
        assert resp.status_code == 502
        assert "disconnected" in resp.json()["error"]["message"]


def test_mock_engine_generates():
    from node_agent.engine import MockEngine, create_engine

    engine = MockEngine()
    engine.load("test-model")
    out = "".join(engine.generate_stream(
        [{"role": "user", "content": "ping pong"}], max_tokens=64, temperature=0.5,
    ))
    assert "ping pong" in out
    assert engine.last_completion_tokens > 0
    assert engine.last_prompt_tokens == 2

    # max_tokens caps output length
    short = "".join(MockEngine().generate_stream(
        [{"role": "user", "content": "a b c d e f g h"}], max_tokens=1, temperature=0.5,
    ))
    assert len(short.split()) == 1

    # factory: mock is always resolvable
    assert create_engine("mock").name == "mock"
