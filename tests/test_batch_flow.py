"""Batch API and work-unit queue tests.

Covers the properties that make node churn boring: leases are reclaimed when a
node dies, results are idempotent, and failed units retry until they're out of
attempts.
"""

import json
import time

import pytest
from fastapi.testclient import TestClient

from orchestrator.batch import BatchStore, UnitState
from orchestrator.config import settings
from orchestrator.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register_node(ws, client, node_id="batchnode1", model="tiny-test-model"):
    ws.send_text(json.dumps({
        "type": "register", "node_id": node_id, "gpu_name": "test-gpu",
        "gpu_vram_mb": 8000, "runtime": "native",
        "model_id": model,
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


def _submit(client, n=3, model="tiny-test-model"):
    resp = client.post("/v1/batches", json={
        "model": model,
        "requests": [
            {"messages": [{"role": "user", "content": f"question {i}"}], "max_tokens": 16}
            for i in range(n)
        ],
    })
    assert resp.status_code == 201, resp.text
    return resp.json()


# ── API surface ─────────────────────────────────────────────────────────

def test_create_and_get_batch(client):
    batch = _submit(client, n=3)
    assert batch["status"] == "in_progress"
    assert batch["request_counts"]["total"] == 3
    assert batch["request_counts"]["pending"] == 3

    fetched = client.get(f"/v1/batches/{batch['id']}").json()
    assert fetched["id"] == batch["id"]

    listed = client.get("/v1/batches").json()
    assert any(b["id"] == batch["id"] for b in listed["data"])


def test_empty_batch_rejected(client):
    resp = client.post("/v1/batches", json={"requests": []})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


def test_unknown_batch_404(client):
    assert client.get("/v1/batches/batch_nope").status_code == 404
    assert client.get("/v1/batches/batch_nope/results").status_code == 404


def test_cancel_batch(client):
    batch = _submit(client, n=2)
    cancelled = client.post(f"/v1/batches/{batch['id']}/cancel").json()
    assert cancelled["status"] == "cancelled"
    # A cancelled batch cannot be cancelled again
    assert client.post(f"/v1/batches/{batch['id']}/cancel").status_code == 409


# ── End-to-end through a node ───────────────────────────────────────────

def test_batch_completes_through_node(client):
    with client.websocket_connect("/nodes/ws") as ws:
        _register_node(ws, client)
        batch = _submit(client, n=2)

        # Batch creation nudges the node, which then requests work
        assert ws.receive_json()["type"] == "work_available"
        ws.send_text(json.dumps({
            "type": "work_request", "node_id": "batchnode1", "capacity": 4,
        }))
        assignment = ws.receive_json()
        assert assignment["type"] == "work_assignment"
        units = assignment["units"]
        assert len(units) == 2
        assert units[0]["messages"][0]["content"].startswith("question")

        for i, unit in enumerate(units):
            ws.send_text(json.dumps({
                "type": "work_result", "node_id": "batchnode1",
                "unit_id": unit["unit_id"], "text": f"answer {i}",
                "prompt_tokens": 5, "completion_tokens": 3,
            }))

        for _ in range(50):
            status = client.get(f"/v1/batches/{batch['id']}").json()
            if status["status"] == "completed":
                break
            time.sleep(0.05)
        assert status["status"] == "completed"
        assert status["request_counts"]["completed"] == 2
        assert status["usage"]["total_tokens"] == 16  # (5+3) * 2

        body = client.get(f"/v1/batches/{batch['id']}/results").text
        records = [json.loads(line) for line in body.strip().split("\n")]
        assert len(records) == 2
        assert [r["index"] for r in records] == [0, 1]
        assert records[0]["response"]["content"] == "answer 0"
        assert all(r["status"] == "complete" for r in records)


def test_node_disconnect_requeues_leased_units(client):
    with client.websocket_connect("/nodes/ws") as ws:
        _register_node(ws, client, node_id="deadnode")
        batch = _submit(client, n=2)
        ws.receive_json()  # work_available
        ws.send_text(json.dumps({
            "type": "work_request", "node_id": "deadnode", "capacity": 4,
        }))
        assert len(ws.receive_json()["units"]) == 2
        # Node dies holding both leases
        ws.close()

    for _ in range(50):
        counts = client.get(f"/v1/batches/{batch['id']}").json()["request_counts"]
        if counts["pending"] == 2:
            break
        time.sleep(0.05)
    assert counts["pending"] == 2, "leases should return to the queue"
    assert counts["in_flight"] == 0


# ── Store semantics ─────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_duplicate_results_ignored():
    store = BatchStore()
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": "hi"}]}], "m"
    )
    unit = (await store.lease("node-a", 1))[0]

    await store.complete(unit.id, "node-a", "first", 1, 1)
    # A retry from a slower duplicate must not overwrite the recorded result
    await store.complete(unit.id, "node-b", "second", 9, 9)

    assert store.units[unit.id].result_text == "first"
    assert store.usage(batch.id)["total_tokens"] == 2
    assert store.get_batch(batch.id).state.value == "completed"


@pytest.mark.anyio
async def test_unit_retries_then_dead_letters():
    store = BatchStore()
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": "hi"}]}], "m"
    )

    for attempt in range(settings.max_unit_attempts):
        leased = await store.lease("node-a", 1)
        assert len(leased) == 1, f"unit should be leasable on attempt {attempt + 1}"
        await store.fail(leased[0].id, "node-a", "boom")

    # Out of attempts: dead-lettered, not leasable again
    assert await store.lease("node-a", 1) == []
    counts = store.counts(batch.id)
    assert counts["failed"] == 1
    assert store.get_batch(batch.id).state.value == "completed"
    assert store.results(batch.id)[0]["error"] == "boom"


@pytest.mark.anyio
async def test_expired_lease_is_reclaimed(monkeypatch):
    store = BatchStore()
    await store.create_batch([{"messages": [{"role": "user", "content": "hi"}]}], "m")
    monkeypatch.setattr(settings, "lease_duration_sec", -1)  # already expired

    leased = await store.lease("node-a", 1)
    assert store.units[leased[0].id].state == UnitState.LEASED

    assert await store.expire_leases() == 1
    assert store.units[leased[0].id].state == UnitState.PENDING
    assert len(await store.lease("node-b", 1)) == 1


@pytest.mark.anyio
async def test_lease_respects_capacity_and_fifo_order():
    store = BatchStore()
    await store.create_batch(
        [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(5)], "m"
    )
    first = await store.lease("node-a", 2)
    second = await store.lease("node-b", 10)

    assert [u.index for u in first] == [0, 1]
    assert [u.index for u in second] == [2, 3, 4]
    assert await store.lease("node-c", 1) == []
