"""Work-unit bookkeeping the HTTP tests do not reach.

Ordering, usage accounting, and which units a given event is allowed to touch.
These are the properties a batch client experiences as "my results came back
correct and complete", so they are worth pinning at the store level where they
are cheap to check.
"""

from orchestrator.batch import BatchState, BatchStore, UnitState


async def _batch(store, n=3, model="tiny-test-model"):
    return await store.create_batch(
        [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(n)], model
    )


# ── results ─────────────────────────────────────────────────────────────

async def test_results_are_ordered_by_submission_not_completion():
    """The fleet finishes units in any order; the client must not see that."""
    store = BatchStore()
    batch = await _batch(store, n=3)
    units = await store.lease("n1", 3)

    for unit in reversed(units):
        await store.complete(unit.id, "n1", f"answer {unit.index}")

    records = store.results(batch.id)
    assert [r["index"] for r in records] == [0, 1, 2]
    assert records[0]["response"]["content"] == "answer 0"


async def test_a_dead_lettered_unit_reports_its_error_not_a_response():
    store = BatchStore()
    batch = await _batch(store, n=1)
    for _ in range(3):
        unit = (await store.lease("n1", 1))[0]
        await store.fail(unit.id, "n1", "engine exploded")

    record = store.results(batch.id)[0]
    assert record["status"] == UnitState.FAILED.value
    assert "engine exploded" in record["error"]
    assert "response" not in record


async def test_usage_counts_only_completed_units():
    store = BatchStore()
    batch = await _batch(store, n=2)
    units = await store.lease("n1", 2)
    await store.complete(units[0].id, "n1", "done", prompt_tokens=5, completion_tokens=3)

    assert store.usage(batch.id) == {
        "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8,
    }


async def test_counts_break_down_by_state():
    store = BatchStore()
    batch = await _batch(store, n=3)
    leased = await store.lease("n1", 2)
    await store.complete(leased[0].id, "n1", "done")

    assert store.counts(batch.id) == {
        "total": 3, "completed": 1, "failed": 0, "in_flight": 1, "pending": 1,
    }


# ── who may touch which unit ────────────────────────────────────────────

async def test_releasing_one_node_leaves_another_node_s_leases_alone():
    store = BatchStore()
    await _batch(store, n=2)
    kept = (await store.lease("keeper", 1))[0]
    lost = (await store.lease("goner", 1))[0]

    assert await store.release_node("goner") == 1
    assert store.units[kept.id].state == UnitState.LEASED
    assert store.units[lost.id].state == UnitState.PENDING


async def test_a_cancelled_batch_stops_handing_out_work():
    store = BatchStore()
    batch = await _batch(store, n=3)
    await store.cancel_batch(batch.id)

    assert await store.lease("n1", 3) == []
    assert store.get_batch(batch.id).state == BatchState.CANCELLED


async def test_cancelling_twice_is_rejected():
    store = BatchStore()
    batch = await _batch(store, n=1)

    assert await store.cancel_batch(batch.id) is True
    assert await store.cancel_batch(batch.id) is False


# ── batch completion ────────────────────────────────────────────────────

async def test_a_batch_completes_only_once_every_unit_settles():
    store = BatchStore()
    batch = await _batch(store, n=2)
    units = await store.lease("n1", 2)

    await store.complete(units[0].id, "n1", "ok")
    assert store.get_batch(batch.id).state == BatchState.IN_PROGRESS

    await store.complete(units[1].id, "n1", "ok")
    assert store.get_batch(batch.id).state == BatchState.COMPLETED


async def test_pending_count_tracks_requeued_work():
    store = BatchStore()
    await _batch(store, n=2)
    await store.lease("n1", 2)
    assert store.pending_count == 0

    await store.release_node("n1")
    assert store.pending_count == 2
