"""The numbers a speedup claim would be published from.

A wrong benchmark is worse than no benchmark: it gets quoted. These pin the
arithmetic (percentiles, spread, per-node throughput) and the parts of the
report that exist to stop the number being over-read - the caveats and the
queue/service split.
"""

import time

from node_agent.cli import bench_record, bench_workload
from orchestrator.batch import BatchStore
from orchestrator.metrics import FleetMetrics, percentile


# ── percentiles ─────────────────────────────────────────────────────────

def test_percentiles_are_values_that_actually_occurred():
    """Nearest-rank, not interpolated: every figure published is a real sample."""
    values = [float(i) for i in range(1, 101)]

    assert percentile(values, 50) == 50.0
    assert percentile(values, 90) == 90.0
    assert percentile(values, 99) == 99.0


def test_percentile_of_nothing_is_none_not_zero():
    """Zero would read as 'instant'; None reads as 'not measured'."""
    assert percentile([], 50) is None


def test_a_single_sample_is_its_own_every_percentile():
    assert percentile([7.5], 50) == percentile([7.5], 99) == 7.5


def test_the_tail_is_not_flattened_by_the_median():
    """The case this exists for: most units fast, one straggler."""
    values = [1.0] * 99 + [60.0]

    assert percentile(values, 50) == 1.0
    assert percentile(values, 99) == 1.0
    assert percentile(values, 100) == 60.0


# ── fleet metrics ───────────────────────────────────────────────────────

def test_queue_and_service_time_are_recorded_separately():
    """Adding machines should cut queue time and leave service time alone.
    Reported as one number, that distinction is invisible."""
    metrics = FleetMetrics()
    metrics.node_joined("n1", "M1")
    metrics.unit_completed("n1", 10, 20, 1.0, queue_sec=4.0, service_sec=1.5)

    tail = metrics.latency_summary()
    assert tail["queue"]["p50"] == 4.0
    assert tail["service"]["p50"] == 1.5


def test_timings_survive_the_node_that_produced_them_leaving():
    """Dropping a departed node's samples would bias the tail towards the
    machines that happened to survive the run."""
    metrics = FleetMetrics()
    metrics.unit_completed("never-registered", 0, 0, queue_sec=9.0, service_sec=2.0)

    assert metrics.latency_summary()["queue"]["count"] == 1


def test_retries_accumulate_across_units():
    metrics = FleetMetrics()
    metrics.node_joined("n1", "M1")
    metrics.unit_completed("n1", 0, 0, retries=2)
    metrics.unit_completed("n1", 0, 0, retries=1)

    assert metrics.unit_retries == 3


def test_negative_timings_never_reach_the_report():
    """Clock skew between lease and completion must not produce a negative."""
    metrics = FleetMetrics()
    metrics.unit_completed("n1", 0, 0, queue_sec=-5.0, service_sec=-1.0, retries=-3)

    assert metrics.latency_summary()["queue"]["p50"] == 0.0
    assert metrics.unit_retries == 0


def test_units_per_hour_exposes_a_slow_machine_in_a_mixed_fleet():
    metrics = FleetMetrics()
    metrics.node_joined("fast", "M4 Pro")
    metrics.nodes["fast"].joined_at = time.time() - 3600
    for _ in range(100):
        metrics.unit_completed("fast", 0, 0)

    assert 95 <= metrics.nodes["fast"].units_per_hour <= 105
    assert "units_per_hour" in metrics.nodes["fast"].snapshot()


def test_the_snapshot_carries_the_tail_and_the_retry_count():
    """These are what a published result quotes; they must be in the payload."""
    snapshot = FleetMetrics().snapshot()

    assert "unit_latency_sec" in snapshot
    assert "unit_retries" in snapshot


# ── timings measured end to end ─────────────────────────────────────────

async def test_a_completed_unit_reports_time_it_waited_and_time_it_took():
    store = BatchStore(metrics=FleetMetrics())
    store.metrics.node_joined("n1", "M1")
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": "q"}]}], "m"
    )
    store.units[batch.unit_ids[0]].created_at = time.time() - 5

    unit = (await store.lease("n1", 1))[0]
    await store.complete(unit.id, "n1", "hi")

    tail = store.metrics.latency_summary()
    assert tail["queue"]["p50"] >= 4.5  # it waited about five seconds
    assert tail["service"]["count"] == 1


async def test_a_requeued_unit_reports_its_retries():
    """A unit that bounced between nodes has to say so, or a fleet that is
    thrashing looks identical to one that is working."""
    store = BatchStore(metrics=FleetMetrics())
    store.metrics.node_joined("n2", "M1")
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": "q"}]}], "m"
    )

    unit = (await store.lease("n1", 1))[0]
    await store.fail(unit.id, "n1", "boom")
    unit = (await store.lease("n2", 1))[0]
    await store.complete(unit.id, "n2", "hi")

    assert store.metrics.unit_retries == 1
    assert store.results(batch.id)[0]["attempts"] == 2


# ── a machine dying mid-run ─────────────────────────────────────────────
#
# The claim is not "units are requeued" - that is mechanism. It is that the
# person who submitted the batch never finds out a machine died. These assert
# the client-visible half.

async def _complete_all(store, node_id, model=None):
    """Lease and finish everything currently available to one node."""
    done = 0
    while True:
        units = await store.lease(node_id, 100, node_model=model)
        if not units:
            return done
        for unit in units:
            await store.complete(unit.id, node_id, f"answer-{unit.index}")
            done += 1


async def test_a_batch_survives_the_machine_running_it_disappearing():
    """Acceptance criterion: no client-visible error despite a node killed
    mid-run, without a graceful shutdown."""
    store = BatchStore(metrics=FleetMetrics())
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(6)], "m"
    )

    doomed = await store.lease("doomed", 3)          # takes half the batch
    assert len(doomed) == 3
    await store.release_node("doomed")               # power cable pulled

    await _complete_all(store, "survivor")

    records = store.results(batch.id)
    assert [r["index"] for r in records] == [0, 1, 2, 3, 4, 5]
    assert all(r["status"] == "complete" for r in records)
    assert not any("error" in r for r in records)


async def test_the_run_record_still_shows_that_churn_happened():
    """Recovering silently is right for the client and wrong for the report -
    a speedup measured through a node death has to disclose it."""
    store = BatchStore(metrics=FleetMetrics())
    store.metrics.node_joined("survivor", "M1")
    await store.create_batch(
        [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(4)], "m"
    )

    await store.lease("doomed", 4)
    await store.release_node("doomed")
    await _complete_all(store, "survivor")

    assert store.metrics.leases_reclaimed == 4
    assert store.metrics.unit_retries == 4  # each unit was attempted twice


async def test_a_lease_that_simply_expires_is_recovered_too():
    """A node that hangs never disconnects, so the reaper is the only path."""
    store = BatchStore(metrics=FleetMetrics())
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": "q"}]}], "m"
    )

    unit = (await store.lease("hung", 1))[0]
    store.units[unit.id].lease_expires_at = time.time() - 1
    assert await store.expire_leases() == 1

    await _complete_all(store, "survivor")
    assert store.results(batch.id)[0]["status"] == "complete"


# ── the workload ────────────────────────────────────────────────────────

def test_the_workload_is_identical_on_every_machine_that_generates_it():
    """A stranger's run is only comparable if they ran the same thing."""
    assert bench_workload(50, 48) == bench_workload(50, 48)


def test_every_request_in_the_workload_is_distinct():
    """Identical prompts would let a cache post a speedup it did not earn."""
    prompts = [r["messages"][0]["content"] for r in bench_workload(200, 48)]

    assert len(set(prompts)) == 200


def test_the_workload_is_shaped_the_way_the_batch_api_accepts():
    """It is built here rather than parsed from a file, so nothing else
    converts `prompt` into `messages` on the way - this caught a 422."""
    unit = bench_workload(1, 48)[0]

    assert unit["messages"] == [
        {"role": "user", "content": "Name three uses for a paperclip. (#0)"}
    ]


def test_the_workload_is_greedy_so_two_runs_are_comparable():
    assert all(r["temperature"] == 0.0 for r in bench_workload(10, 48))


# ── the record ──────────────────────────────────────────────────────────

def _record(runs, nodes=2, before=None, after=None):
    base = {"unit_retries": 0, "leases_reclaimed": 0, "units_failed": 0}
    return bench_record(
        workload={"requests": 100, "max_tokens": 48, "temperature": 0.0,
                  "model": "m", "replicates": len(runs)},
        runs=runs,
        before={**base, **(before or {})},
        after={**base, "nodes": [{"gpu": f"M{i}"} for i in range(nodes)],
               **(after or {})},
    )


def test_the_median_is_reported_not_the_best_run():
    """Quoting the fastest replicate is how honest benchmarks become dishonest."""
    assert _record([10.0, 12.0, 20.0])["median_sec"] == 12.0


def test_the_spread_is_published_alongside_the_median():
    record = _record([10.0, 12.0, 14.0])

    assert record["fastest_sec"] == 10.0
    assert record["slowest_sec"] == 14.0
    assert record["spread_pct"] > 0


def test_a_single_node_run_refuses_to_call_itself_a_speedup():
    caveats = " ".join(_record([10.0, 11.0, 12.0], nodes=1)["does_not_establish"])

    assert "baseline, not a speedup" in caveats


def test_a_wide_spread_is_called_out_as_noise():
    caveats = " ".join(_record([10.0, 15.0, 30.0])["does_not_establish"])

    assert "noise" in caveats


def test_too_few_replicates_is_called_out():
    caveats = " ".join(_record([10.0])["does_not_establish"])

    assert "3 replicates" in caveats


def test_a_clean_multi_node_run_still_states_its_limits():
    """There is no configuration that produces a caveat-free number."""
    assert _record([10.0, 10.1, 10.2], nodes=3)["does_not_establish"]


def test_counters_are_reported_as_deltas_not_fleet_lifetime_totals():
    """A second bench against a warm fleet must not inherit the first's retries."""
    record = _record(
        [10.0, 10.0, 10.0],
        before={"unit_retries": 40, "leases_reclaimed": 7},
        after={"unit_retries": 43, "leases_reclaimed": 9},
    )

    assert record["retries"] == 3
    assert record["leases_reclaimed"] == 2


def test_throughput_is_derived_from_the_median_run():
    assert _record([10.0, 10.0, 10.0])["units_per_sec"] == 10.0
