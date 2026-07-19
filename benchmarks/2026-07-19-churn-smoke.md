# Churn smoke test — 2026-07-19

Not a benchmark (single unpaired run, no replicates). A smoke test of the
work-unit lease semantics on real hardware with a real model.

## Setup

- Fleet: 2 node agents on one MacBook (M3 Pro, 18 GB), MLX engine
- Model: mlx-community/Llama-3.2-1B-Instruct-4bit
- Orchestrator: localhost, in-memory BatchStore
- Batch: 24 requests x 300 max_tokens ("write a detailed paragraph...")
- Node lease size: 4 units (NODE_BATCH_SIZE default)

## Procedure

1. Submit the 24-unit batch to the 2-node fleet.
2. After 8 s (7 units complete, 8 leased, 9 pending), `kill -9` one node —
   a hard SIGKILL, so no clean shutdown or lease hand-back.
3. Let the surviving node drain the queue.

## Observations

| Moment | completed | in_flight | pending |
|---|---|---|---|
| Before kill | 7 | 8 | 9 |
| ~5 s after kill | 12 | 4 | 8 |
| Final | 24 | 0 | 0 |

Final batch state: `completed`, 24/24, **0 failed**.

Result integrity:
- 24 records, indices contiguous 0–23, submission order preserved
- All records non-empty
- **Exactly 4 units had attempts > 1** — the 4 leases the killed node held

Wall clock: 26 s (created → completed), 6,765 completion tokens
(≈260 tok/s aggregate, degrading to one node partway). Treat the token rate
as an anecdote, not a measurement: single run, shared machine, no controls.

## What this shows

The lease/requeue path works end-to-end under an uncooperative failure
(SIGKILL, no goodbye). Losing a node mid-batch cost exactly the work that was
in flight on it, and that work was retried automatically on the survivor.
No operator action, no client-visible error, no duplicate or missing records.

## What it does not show

- Multi-machine or multi-city behavior (all nodes were local)
- Behavior under network partition (as opposed to process death)
- Throughput under contention, or any cost-per-token figure
- Verification of output correctness on untrusted nodes (Phase 2d)
