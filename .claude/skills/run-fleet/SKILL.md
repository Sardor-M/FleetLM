---
name: run-fleet
description: Start a local FleetLM fleet and drive real work through it - orchestrator, one or more nodes, an interactive request, and a batch. Use when asked to run, demo, or verify FleetLM end-to-end, or after changing the orchestrator, node agent, or wire protocol.
---

# Run a local fleet

Verify FleetLM by driving it, not by reading it. Tests cover the protocol;
this covers "does the whole thing actually work."

## 1. Start the orchestrator

```bash
.venv/bin/python -m uvicorn orchestrator.main:app --host 127.0.0.1 --port 8080 \
  > /tmp/fleet-orch.log 2>&1 &
sleep 2 && curl -s http://127.0.0.1:8080/health
```

## 2. Attach a node

Real inference (Apple silicon, ~1.3 s to load a cached model):

```bash
.venv/bin/fleetlm join http://localhost:8080 > /tmp/fleet-node.log 2>&1 &
```

No GPU, no download - exercises the identical protocol path:

```bash
.venv/bin/fleetlm join http://localhost:8080 --engine mock --model demo &
```

Wait for readiness rather than sleeping a fixed time:

```bash
until curl -s http://127.0.0.1:8080/health | grep -q '"ready_nodes":1'; do sleep 2; done
```

First run with a real model downloads weights; that can take minutes, and
Hugging Face outages surface here as a load failure in the node log.

## 3. Drive work through it

```bash
# interactive
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'

# streaming (expect chat.completion.chunk SSE events)
curl -sN -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Count to 5"}],"stream":true}' | head -5

# batch - the fleet's real workload
ID=$(curl -s -X POST http://127.0.0.1:8080/v1/batches \
  -H "Content-Type: application/json" \
  -d '{"requests":[{"messages":[{"role":"user","content":"Name a color."}],"max_tokens":20}]}' \
  | .venv/bin/python -c "import sys,json; print(json.load(sys.stdin)['id'])")

until curl -s http://127.0.0.1:8080/v1/batches/$ID \
  | .venv/bin/python -c "import sys,json; sys.exit(0 if json.load(sys.stdin)['status']=='completed' else 1)"
do sleep 3; done

curl -s http://127.0.0.1:8080/v1/batches/$ID/results   # JSONL, submission order
curl -s http://127.0.0.1:8080/metrics                  # what the fleet did
```

## 4. Verify churn, if you touched leases or the node lifecycle

The property worth protecting: killing a node loses no work.

```bash
# submit enough long units that work is still in flight, then:
kill -9 $(pgrep -f node_agent | head -1)
```

Then confirm the batch still reaches `completed` with `failed: 0`, and that
`attempts > 1` appears on exactly the units the dead node held.

## 5. Clean up - always

```bash
pkill -9 -f node_agent; pkill -f "uvicorn orchestrator"
```

## Gotchas learned the hard way

- `pkill -f "python -m node_agent"` silently matches nothing: the process
  shows as `.../Python -m node_agent`, capitalized. Use `pkill -f node_agent`
  and verify with `pgrep -fl node_agent`.
- A fast node can finish a small batch before you can observe it mid-flight.
  For churn tests use many units with large `max_tokens`.
- A raw test WebSocket does not request work on its own; only the real node
  agent does. In tests, send `work_request` explicitly.
- If nothing is ready after a minute, read the node log - a model load
  failure is reported there, not by the orchestrator.
