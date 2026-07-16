# FleetLM — LLM Inference on a Fleet of Everyday Laptops

FleetLM serves real LLM completions from consumer machines — laptops people already own — behind one OpenAI-compatible API. A lightweight orchestrator routes each request over a single outbound WebSocket to a node that holds the **entire model** in unified memory and streams tokens back. No datacenter GPU in the serving path, no inbound connections to any node, and a node dying mid-request costs one retry, not a session.

The design follows one rule learned the hard way: **match the workload to what consumer hardware and home internet are actually good at** — memory-bound inference and throughput — and keep wide-area latency off the per-token critical path.

In this setup:

- One command starts the orchestrator; one command turns any Apple-silicon Mac into a serving node (`python -m node_agent`).
- Nodes hold the whole model (int8/4-bit via MLX or llama.cpp) — no layer sharding, no cross-node activation traffic, no pipeline to break.
- Every node connection is **outbound** — NAT and firewalls are a non-problem by construction.
- The API is OpenAI-compatible: `/v1/chat/completions` with both JSON and SSE streaming responses, plus `/v1/models` reflecting what the fleet actually serves.
- Requests route to the least-loaded ready node serving the requested model.
- Nodes join, leave, or crash without stalling the fleet; in-flight sessions on a dead node fail fast with a clean error.
- A dependency-free `mock` engine runs the full wire protocol end-to-end, so the system is testable without downloading a model.
- 10 integration tests drive the real protocol: register → serve → generate → stream → complete, plus error and node-loss paths.
- A live dashboard (`/`) shows the fleet; a browser page (`/compute`) is the future zero-install contributor on-ramp (WebGPU).

The rest of this README explains each decision.

---

## 1 · What's the right topology for a fleet you can't wire?

Consumer machines sit behind NAT, join and leave without warning, and connect over links that stall. So no node is ever required to accept a connection: each node opens **one outbound WebSocket** to the orchestrator, registers with its hardware profile and the model it serves, and everything — control, requests, token streams, heartbeats — flows over that socket. Traffic is a star: requests fan out one-to-one, tokens flow back, and no node ever talks to another node.

The orchestrator keeps only soft state (a node registry, in-flight sessions). Heartbeats every 5 s evict stale nodes; a node that reconnects simply re-registers. Losing the orchestrator loses no durable data — nodes reconnect and re-register.

## 2 · Where should the model live?

Our first architecture sharded transformer layers across nodes (pipeline parallelism) — browser tab A runs layers 0–15, tab B runs 16–31. We reversed that decision after studying Pluralis Research's Stoa run [1], which put real numbers behind the alternative: **replicate the whole model on every node, shard nothing.**

Sharding across home internet puts a 50–200 ms WAN hop inside every token's forward pass, makes every node a single point of failure for every in-flight request, and requires the fleet to maintain complete layer coverage at all times. Replication has none of these: a 1–8B model at 4–8 bits fits comfortably in the unified memory of an ordinary MacBook, nodes are fungible, and failure semantics collapse to "retry on another replica." Layer sharding remains on the roadmap for one honest use case only: models too large for any single consumer device (the protocol still carries a `layer_shard` mode for it).

## 3 · How does a request become tokens?

```
Client ── POST /v1/chat/completions ──▶ Orchestrator ── generate_request ──▶ Node (whole model)
Client ◀── JSON or SSE stream ───────── Orchestrator ◀── generate_chunk* ──── (MLX / llama.cpp)
                                                     ◀── generate_complete ──
```

The wire protocol is small and typed (`orchestrator/protocol/messages.py`):

1. Node → `register` (hardware, `mode: whole_model`, `model_id`)
2. Orchestrator → `serve_model`; node loads weights, replies `model_loaded`
3. Client request arrives; the router picks the least-loaded ready node serving that model
4. Orchestrator → `generate_request` (messages, sampling params, session id)
5. Node → `generate_chunk` per text piece, then `generate_complete` with usage counts
6. The orchestrator relays chunks to the client — aggregated JSON, or SSE `chat.completion.chunk` events when `"stream": true`

Timeouts bound every stage (silence between chunks, total generation time), and a node disconnect immediately fails its in-flight sessions with a 502 rather than hanging the client.

## 4 · What runs on a node?

The node agent is ~200 lines of asyncio around a pluggable engine (`node_agent/engine/whole_model.py`):

| Engine | When | Models |
|---|---|---|
| `mlx` | Apple silicon (default when installed) | Hugging Face repos, e.g. `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| `llama_cpp` | Any platform with llama-cpp-python | local GGUF files |
| `mock` | Tests, demos, CI — zero dependencies | none (canned echo) |

Generation runs in a worker thread so heartbeats and control messages never block behind a long completion; all outbound traffic funnels through one queue so nothing writes to the socket concurrently. Engine selection: `NODE_ENGINE=auto|mlx|llama_cpp|mock`, model: `NODE_MODEL=...`.

## 5 · Quick start

```bash
# Prerequisites: Python 3.11+
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Real inference on Apple silicon:
pip install mlx-lm

# 1. Start the orchestrator
uvicorn orchestrator.main:app --host 0.0.0.0 --port 8080

# 2. Start a node (separate terminal; first run downloads the model)
python -m node_agent
#    ...or the dependency-free demo path:
NODE_ENGINE=mock python -m node_agent

# 3. Ask for tokens
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Llama-3.2-1B-Instruct-4bit","messages":[{"role":"user","content":"Hello"}]}'

# Streaming: add "stream": true. Dashboard: http://localhost:8080/
```

Run the tests with `pytest` (10 tests, no model download required).

## 6 · What works today, honestly

| Piece | Status |
|---|---|
| Orchestrator, registry, routing, heartbeat eviction | Working, tested |
| Whole-model node agent (MLX / llama.cpp / mock) | Working; MLX path verified on Apple silicon |
| OpenAI-compatible API, JSON + SSE streaming | Working, tested |
| Node failure → clean in-flight session failure | Working, tested |
| Browser WebGPU node (`/compute`) | Protocol demo only — registers and heartbeats; no browser inference yet |
| Layer-shard pipeline mode | Protocol stub, deliberately deferred (see §2) |
| Performance numbers | Not yet published — benchmarks come before claims |

## 7 · What's next

1. **Object-storage data plane** — model artifacts as chunked int8 shards behind a version pointer (Stoa-style [1]); target: cold node productive in ~2 minutes.
2. **Batch API** (`/v1/batches`-style) — latency-tolerant bulk inference (synthetic data, evals, RL rollouts) is where consumer fleets are provably competitive [1]; this is the first product surface.
3. **Browser on-ramp** — WebLLM [4] running whole small models in a tab, joining the fleet as a weaker replica; zero-install contribution stays the differentiator.
4. **Sharding, last** — pipeline parallelism only for models that fit on no single device, gated on cross-stack numerics measurement.

---

## References

1. Miahi, E. *RL Post-Training on Macs.* Pluralis Research Blog, July 2026. — the run that reshaped this project's architecture: whole-model consumer workers, outbound-only star topology, staleness budgets, measurement discipline. https://pluralis.ai/blog/rl-post-training-on-macs/
2. Miahi, E., Belilovsky, E. *Understanding and Exploiting Weight Update Sparsity for Communication-Efficient Distributed RL.* arXiv:2602.03839 (PULSE — ~110× weight-delta compression).
3. Hannun, A., et al. *MLX: Efficient and flexible machine learning on Apple silicon.* https://github.com/ml-explore/mlx — and `mlx-lm`, the node agent's default engine.
4. MLC AI. *WebLLM: High-performance in-browser LLM inference.* https://github.com/mlc-ai/web-llm
5. Kwon, W., et al. *Efficient Memory Management for LLM Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180.
6. EXO Labs. *exo: run your own AI cluster at home.* https://github.com/exo-explore/exo — prior art for multi-Mac inference (wired; this project's fleet is not).
7. Liquid AI. *LFM2.5-8B-A1B: An On-Device Mixture of Experts.* https://www.liquid.ai/blog/lfm2-5-8b-a1b — the model class (big in memory, light in compute) that consumer fleets suit best.
8. Qi, P., et al. *Rethinking the Trust Region in LLM Reinforcement Learning.* arXiv:2602.04879 (DPPO) — and the broader lesson of gating on measured gaps, which informs our cross-stack numerics plan.

Further internal reading: [`docs/analysis/`](docs/analysis/) — a five-part breakdown of the Stoa post and its mapping onto this project; [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/POC_PLAN.md`](docs/POC_PLAN.md) for the original design history.
