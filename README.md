# Distributed LLM Inference Platform

A platform that harnesses idle consumer compute (laptops, desktops) to run distributed LLM inference, selling capacity to B2B customers via an OpenAI-compatible API.

## How It Works

1. **Contributors** visit your web platform in Chrome - **zero install, just open a URL**
2. The browser loads model weight shards via WebGPU (cached in browser storage)
3. The **Orchestrator** assigns transformer layers to each browser tab based on GPU capabilities
4. **B2B customers** send inference requests via REST API
5. The orchestrator routes requests through a **pipeline** of browser-based compute nodes
6. Each browser tab runs its assigned transformer layers via WebGPU compute shaders
7. Generated tokens stream back to the customer

## Architecture

```
B2B API Request → Orchestrator → [Browser Tab A: L0-15] → [Browser Tab B: L16-31] → Token Output
                      ↑               ↑ (WebGPU)              ↑ (WebGPU)
                  Scheduler          User Laptop 1           User Laptop 2
                  + Router         (opens your URL)        (opens your URL)
```

**Key design choices**:
- **Browser-native GPU compute** via WebGPU/WebNN - no extension, no native install needed
- **Pipeline parallelism** (not tensor parallelism) - only 8KB per token between nodes
- **WebLLM** for optimized WebGPU inference kernels (~80% native performance)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestrator | Python (FastAPI + uvicorn) |
| Browser Compute | WebGPU + WebLLM (or WebNN when stable) |
| Communication | WebSocket (binary frames for activations) |
| B2B API | OpenAI-compatible REST |
| Weight Caching | Browser Cache API / IndexedDB |
| GPU Access | Chrome Dawn engine (Metal/Vulkan/D3D12) |
| Optional Native Path | Python + llama-cpp-python (for power users) |

### Why Browser-Native (WebGPU)?

- **Zero install** - contributor just opens a URL in Chrome
- **No extension store approval** needed
- **Cross-platform automatically** - Chrome handles Metal/Vulkan/D3D12
- **Sandboxed & secure** - can't access filesystem, users trust browsers
- **~80% native GPU performance** via WebLLM's optimized WebGPU kernels
- **WebNN (origin trial)** will add NPU/hardware accelerator access for even better performance

### Why Python for the Orchestrator?

The orchestrator spends 99.9% of its time waiting on network I/O (50-200ms per hop). Python adds <0.5ms overhead - **0.3% of total latency**:

- **10x faster development** - FastAPI server in 50 lines
- **Native ML ecosystem** - PyTorch, HuggingFace, numpy
- **Auto-generated API docs** - Swagger UI at `/docs` for free

## Documentation

- [`docs/POC_PLAN.md`](docs/POC_PLAN.md) - Detailed 5-phase POC plan with timelines and success criteria
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - Technical architecture, protocols, and data flow specs
- [`docs/BROWSER_GPU_APIS.md`](docs/BROWSER_GPU_APIS.md) - WebGPU, WebNN, and Chrome AI APIs deep-dive
- [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) - Project structure, starter code, and setup guide
- [`docs/architecture-flow.drawio`](docs/architecture-flow.drawio) - Visual diagrams (open in draw.io or diagrams.net)

## Quick Start

```bash
# Prerequisites: Python 3.11+, Chrome 120+

# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Install llama-cpp-python with GPU support (macOS Metal)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

# Start orchestrator
uvicorn orchestrator.main:app --host 0.0.0.0 --port 8080 --reload

# Start a compute node (separate terminal)
ORCHESTRATOR_URL=ws://localhost:8080/nodes/ws python -m node_agent

# Test the API
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3-8b","messages":[{"role":"user","content":"Hello"}]}'

# View auto-generated API docs
open http://localhost:8080/docs
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Decode latency | <200ms/token | Over internet (50ms RTT), 2-stage pipeline |
| Throughput (single request) | ~6 tok/s | Limited by network latency, not Python |
| Throughput (batched, 8 requests) | ~48 tok/s aggregate | Pipeline parallelism amortizes latency |
| Failover time | <10 seconds | Detect + reroute + KV-cache recompute |
| Min VRAM per node | 2 GB | For 16 layers of 8B model (Q4) |
| Python overhead | <0.5ms/token | 0.3% of total latency |
