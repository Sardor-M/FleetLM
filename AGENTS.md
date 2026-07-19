# AGENTS.md

Conventions for AI coding agents working in this repository. Humans should
read this too — it is the same contract either way.

FleetLM serves LLM inference from a fleet of consumer laptops behind an
OpenAI-compatible API. An orchestrator hands out work over outbound
WebSockets; each node holds a whole model and generates locally.

## Commands

```bash
pip install -e ".[dev]"                          # setup
pytest -q                                        # 32 tests, no model needed
ruff check orchestrator node_agent tests         # lint (must pass)

uvicorn orchestrator.main:app --port 8080        # run the orchestrator
fleetlm join http://localhost:8080 --engine mock --model demo   # a node, no GPU
```

The `mock` engine runs the entire wire protocol with zero dependencies. Use it
for anything that isn't specifically about inference quality — never skip an
end-to-end check just because a model isn't available.

## Architecture invariants

Do not break these without an explicit decision recorded in the README:

1. **One node holds one whole model.** No layer sharding, no cross-node
   activations, no pipeline. This was tried, measured, and removed; see the
   README. Do not reintroduce `mode`, layer assignment, or activation
   messages.
2. **Nodes only make outbound connections.** Nothing may require a node to
   accept an inbound connection, open a port, or have a stable address.
3. **Nodes pull work; the orchestrator never pushes it.** A node asks for as
   much as it has room for. Load balancing happens by nodes asking for less.
4. **Work units are self-contained and idempotent.** Any node can run any
   unit; the first result recorded wins; a duplicate is ignored. This is what
   makes node death boring — preserve it.
5. **The orchestrator holds only soft state.** Losing it must not lose
   durable data. Nodes reconnect and re-register.

## Code style

- Ruff, line length 100, target py311. Lint must pass before commit.
- Type hints on function signatures; `from __future__ import annotations`.
- Module docstrings explain *why the module exists*, not what each function
  does. Look at `orchestrator/batch.py` for the intended register.
- Comments state constraints the code cannot express — a protocol
  requirement, a failure mode, a reason for an unobvious choice. Never
  narrate the next line, and never leave notes addressed to a reviewer.
- Keep new modules flat. Single-file packages were deliberately flattened;
  don't create `thing/thing.py` for one module.
- Log at the boundary where something interesting happened, with the short
  node id (`node_id[:8]`), not the full 32 characters.

## Tests

- Tests drive the **real protocol**: connect an actual WebSocket, register,
  serve, generate, complete. Prefer that over mocking internals.
- Every failure path gets a test. Node disconnect, duplicate result, expired
  lease, exhausted retries, bad token — these are the product, not edge
  cases.
- Tests must not download models or need a GPU. CI runs on Linux with no
  accelerator.
- Keep the suite fast (currently well under a second). A slow suite stops
  being run.

## Claims and measurements

This project's credibility rests on not overstating what it does.

- **Never put a performance number in the README that wasn't measured.** An
  earlier version claimed "~80% native performance" and "<200 ms/token" with
  no benchmark behind either; both were removed.
- A single unpaired run is a **smoke test**, not a benchmark. Label it as
  such and write down what it does *not* show.
- Real benchmarks use matched inputs, fixed seeds, and at least three
  replicates, and report only differences that clear the noise.
- The README status table must stay honest. If something is a stub, say so
  there rather than letting it look finished.

## Commits

[Conventional Commits](https://www.conventionalcommits.org). Format:

```
<type>[optional scope][!]: <description>

[body: why, wrapped at 72 columns]

[BREAKING CHANGE: what an implementer must change]
```

Types in use: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`,
`build`, `ci`.

Rules:

- Subject in the imperative mood, lowercase after the type, no trailing
  period, 72 characters or fewer. "add lease reaper", not "Added lease
  reaper." or "adds lease reaper".
- `!` after the type for a breaking change, plus a `BREAKING CHANGE:` footer
  naming exactly what callers must change. Wire-protocol changes are
  breaking — a node in the field may be running the old one.
- The body explains **why**. The diff already shows what.
- One logical change per commit. A refactor and a feature are two commits.

Examples from this repository:

```
refactor!: drop layer sharding and flatten package layout
feat: add batch API and leased work-unit queue
chore: add project config and dependencies
```

Do not commit unless asked. When asked, do not add files the request didn't
cover, and never `git add -A` without checking what it picked up. `docs/` and
`benchmarks/` are intentionally untracked.

## Things that are deliberately absent

Agents often "helpfully" restore these. Don't:

- **Layer sharding / pipeline parallelism** — removed on purpose (README §
  "Three decisions").
- **`requirements.txt`** — `pyproject.toml` is the single source of truth.
- **Browser inference** — `/compute` is an honest capability probe. It must
  not pretend a browser tab can serve until it actually can.
- **A placeholder engine returning fake tensors** — a previous one returned
  `np.random.randn()` while logging "layers loaded". If something cannot do
  the work, it must fail loudly, not fabricate output.
