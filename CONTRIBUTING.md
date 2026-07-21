# Contributing to FleetLM

Two very different ways to help, and both matter.

## 1. Contribute a machine

The fleet is the product. If you have a Mac with spare memory sitting idle:

```bash
pip install -e ".[dev]" && pip install mlx-lm
fleetlm doctor                       # what can this machine contribute?
fleetlm join <fleet-url> --token <token>
```

Your node only makes outbound connections - no open ports, no static IP, no
firewall changes. Ctrl-C to leave; work in flight is retried elsewhere.

## 2. Contribute code

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                            # 32 tests, no model download needed
ruff check orchestrator node_agent tests
```

The `mock` engine runs the entire wire protocol with no dependencies, so you
can develop and test the distributed system without a GPU or model weights:

```bash
uvicorn orchestrator.main:app --port 8080
fleetlm join http://localhost:8080 --engine mock --model demo
```

Conventions - commit format, code style, architecture invariants, and the
rules for publishing numbers - are in [`AGENTS.md`](AGENTS.md). It is written
for AI coding agents but is the same contract for people.

### What we look for

- **Tests that drive the real protocol.** Tests here connect actual
  WebSockets and exercise register → serve → generate → complete, including
  failure paths. Prefer that over mocking internals.
- **Honest status.** If something is a stub, say so in the README table
  rather than letting it look finished. We removed a placeholder engine that
  returned random numbers for exactly this reason.
- **Measurements over claims.** Performance statements need the setup
  written down and should say what they do *not* show. Single unpaired runs
  are smoke tests, not benchmarks.
- **Comments that explain constraints,** not what the next line does.

### Project shape

```
orchestrator/       the single coordination point
  main.py           app, lifespan, dashboard, /health, /metrics
  protocol.py       every wire message, in one file
  batch.py          work-unit queue: leases, retries, results
  session.py        one interactive generation's lifecycle
  metrics.py        fleet counters
  fleet/            registry, heartbeat eviction, routing
  api/              completions (interactive), batches (bulk), nodes (WebSocket)
node_agent/         what runs on a contributor's machine
  __main__.py       connect, heartbeat, pull work, generate
  engine.py         MLX / llama.cpp / mock backends
  cli.py            `fleetlm join`, `fleetlm doctor`
web_compute/        dashboard and the contribute page
```

### Areas that need help

The two most valuable right now:

1. **Cost accounting** - measured energy per million tokens on real hardware,
   compared honestly against cloud batch pricing.
2. **Output verification** - deciding whether a result from an untrusted node
   can be trusted, cheaply. This is the open problem for consumer-hardware
   inference and the most interesting thing in the project.

### Commits

[Conventional Commits](https://www.conventionalcommits.org): `feat:`, `fix:`,
`refactor:`, `perf:`, `test:`, `docs:`, `chore:`, `build:`, `ci:`. Imperative
subject, 72 characters or fewer, no trailing period. Add `!` and a
`BREAKING CHANGE:` footer for wire-protocol changes - a node in the field may
still be running the old one. Full rules and examples in
[`AGENTS.md`](AGENTS.md).

By contributing you agree your work is licensed under the repository's MIT license.
