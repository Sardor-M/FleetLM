# Changelog

All notable changes to FleetLM are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Nothing has been released yet, so every entry below sits under Unreleased and
is grouped by the date the work landed on `main`.

A note on what appears here: performance and cost figures are only recorded once
they have been measured under the rules in [`AGENTS.md`](AGENTS.md). Anything
still unmeasured is listed as pending rather than claimed.

## [Unreleased]

### Added

- Batched decoding on the node. A node now decodes its whole lease in a single
  pass instead of running units one at a time. `generate_batch()` joins the
  engine interface; the base implementation stays sequential so the llama.cpp
  and mock engines are unaffected, and the MLX engine overrides it with a real
  batched decode grouped by sampling temperature.
- Test coverage for the fleet registry, the router, session lifecycle, the
  work-unit store, and the node wire protocol. The suite went from 39 to 77
  tests and still needs no model download, no GPU, and under a second to run.
- `CHANGELOG.md`.
- An editable Excalidraw source and a rendered PNG/SVG of the architecture flow
  alongside the animated GIF.

### Changed

- The engine lock is reentrant. A batched decode that falls back to sequential
  re-enters `generate_stream` while already holding the lock, which would
  deadlock on a plain lock.
- Units in a batch are each credited an equal share of the batch's wall clock,
  so a node's reported tokens/sec measures real throughput rather than falling
  as the batch widens.
- Em dashes across the repository are now plain hyphens.
- The README status table marks browser nodes, the multi-machine fleet, and cost
  per token as expected-to-test rather than describing them as absent.

### Fixed

- A work unit that raises now fails on its own instead of taking down every
  other unit sharing its batch.
- A batched decode that fails as a whole falls back to sequential execution
  rather than dropping the leases it was holding.

### Pending measurement

Tracked as open issues, deliberately unclaimed until they have numbers behind
them: a fleet run across multiple machines and real networks, energy and cost
per million tokens, verification of results returned by untrusted nodes, and the
numeric divergence between inference backends.

## Earlier work

### 2026-07-21

- Flow diagram redrawn to match the architecture as built.

### 2026-07-19

- **Breaking:** layer sharding removed and the package layout flattened. One
  node now holds one whole model. No layer assignment, no cross-node
  activations, no pipeline. A node running the old wire protocol cannot talk to
  this orchestrator.
- Batch API with a leased work-unit queue. Work is handed out as small,
  self-contained, idempotent units: a lease that expires returns its unit to the
  queue, a duplicate result is ignored, and a unit that keeps failing is
  dead-lettered with its error.
- Fleet metrics endpoint, join-token authentication, and the `fleetlm` CLI.
- Contributor and agent conventions written down in `AGENTS.md`.

### 2026-07-16

- Whole-model nodes serving real tokens through an OpenAI-compatible
  `/v1/chat/completions`, with both JSON and SSE streaming.
- Orchestrator scaffold: node registry, routing, heartbeat eviction, and the
  outbound-only WebSocket that lets a node join without opening a port.
- Project renamed to FleetLM.

### 2026-04-01

- Initial project skeleton and packaging.
