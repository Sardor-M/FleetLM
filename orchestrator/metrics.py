"""Fleet metrics - the numbers any claim about FleetLM has to rest on.

Deliberately narrow: counters and simple rates, recorded where work actually
completes. Everything here is derived from events the orchestrator already
sees, so a node cannot inflate its own contribution by reporting a number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class NodeMetrics:
    node_id: str
    gpu_name: str = "unknown"
    joined_at: float = field(default_factory=time.time)
    first_ready_at: float | None = None

    units_completed: int = 0
    units_failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_seconds: float = 0.0

    # Interactive (non-batch) sessions
    sessions_completed: int = 0
    sessions_failed: int = 0

    @property
    def join_to_ready_sec(self) -> float | None:
        if self.first_ready_at is None:
            return None
        return self.first_ready_at - self.joined_at

    @property
    def tokens_per_second(self) -> float:
        """Completion tokens per second of measured generation time."""
        if self.generation_seconds <= 0:
            return 0.0
        return self.completion_tokens / self.generation_seconds

    def snapshot(self) -> dict:
        return {
            "node_id": self.node_id[:8],
            "gpu": self.gpu_name,
            "uptime_sec": round(time.time() - self.joined_at, 1),
            "join_to_ready_sec": (
                round(self.join_to_ready_sec, 1)
                if self.join_to_ready_sec is not None else None
            ),
            "units_completed": self.units_completed,
            "units_failed": self.units_failed,
            "sessions_completed": self.sessions_completed,
            "sessions_failed": self.sessions_failed,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "generation_sec": round(self.generation_seconds, 1),
            "tokens_per_sec": round(self.tokens_per_second, 1),
        }


class FleetMetrics:
    """Cumulative fleet counters, including nodes that have since left."""

    def __init__(self):
        self.started_at = time.time()
        self.nodes: dict[str, NodeMetrics] = {}
        self.departed: list[dict] = []  # snapshots of nodes that disconnected
        self.batches_created = 0
        self.batches_completed = 0
        self.leases_reclaimed = 0  # units returned by disconnect or lease expiry

    # ── Node lifecycle ──────────────────────────────────────────────────

    def node_joined(self, node_id: str, gpu_name: str) -> None:
        self.nodes[node_id] = NodeMetrics(node_id=node_id, gpu_name=gpu_name)

    def node_ready(self, node_id: str) -> None:
        node = self.nodes.get(node_id)
        if node and node.first_ready_at is None:
            node.first_ready_at = time.time()

    def node_left(self, node_id: str) -> None:
        node = self.nodes.pop(node_id, None)
        if node:
            self.departed.append(node.snapshot())

    # ── Work events ─────────────────────────────────────────────────────

    def unit_completed(
        self,
        node_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        seconds: float = 0.0,
    ) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        node.units_completed += 1
        node.prompt_tokens += prompt_tokens
        node.completion_tokens += completion_tokens
        node.generation_seconds += max(0.0, seconds)

    def unit_failed(self, node_id: str) -> None:
        node = self.nodes.get(node_id)
        if node:
            node.units_failed += 1

    def session_completed(
        self, node_id: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        node.sessions_completed += 1
        node.prompt_tokens += prompt_tokens
        node.completion_tokens += completion_tokens

    def session_failed(self, node_id: str) -> None:
        node = self.nodes.get(node_id)
        if node:
            node.sessions_failed += 1

    def leases_returned(self, count: int) -> None:
        self.leases_reclaimed += count

    # ── Reporting ───────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        live = [n.snapshot() for n in self.nodes.values()]
        everyone = live + self.departed

        completion_tokens = sum(n["completion_tokens"] for n in everyone)
        prompt_tokens = sum(n["prompt_tokens"] for n in everyone)
        units = sum(n["units_completed"] for n in everyone)
        failures = sum(n["units_failed"] for n in everyone)
        generation_sec = sum(n["generation_sec"] for n in everyone)
        elapsed = max(1e-6, time.time() - self.started_at)

        ready_times = [
            n["join_to_ready_sec"] for n in everyone
            if n["join_to_ready_sec"] is not None
        ]

        return {
            "fleet_uptime_sec": round(elapsed, 1),
            "nodes_live": len(live),
            "nodes_departed": len(self.departed),
            "batches_created": self.batches_created,
            "batches_completed": self.batches_completed,
            "leases_reclaimed": self.leases_reclaimed,
            "units_completed": units,
            "units_failed": failures,
            "unit_success_rate": (
                round(units / (units + failures), 4) if units + failures else None
            ),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            # Aggregate rate while generating, vs. wall-clock rate for the fleet
            "fleet_tokens_per_sec_generating": (
                round(completion_tokens / generation_sec, 1) if generation_sec else 0.0
            ),
            "fleet_tokens_per_sec_wallclock": round(completion_tokens / elapsed, 2),
            "median_join_to_ready_sec": (
                round(sorted(ready_times)[len(ready_times) // 2], 1)
                if ready_times else None
            ),
            "nodes": live,
            "departed": self.departed[-20:],
        }
