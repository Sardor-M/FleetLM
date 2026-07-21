"""Batch jobs and the leased work-unit queue.

The fleet's unit of work is one small, self-contained, idempotent work unit:
a single chat request that any node can run and whose result is written once.
That shape is what makes node churn boring - a lease that expires simply
returns its unit to the queue, and a duplicate result is ignored rather than
corrupting anything.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from orchestrator.config import settings

logger = logging.getLogger("orchestrator.batch")


class UnitState(str, Enum):
    PENDING = "pending"
    LEASED = "leased"
    COMPLETE = "complete"
    FAILED = "failed"


class BatchState(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class WorkUnit:
    id: str
    batch_id: str
    index: int  # position in the submitted batch, so results can be ordered
    messages: list[dict]
    model: str | None = None
    max_tokens: int = 256
    temperature: float = 0.7

    state: UnitState = UnitState.PENDING
    attempts: int = 0
    node_id: str | None = None
    lease_expires_at: float = 0.0
    # Split so the two halves of a unit's life can be told apart: how long it
    # waited for a free node, and how long that node then took. Adding machines
    # should shrink the first and leave the second alone - if the second moves
    # instead, the fleet is contending, not scaling.
    created_at: float = field(default_factory=time.time)
    leased_at: float = 0.0
    result_text: str | None = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    served_by: str | None = None  # model the node actually ran

    def payload(self) -> dict:
        """What the node needs to run this unit."""
        return {
            "unit_id": self.id,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def result_record(self) -> dict:
        """One JSONL line of the batch output."""
        record = {
            "index": self.index,
            "unit_id": self.id,
            "status": self.state.value,
            "attempts": self.attempts,
        }
        if self.state == UnitState.COMPLETE:
            record["response"] = {
                "role": "assistant",
                "content": self.result_text or "",
            }
            record["usage"] = {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            }
            # Which model actually ran, so a result is never anonymous about
            # its own provenance. Falls back to what was asked for when a node
            # is too old to report it.
            record["model"] = self.served_by or self.model
        else:
            record["error"] = self.error or "unknown error"
        return record


@dataclass
class Batch:
    id: str
    model: str | None
    created_at: float = field(default_factory=time.time)
    state: BatchState = BatchState.IN_PROGRESS
    unit_ids: list[str] = field(default_factory=list)
    completed_at: float | None = None


class BatchStore:
    """In-memory batches and their work units.

    Phase 2b moves payloads and results to object storage; the lease/retry
    semantics here stay the same.
    """

    def __init__(self, metrics=None):
        self.batches: dict[str, Batch] = {}
        self.units: dict[str, WorkUnit] = {}
        self._pending: list[str] = []  # FIFO of unit ids ready to lease
        self._lock = asyncio.Lock()
        self.metrics = metrics

    # ── Submission ──────────────────────────────────────────────────────

    async def create_batch(self, requests: list[dict], model: str | None) -> Batch:
        batch = Batch(id=f"batch_{uuid.uuid4().hex[:12]}", model=model)
        async with self._lock:
            for i, req in enumerate(requests):
                unit = WorkUnit(
                    id=f"unit_{uuid.uuid4().hex[:12]}",
                    batch_id=batch.id,
                    index=i,
                    messages=req.get("messages", []),
                    # `or`, not a .get default: the API model_dumps a
                    # BatchRequestItem whose `model` key is present and
                    # None, so a default would never be reached and every
                    # unit would lose the batch's model.
                    model=req.get("model") or model,
                    max_tokens=req.get("max_tokens", 256),
                    temperature=req.get("temperature", 0.7),
                )
                self.units[unit.id] = unit
                batch.unit_ids.append(unit.id)
                self._pending.append(unit.id)
            self.batches[batch.id] = batch
        if self.metrics:
            self.metrics.batches_created += 1
        logger.info(f"Batch {batch.id} created with {len(batch.unit_ids)} units")
        return batch

    # ── Leasing ─────────────────────────────────────────────────────────

    async def lease(
        self, node_id: str, count: int, node_model: str | None = None
    ) -> list[WorkUnit]:
        """Hand out up to `count` pending units this node is able to serve.

        A unit that names a model is only handed to a node serving that model.
        Without this a fleet running mixed models answers a request for one
        with another, and a single batch can come back as a silent mixture.
        A unit that names no model, or a node that reports none, stays
        eligible for anything.

        Units this node cannot take keep their place in the queue rather than
        being dropped - another node will pick them up.
        """
        leased: list[WorkUnit] = []
        now = time.time()
        async with self._lock:
            still_pending: list[str] = []
            for i, uid in enumerate(self._pending):
                if len(leased) >= count:
                    still_pending.extend(self._pending[i:])
                    break
                unit = self.units.get(uid)
                if unit is None or unit.state != UnitState.PENDING:
                    continue  # stale queue entry
                batch = self.batches.get(unit.batch_id)
                if batch is None or batch.state == BatchState.CANCELLED:
                    continue  # the batch went away
                if unit.model and node_model and unit.model != node_model:
                    still_pending.append(uid)  # not for this node
                    continue
                unit.state = UnitState.LEASED
                unit.node_id = node_id
                unit.attempts += 1
                unit.leased_at = now
                unit.lease_expires_at = now + settings.lease_duration_sec
                leased.append(unit)
            self._pending = still_pending
        if leased:
            logger.info(f"Leased {len(leased)} units to node {node_id[:8]}")
        return leased

    async def complete(
        self,
        unit_id: str,
        node_id: str,
        text: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        generation_sec: float = 0.0,
        served_by: str | None = None,
    ) -> None:
        """Record a result. The first result to arrive wins; later ones are ignored."""
        async with self._lock:
            unit = self.units.get(unit_id)
            if unit is None:
                logger.warning(f"Result for unknown unit {unit_id}")
                return
            if unit.state in (UnitState.COMPLETE, UnitState.FAILED):
                logger.debug(f"Duplicate result for {unit_id} from {node_id[:8]}, ignored")
                return
            unit.state = UnitState.COMPLETE
            unit.result_text = text
            unit.prompt_tokens = prompt_tokens
            unit.completion_tokens = completion_tokens
            unit.node_id = node_id
            unit.served_by = served_by
            unit.lease_expires_at = 0.0
            # Measured here rather than reported by the node: a node cannot
            # understate how long the fleet took to answer.
            now = time.time()
            queue_sec = max(0.0, unit.leased_at - unit.created_at) if unit.leased_at else 0.0
            service_sec = max(0.0, now - unit.leased_at) if unit.leased_at else 0.0
            retries = max(0, unit.attempts - 1)
            self._maybe_finish_batch(unit.batch_id)
        if self.metrics:
            self.metrics.unit_completed(
                node_id,
                prompt_tokens,
                completion_tokens,
                generation_sec,
                queue_sec=queue_sec,
                service_sec=service_sec,
                retries=retries,
            )

    async def fail(self, unit_id: str, node_id: str, error: str) -> None:
        """A node reported failure: requeue unless the unit is out of attempts."""
        async with self._lock:
            unit = self.units.get(unit_id)
            if unit is None or unit.state in (UnitState.COMPLETE, UnitState.FAILED):
                return
            unit.error = error
            if unit.attempts >= settings.max_unit_attempts:
                unit.state = UnitState.FAILED
                logger.error(
                    f"Unit {unit_id} failed permanently after {unit.attempts} attempts: {error}"
                )
                self._maybe_finish_batch(unit.batch_id)
                if self.metrics:
                    self.metrics.unit_failed(node_id)
            else:
                unit.state = UnitState.PENDING
                unit.node_id = None
                unit.lease_expires_at = 0.0
                self._pending.append(unit.id)
                logger.warning(
                    f"Unit {unit_id} failed on {node_id[:8]} "
                    f"(attempt {unit.attempts}), requeued: {error}"
                )

    async def release_node(self, node_id: str) -> int:
        """Return every unit leased to a departed node back to the queue."""
        requeued = 0
        async with self._lock:
            for unit in self.units.values():
                if unit.state == UnitState.LEASED and unit.node_id == node_id:
                    unit.state = UnitState.PENDING
                    unit.node_id = None
                    unit.lease_expires_at = 0.0
                    self._pending.append(unit.id)
                    requeued += 1
        if requeued:
            logger.info(f"Requeued {requeued} units from departed node {node_id[:8]}")
            if self.metrics:
                self.metrics.leases_returned(requeued)
        return requeued

    async def expire_leases(self) -> int:
        """Reclaim units whose lease ran out (node hung, crashed, or went dark)."""
        now = time.time()
        expired = 0
        async with self._lock:
            for unit in self.units.values():
                if unit.state == UnitState.LEASED and now > unit.lease_expires_at:
                    unit.state = UnitState.PENDING
                    unit.node_id = None
                    unit.lease_expires_at = 0.0
                    self._pending.append(unit.id)
                    expired += 1
        if expired:
            logger.warning(f"Expired {expired} stale leases, units requeued")
            if self.metrics:
                self.metrics.leases_returned(expired)
        return expired

    # ── Status ──────────────────────────────────────────────────────────

    def _maybe_finish_batch(self, batch_id: str) -> None:
        """Mark a batch completed once no unit is outstanding. Caller holds the lock."""
        batch = self.batches.get(batch_id)
        if batch is None or batch.state != BatchState.IN_PROGRESS:
            return
        for uid in batch.unit_ids:
            unit = self.units.get(uid)
            if unit and unit.state in (UnitState.PENDING, UnitState.LEASED):
                return
        batch.state = BatchState.COMPLETED
        batch.completed_at = time.time()
        if self.metrics:
            self.metrics.batches_completed += 1
        logger.info(f"Batch {batch_id} completed")

    async def cancel_batch(self, batch_id: str) -> bool:
        async with self._lock:
            batch = self.batches.get(batch_id)
            if batch is None or batch.state != BatchState.IN_PROGRESS:
                return False
            batch.state = BatchState.CANCELLED
            for uid in batch.unit_ids:
                unit = self.units.get(uid)
                if unit and unit.state in (UnitState.PENDING, UnitState.LEASED):
                    unit.state = UnitState.FAILED
                    unit.error = "batch cancelled"
        logger.info(f"Batch {batch_id} cancelled")
        return True

    def get_batch(self, batch_id: str) -> Batch | None:
        return self.batches.get(batch_id)

    def counts(self, batch_id: str) -> dict:
        batch = self.batches.get(batch_id)
        if batch is None:
            return {}
        tally = {s.value: 0 for s in UnitState}
        for uid in batch.unit_ids:
            unit = self.units.get(uid)
            if unit:
                tally[unit.state.value] += 1
        return {
            "total": len(batch.unit_ids),
            "completed": tally[UnitState.COMPLETE.value],
            "failed": tally[UnitState.FAILED.value],
            "in_flight": tally[UnitState.LEASED.value],
            "pending": tally[UnitState.PENDING.value],
        }

    def results(self, batch_id: str) -> list[dict]:
        batch = self.batches.get(batch_id)
        if batch is None:
            return []
        units = [self.units[uid] for uid in batch.unit_ids if uid in self.units]
        return [u.result_record() for u in sorted(units, key=lambda u: u.index)]

    def usage(self, batch_id: str) -> dict:
        batch = self.batches.get(batch_id)
        if batch is None:
            return {}
        prompt = completion = 0
        for uid in batch.unit_ids:
            unit = self.units.get(uid)
            if unit and unit.state == UnitState.COMPLETE:
                prompt += unit.prompt_tokens
                completion += unit.completion_tokens
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    @property
    def pending_count(self) -> int:
        return sum(1 for u in self.units.values() if u.state == UnitState.PENDING)

    def summary(self) -> dict:
        return {
            "batches": len(self.batches),
            "in_progress": sum(
                1 for b in self.batches.values() if b.state == BatchState.IN_PROGRESS
            ),
            "pending_units": self.pending_count,
        }
