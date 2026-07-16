"""Manages active inference sessions across the distributed pipeline."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("orchestrator.session")


class SessionState(str, Enum):
    PREFILL = "prefill"
    DECODING = "decoding"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class InferenceSession:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model: str = "llama-3-8b"
    prompt_tokens: list[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    pipeline_node_ids: list[str] = field(default_factory=list)
    state: SessionState = SessionState.PREFILL
    max_tokens: int = 256
    temperature: float = 0.7

    # Async event for waiting on next activation
    _activation_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _activation_data: bytes | None = field(default=None, repr=False)

    def set_activation(self, data: bytes) -> None:
        """Called when a node sends back an activation result."""
        self._activation_data = data
        self._activation_event.set()

    async def wait_activation(self, timeout: float = 10.0) -> bytes | None:
        """Wait for the next activation from a pipeline node."""
        try:
            await asyncio.wait_for(self._activation_event.wait(), timeout=timeout)
            data = self._activation_data
            self._activation_event.clear()
            self._activation_data = None
            return data
        except asyncio.TimeoutError:
            logger.error(f"Session {self.id}: activation timeout after {timeout}s")
            return None


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, InferenceSession] = {}

    def create(self, **kwargs) -> InferenceSession:
        session = InferenceSession(**kwargs)
        self.sessions[session.id] = session
        logger.info(f"Session created: {session.id}")
        return session

    def get(self, session_id: str) -> InferenceSession | None:
        return self.sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    @property
    def active_count(self) -> int:
        return sum(
            1 for s in self.sessions.values()
            if s.state in (SessionState.PREFILL, SessionState.DECODING)
        )
