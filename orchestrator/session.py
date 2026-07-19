"""Manages active inference sessions across the distributed fleet."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

from orchestrator.config import settings

logger = logging.getLogger("orchestrator.session")


class SessionState(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETE = "complete"
    FAILED = "failed"


class SessionFailure(Exception):
    """Raised while streaming when the serving node fails or times out."""


@dataclass
class InferenceSession:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model: str = "unknown"
    node_id: str | None = None
    state: SessionState = SessionState.PENDING
    max_tokens: int = 256
    temperature: float = 0.7
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Chunks stream in from the serving node as ("chunk"|"done"|"error", payload)
    _events: asyncio.Queue = field(default_factory=asyncio.Queue, repr=False)

    def push_chunk(self, text: str) -> None:
        self.state = SessionState.GENERATING
        self._events.put_nowait(("chunk", text))

    def complete(self, finish_reason: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.state = SessionState.COMPLETE
        self.finish_reason = finish_reason or "stop"
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self._events.put_nowait(("done", None))

    def fail(self, message: str) -> None:
        if self.state in (SessionState.COMPLETE, SessionState.FAILED):
            return
        self.state = SessionState.FAILED
        self._events.put_nowait(("error", message))

    async def stream(self) -> AsyncIterator[str]:
        """Yield text chunks until the node reports completion.

        Raises SessionFailure on node error, node loss, or timeout.
        """
        deadline = time.monotonic() + settings.generation_timeout_sec
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.fail("generation timed out")
                raise SessionFailure(
                    f"generation exceeded {settings.generation_timeout_sec}s"
                )
            timeout = min(remaining, settings.chunk_timeout_sec)
            try:
                kind, payload = await asyncio.wait_for(self._events.get(), timeout=timeout)
            except asyncio.TimeoutError:
                self.fail("no output from node")
                raise SessionFailure(
                    f"node produced no output for {timeout:.0f}s"
                ) from None
            if kind == "chunk":
                yield payload
            elif kind == "done":
                return
            else:  # error
                raise SessionFailure(payload or "node reported an error")


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

    def fail_sessions_for_node(self, node_id: str) -> None:
        """Fail all in-flight sessions served by a node that disconnected."""
        for session in self.sessions.values():
            if session.node_id == node_id:
                session.fail("serving node disconnected")

    @property
    def active_count(self) -> int:
        return sum(
            1 for s in self.sessions.values()
            if s.state in (SessionState.PENDING, SessionState.GENERATING)
        )
