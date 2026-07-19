"""Whole-model inference engines for the native node agent.

Phase 1 design: each node holds the entire model and serves complete
generations — no layer sharding, no cross-node activation traffic.

Backends (selected via NODE_ENGINE=auto|mlx|llama_cpp|mock):
  - mlx:       Apple silicon, `pip install mlx-lm`. Models by HF repo id,
               e.g. mlx-community/Llama-3.2-1B-Instruct-4bit
  - llama_cpp: any platform, `pip install llama-cpp-python`. Models by
               local GGUF path.
  - mock:      no dependencies; deterministic canned output so the full
               orchestrator <-> node flow can run and be tested without
               downloading a model.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Iterator

logger = logging.getLogger("node_agent.engine")

Messages = list[dict]  # [{"role": ..., "content": ...}, ...]


class EngineError(Exception):
    pass


class BaseEngine:
    """One engine instance serves one model, one generation at a time."""

    name = "base"

    def __init__(self):
        # MLX and llama.cpp are single-stream; serialize concurrent requests.
        self._lock = threading.Lock()
        self.model_id: str | None = None
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def load(self, model_id: str) -> None:
        raise NotImplementedError

    def generate_stream(
        self, messages: Messages, max_tokens: int, temperature: float
    ) -> Iterator[str]:
        """Yield text pieces; set last_*_tokens before finishing."""
        raise NotImplementedError


class MlxEngine(BaseEngine):
    """mlx-lm backend for Apple silicon."""

    name = "mlx"

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None

    def load(self, model_id: str) -> None:
        from mlx_lm import load

        logger.info(f"Loading {model_id} via mlx-lm (downloads on first run)...")
        t0 = time.time()
        self.model, self.tokenizer = load(model_id)
        self.model_id = model_id
        logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    def generate_stream(
        self, messages: Messages, max_tokens: int, temperature: float
    ) -> Iterator[str]:
        from mlx_lm import stream_generate

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        kwargs = {"max_tokens": max_tokens}
        try:
            from mlx_lm.sample_utils import make_sampler

            kwargs["sampler"] = make_sampler(temp=temperature)
        except ImportError:
            logger.warning("mlx_lm.sample_utils unavailable; using default sampling")

        with self._lock:
            response = None
            for response in stream_generate(self.model, self.tokenizer, prompt, **kwargs):
                yield response.text
            if response is not None:
                self.last_prompt_tokens = response.prompt_tokens
                self.last_completion_tokens = response.generation_tokens


class LlamaCppEngine(BaseEngine):
    """llama-cpp-python backend (GGUF models, any platform)."""

    name = "llama_cpp"

    def __init__(self):
        super().__init__()
        self.llm = None

    def load(self, model_id: str) -> None:
        from llama_cpp import Llama

        logger.info(f"Loading GGUF model from {model_id} via llama.cpp...")
        t0 = time.time()
        self.llm = Llama(
            model_path=model_id,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )
        self.model_id = model_id
        logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    def generate_stream(
        self, messages: Messages, max_tokens: int, temperature: float
    ) -> Iterator[str]:
        with self._lock:
            completion_tokens = 0
            stream = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    completion_tokens += 1
                    yield text
            # llama.cpp streaming does not report usage; approximate.
            self.last_prompt_tokens = sum(
                len(m.get("content", "").split()) for m in messages
            )
            self.last_completion_tokens = completion_tokens


class MockEngine(BaseEngine):
    """Dependency-free engine that echoes the prompt. For tests and plumbing demos."""

    name = "mock"

    def load(self, model_id: str) -> None:
        self.model_id = model_id
        logger.info(f"Mock engine ready (pretending to serve {model_id})")

    def generate_stream(
        self, messages: Messages, max_tokens: int, temperature: float
    ) -> Iterator[str]:
        with self._lock:
            last_user = next(
                (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            words = f"[mock:{self.model_id}] You said: {last_user}".split(" ")
            words = words[: max(1, max_tokens)]
            for i, word in enumerate(words):
                yield word if i == 0 else f" {word}"
                time.sleep(0.005)
            self.last_prompt_tokens = sum(
                len(m.get("content", "").split()) for m in messages
            )
            self.last_completion_tokens = len(words)


def create_engine(kind: str = "auto") -> BaseEngine:
    """Resolve an engine by name; 'auto' prefers mlx, then llama_cpp, then mock."""
    kind = (kind or "auto").lower()

    if kind == "mlx":
        return MlxEngine()
    if kind == "llama_cpp":
        return LlamaCppEngine()
    if kind == "mock":
        return MockEngine()
    if kind != "auto":
        raise EngineError(f"Unknown engine '{kind}' (use auto|mlx|llama_cpp|mock)")

    try:
        import mlx_lm  # noqa: F401
        return MlxEngine()
    except ImportError:
        pass
    try:
        import llama_cpp  # noqa: F401
        return LlamaCppEngine()
    except ImportError:
        pass
    logger.warning(
        "Neither mlx-lm nor llama-cpp-python installed; using mock engine. "
        "Install one for real inference: pip install mlx-lm"
    )
    return MockEngine()
