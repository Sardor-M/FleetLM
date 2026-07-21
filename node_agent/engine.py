"""Whole-model inference engines for the native node agent.

Phase 1 design: each node holds the entire model and serves complete
generations - no layer sharding, no cross-node activation traffic.

Backends (selected via NODE_ENGINE=auto|ollama|mlx|llama_cpp|mock):
  - ollama:    the easy path. Talks HTTP to a local Ollama daemon, so models
               are managed with `ollama pull` instead of a Python install and
               a HuggingFace download. Runs out of process, so batch work is
               genuinely concurrent.
  - mlx:       Apple silicon, `pip install mlx-lm`. Models by HF repo id,
               e.g. mlx-community/Llama-3.2-1B-Instruct-4bit
  - llama_cpp: any platform, `pip install llama-cpp-python`. Models by
               local GGUF path.
  - mock:      no dependencies; deterministic canned output so the full
               orchestrator <-> node flow can run and be tested without
               downloading a model.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator

logger = logging.getLogger("node_agent.engine")

Messages = list[dict]  # [{"role": ..., "content": ...}, ...]


class EngineError(Exception):
    pass


@dataclass
class BatchItem:
    """One request in a batched generation."""

    messages: Messages
    max_tokens: int = 256
    temperature: float = 0.7


@dataclass
class BatchOutput:
    """Result for a single item. `error` set means that item alone failed."""

    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


class BaseEngine:
    """One engine instance serves one model.

    Interactive generations are single-stream, but batch work runs through
    `generate_batch`, which an engine may implement as a real batched decode.
    """

    name = "base"

    def __init__(self):
        # Serializes access to the model. Reentrant because generate_batch may
        # fall back to generate_stream while already holding it.
        self._lock = threading.RLock()
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

    def generate_batch(self, items: list[BatchItem]) -> list[BatchOutput]:
        """Run `items`, returning one output each, in the order given.

        The default runs them one at a time, so an engine with no batched
        decode path keeps working unchanged.
        """
        return [self._generate_one(item) for item in items]

    def _generate_one(self, item: BatchItem) -> BatchOutput:
        """Run a single item. A raising item fails alone, never the batch."""
        try:
            text = "".join(
                self.generate_stream(item.messages, item.max_tokens, item.temperature)
            )
        except Exception as e:
            return BatchOutput(error=str(e))
        return BatchOutput(
            text=text,
            prompt_tokens=self.last_prompt_tokens,
            completion_tokens=self.last_completion_tokens,
        )


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

    def _encode_prompt(self, messages: Messages) -> list[int]:
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        # apply_chat_template tokenizes by default, but some wrappers hand back
        # a string; batch_generate only accepts token ids.
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        return prompt

    def _sampler_kwargs(self, temperature: float) -> dict:
        try:
            from mlx_lm.sample_utils import make_sampler
        except ImportError:
            logger.warning("mlx_lm.sample_utils unavailable; using default sampling")
            return {}
        return {"sampler": make_sampler(temp=temperature)}

    def generate_stream(
        self, messages: Messages, max_tokens: int, temperature: float
    ) -> Iterator[str]:
        from mlx_lm import stream_generate

        prompt = self._encode_prompt(messages)
        kwargs = {"max_tokens": max_tokens, **self._sampler_kwargs(temperature)}

        with self._lock:
            response = None
            for response in stream_generate(self.model, self.tokenizer, prompt, **kwargs):
                yield response.text
            if response is not None:
                self.last_prompt_tokens = response.prompt_tokens
                self.last_completion_tokens = response.generation_tokens

    def generate_batch(self, items: list[BatchItem]) -> list[BatchOutput]:
        """Decode every item in one pass.

        Decode is memory-bandwidth bound: a step streams the whole weight set
        out of unified memory whatever the batch width, so the read amortises
        across sequences and extra width is close to free.
        """
        if len(items) < 2:
            return super().generate_batch(items)
        try:
            from mlx_lm import batch_generate
        except ImportError:
            logger.warning("mlx-lm has no batch_generate; running units sequentially")
            return super().generate_batch(items)

        outputs: list[BatchOutput] = [BatchOutput() for _ in items]
        # The sampler is shared across a batch_generate call, so units that
        # sample differently cannot ride together.
        groups: dict[float, list[int]] = {}
        for i, item in enumerate(items):
            groups.setdefault(item.temperature, []).append(i)

        with self._lock:
            for temperature, idxs in groups.items():
                try:
                    self._decode_group(batch_generate, items, idxs, outputs, temperature)
                except Exception as e:
                    # A batched decode that blows up must not lose the units.
                    logger.warning(f"Batched decode failed ({e}); falling back to sequential")
                    for i in idxs:
                        outputs[i] = self._generate_one(items[i])
        return outputs

    def _decode_group(self, batch_generate, items, idxs, outputs, temperature) -> None:
        """One batch_generate call over `idxs`. Caller holds the lock."""
        prompts = [self._encode_prompt(items[i].messages) for i in idxs]
        response = batch_generate(
            self.model,
            self.tokenizer,
            prompts=prompts,
            max_tokens=[items[i].max_tokens for i in idxs],
            verbose=False,
            **self._sampler_kwargs(temperature),
        )
        for slot, i in enumerate(idxs):
            text = response.texts[slot]
            outputs[i] = BatchOutput(
                text=text,
                prompt_tokens=len(prompts[slot]),
                # BatchStats reports batch-wide totals only, so per-unit
                # completions are re-encoded to attribute them.
                completion_tokens=len(self.tokenizer.encode(text, add_special_tokens=False)),
            )


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


class OllamaEngine(BaseEngine):
    """Ollama daemon backend, reached over HTTP.

    Ollama runs as its own process, so the lock that serialises the in-process
    backends is not needed for batch work: `generate_batch` fires concurrent
    requests and lets the daemon schedule them. Batch width on the daemon side
    is set with OLLAMA_NUM_PARALLEL, not from here.

    `generate_stream` still takes the lock, because it reports usage through
    `last_*_tokens` on the instance and concurrent streams would race on it.
    Batch results carry their counts per unit, so that path needs no lock.
    """

    name = "ollama"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(self, host: str | None = None):
        super().__init__()
        raw = host or os.environ.get("OLLAMA_HOST") or self.DEFAULT_HOST
        if "//" not in raw:
            raw = f"http://{raw}"
        self.host = raw.rstrip("/")

    # ── daemon helpers ──────────────────────────────────────────────────

    def _get(self, path: str, timeout: float = 10.0):
        import httpx

        try:
            r = httpx.get(f"{self.host}{path}", timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise EngineError(
                f"Ollama is not reachable at {self.host} ({e}). "
                f"Start it with: ollama serve"
            ) from e

    def available_models(self) -> list[str]:
        return [m.get("name", "") for m in self._get("/api/tags").get("models", [])]

    def _resolve(self, model_id: str, available: list[str]) -> str | None:
        """Ollama tags models as `name:tag`; accept a bare name for `:latest`."""
        if model_id in available:
            return model_id
        if ":" not in model_id and f"{model_id}:latest" in available:
            return f"{model_id}:latest"
        return None

    def load(self, model_id: str) -> None:
        available = self.available_models()
        resolved = self._resolve(model_id, available)
        if resolved is None:
            have = ", ".join(sorted(available)) or "none"
            raise EngineError(
                f"Ollama has no model '{model_id}'. Pull it with: "
                f"ollama pull {model_id}   (currently pulled: {have})"
            )
        self.model_id = resolved
        logger.info(f"Ollama ready at {self.host}, serving {resolved}")

    def _payload(self, messages: Messages, max_tokens: int, temperature: float,
                 stream: bool) -> dict:
        return {
            "model": self.model_id,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    # ── generation ──────────────────────────────────────────────────────

    def generate_stream(
        self, messages: Messages, max_tokens: int, temperature: float
    ) -> Iterator[str]:
        import httpx

        with self._lock:
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
                with client.stream(
                    "POST", f"{self.host}/api/chat",
                    json=self._payload(messages, max_tokens, temperature, stream=True),
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if chunk.get("error"):
                            raise EngineError(f"ollama: {chunk['error']}")
                        piece = chunk.get("message", {}).get("content", "")
                        if piece:
                            yield piece
                        if chunk.get("done"):
                            # Counts only appear on the final chunk.
                            self.last_prompt_tokens = chunk.get("prompt_eval_count", 0)
                            self.last_completion_tokens = chunk.get("eval_count", 0)

    def _complete(self, item: BatchItem) -> BatchOutput:
        """One non-streaming completion. No shared state, so it is safe to run
        concurrently with other calls."""
        import httpx

        try:
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
                r = client.post(
                    f"{self.host}/api/chat",
                    json=self._payload(
                        item.messages, item.max_tokens, item.temperature, stream=False
                    ),
                )
                r.raise_for_status()
                body = r.json()
            if body.get("error"):
                return BatchOutput(error=f"ollama: {body['error']}")
            return BatchOutput(
                text=body.get("message", {}).get("content", ""),
                prompt_tokens=body.get("prompt_eval_count", 0),
                completion_tokens=body.get("eval_count", 0),
            )
        except Exception as e:
            return BatchOutput(error=str(e))

    def generate_batch(self, items: list[BatchItem]) -> list[BatchOutput]:
        """Run the whole lease concurrently against the daemon.

        Deliberately does not take the engine lock: the work happens in another
        process, and every result carries its own counts, so there is no shared
        state to protect.
        """
        if len(items) == 1:
            return [self._complete(items[0])]
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            return list(pool.map(self._complete, items))


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


def ollama_is_running(host: str | None = None) -> bool:
    """True when an Ollama daemon answers. Used for engine auto-detection."""
    try:
        OllamaEngine(host).available_models()
        return True
    except Exception:
        return False


def create_engine(kind: str = "auto") -> BaseEngine:
    """Resolve an engine by name.

    'auto' prefers a running Ollama daemon, because it is the path that needs
    no Python inference stack, then falls back to mlx, llama_cpp, and mock.
    """
    kind = (kind or "auto").lower()

    if kind == "ollama":
        return OllamaEngine()
    if kind == "mlx":
        return MlxEngine()
    if kind == "llama_cpp":
        return LlamaCppEngine()
    if kind == "mock":
        return MockEngine()
    if kind != "auto":
        raise EngineError(
            f"Unknown engine '{kind}' (use auto|ollama|mlx|llama_cpp|mock)"
        )

    if ollama_is_running():
        logger.info("Found a running Ollama daemon; using it")
        return OllamaEngine()
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
        "No inference engine found; using the mock engine. For real inference, "
        "install Ollama (ollama.com) and run: ollama pull llama3.2"
    )
    return MockEngine()
