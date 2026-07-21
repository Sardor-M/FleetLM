"""The Ollama engine's contract, without needing a daemon.

CI has no Ollama running and must never pull a model, so everything here stubs
the HTTP layer. What is worth pinning is the behaviour a contributor actually
hits: a daemon that is not running, a model that was never pulled, Ollama's
`name:tag` convention, and the promise that batch work runs concurrently and
keeps its results in order.
"""

import pytest

from node_agent.engine import BatchItem, BatchOutput, EngineError, OllamaEngine


def _items(*prompts, max_tokens=16, temperature=0.0):
    return [
        BatchItem(
            messages=[{"role": "user", "content": p}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for p in prompts
    ]


# ── reaching the daemon ─────────────────────────────────────────────────

def test_host_defaults_to_the_local_daemon():
    assert OllamaEngine().host == "http://localhost:11434"


def test_a_host_without_a_scheme_is_accepted():
    """OLLAMA_HOST is conventionally set as host:port, with no scheme."""
    assert OllamaEngine("192.168.1.50:11434").host == "http://192.168.1.50:11434"


def test_a_trailing_slash_does_not_double_up_in_urls():
    assert OllamaEngine("http://localhost:11434/").host == "http://localhost:11434"


def test_an_unreachable_daemon_says_how_to_start_it():
    engine = OllamaEngine("http://127.0.0.1:1")  # nothing listens here
    with pytest.raises(EngineError, match="ollama serve"):
        engine.load("llama3.2")


# ── model resolution ────────────────────────────────────────────────────

def test_a_bare_name_resolves_to_the_latest_tag(monkeypatch):
    """`ollama pull llama3.2` stores `llama3.2:latest`; users type the bare name."""
    engine = OllamaEngine()
    monkeypatch.setattr(engine, "available_models", lambda: ["llama3.2:latest"])

    engine.load("llama3.2")

    assert engine.model_id == "llama3.2:latest"


def test_an_exact_tag_is_used_as_given(monkeypatch):
    engine = OllamaEngine()
    monkeypatch.setattr(engine, "available_models", lambda: ["llama3.2:1b", "llama3.2:latest"])

    engine.load("llama3.2:1b")

    assert engine.model_id == "llama3.2:1b"


def test_a_model_that_was_never_pulled_names_the_pull_command(monkeypatch):
    engine = OllamaEngine()
    monkeypatch.setattr(engine, "available_models", lambda: ["phi4-mini:latest"])

    with pytest.raises(EngineError) as excinfo:
        engine.load("llama3.2")

    message = str(excinfo.value)
    assert "ollama pull llama3.2" in message
    assert "phi4-mini:latest" in message  # tells them what they do have


# ── batch behaviour ─────────────────────────────────────────────────────

def test_batch_returns_one_result_per_item_in_order(monkeypatch):
    engine = OllamaEngine()
    monkeypatch.setattr(
        engine, "_complete",
        lambda item: BatchOutput(text=item.messages[0]["content"].upper()),
    )

    outputs = engine.generate_batch(_items("alpha", "bravo", "charlie"))

    assert [o.text for o in outputs] == ["ALPHA", "BRAVO", "CHARLIE"]


def test_batch_runs_items_concurrently(monkeypatch):
    """The daemon is a separate process, so a lease should not be serialised.

    Each stub sleeps; run sequentially the batch would take n * delay, so a
    wall clock well under that is the evidence of overlap.
    """
    import time

    delay = 0.2
    engine = OllamaEngine()

    def slow(item):
        time.sleep(delay)
        return BatchOutput(text="ok")

    monkeypatch.setattr(engine, "_complete", slow)

    started = time.monotonic()
    outputs = engine.generate_batch(_items("a", "b", "c", "d"))
    elapsed = time.monotonic() - started

    assert len(outputs) == 4
    assert elapsed < delay * 2, f"4 items took {elapsed:.2f}s - they ran serially"


def test_one_failing_item_does_not_sink_the_batch(monkeypatch):
    engine = OllamaEngine()

    def flaky(item):
        if "boom" in item.messages[0]["content"]:
            return BatchOutput(error="ollama: model runner crashed")
        return BatchOutput(text="fine")

    monkeypatch.setattr(engine, "_complete", flaky)

    outputs = engine.generate_batch(_items("ok", "boom", "ok"))

    assert outputs[0].error is None
    assert "crashed" in outputs[1].error
    assert outputs[2].error is None


def test_a_single_item_still_produces_one_output(monkeypatch):
    """The n==1 path skips the thread pool; it must not skip the result."""
    engine = OllamaEngine()
    monkeypatch.setattr(engine, "_complete", lambda item: BatchOutput(text="solo"))

    assert [o.text for o in engine.generate_batch(_items("only"))] == ["solo"]
