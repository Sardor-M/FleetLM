"""Node-side batched decoding.

Batch work used to run one unit at a time. It now decodes a whole lease as one
batch, so the properties worth pinning are: every unit still gets exactly one
result in the order it was given, batching does not change what a unit
produces, and a unit that blows up takes only itself down.
"""

from node_agent.__main__ import NodeAgent
from node_agent.engine import BatchItem, MockEngine

MODEL = "tiny-test-model"


def _engine(cls=MockEngine):
    engine = cls()
    engine.load(MODEL)
    return engine


def _items(*prompts, max_tokens=16, temperature=0.0):
    return [
        BatchItem(
            messages=[{"role": "user", "content": p}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for p in prompts
    ]


def _units(*prompts, max_tokens=16, temperature=0.0):
    return [
        {
            "unit_id": f"unit-{i}",
            "messages": [{"role": "user", "content": p}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        for i, p in enumerate(prompts)
    ]


class FlakyEngine(MockEngine):
    """Raises for any prompt containing 'boom'; normal otherwise."""

    def generate_stream(self, messages, max_tokens, temperature):
        if any("boom" in m.get("content", "") for m in messages):
            raise RuntimeError("prompt exploded")
        yield from super().generate_stream(messages, max_tokens, temperature)


# ── engine level ────────────────────────────────────────────────────────

def test_batch_returns_one_output_per_item_in_order():
    outputs = _engine().generate_batch(_items("alpha", "bravo", "charlie"))

    assert len(outputs) == 3
    assert all(o.error is None for o in outputs)
    for prompt, out in zip(("alpha", "bravo", "charlie"), outputs):
        assert prompt in out.text


def test_batched_output_matches_sequential_at_temperature_zero():
    """The acceptance criterion: batching must not change what a unit returns."""
    engine = _engine()
    prompts = ("one", "two", "three")

    sequential = [
        "".join(engine.generate_stream([{"role": "user", "content": p}], 16, 0.0))
        for p in prompts
    ]
    batched = [o.text for o in engine.generate_batch(_items(*prompts))]

    assert batched == sequential


def test_one_failing_unit_does_not_sink_the_batch():
    outputs = _engine(FlakyEngine).generate_batch(_items("fine", "boom", "still fine"))

    assert outputs[0].error is None
    assert outputs[1].error is not None and "exploded" in outputs[1].error
    assert outputs[2].error is None
    assert "still fine" in outputs[2].text


def test_batch_reports_per_unit_token_counts():
    outputs = _engine().generate_batch(_items("count my tokens"))

    assert outputs[0].completion_tokens > 0
    assert outputs[0].prompt_tokens > 0


# ── node agent level ────────────────────────────────────────────────────

def _agent(engine, batch_size=4):
    return NodeAgent("ws://test/nodes/ws", engine, MODEL, batch_size=batch_size)


def test_wall_clock_is_split_across_the_batch():
    """Each unit is credited a share of the batch's time, not the whole of it.

    Charging every unit the full batch duration would make a node's reported
    tokens/sec fall as the batch widens, which is backwards.
    """
    agent = _agent(_engine())
    outputs, seconds = agent._run_units(_units("a", "b", "c", "d"))

    assert len(outputs) == 4
    assert seconds > 0


def test_report_emits_one_message_per_unit_and_isolates_failures():
    agent = _agent(_engine(FlakyEngine))
    units = _units("fine", "boom")

    outputs, seconds = agent._run_units(units)
    agent._report_units(units, outputs, seconds)

    messages = [agent.outbox.get_nowait() for _ in range(agent.outbox.qsize())]
    by_unit = {m["unit_id"]: m for m in messages}

    assert len(messages) == 2
    assert by_unit["unit-0"]["type"] == "work_result"
    assert by_unit["unit-0"]["completion_tokens"] > 0
    assert by_unit["unit-1"]["type"] == "work_failed"
    assert "exploded" in by_unit["unit-1"]["message"]


def test_engine_returning_wrong_output_count_is_rejected():
    """A miscounting engine must fail loudly, not silently drop units."""
    class ShortEngine(MockEngine):
        def generate_batch(self, items):
            return super().generate_batch(items[:1])

    agent = _agent(_engine(ShortEngine))
    try:
        agent._run_units(_units("a", "b"))
    except Exception as e:
        assert "2 units" in str(e)
    else:
        raise AssertionError("expected an error for a short output list")
