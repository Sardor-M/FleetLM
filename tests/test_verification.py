"""Adversarial tests: nodes that lie, and what the fleet notices.

Acceptance criteria for verifying untrusted work are a detection rate against
deliberate cheats and a false-positive rate against honest nodes. Both are
asserted here rather than described, because the failure this guards against
is a verifier that looks like it works and accuses nobody.

The cheats modelled are the ones the design actually admits: a node returning
output from a smaller or different model, a node truncating to save decode
steps, and a node replaying an answer it already produced. A node that
computes honestly on canaries and cheats elsewhere is *not* caught here, and
no test pretends otherwise - that needs redundant execution.
"""

import pytest

from orchestrator.batch import BatchStore, UnitState
from orchestrator.verification import (
    DEFAULT_AGREEMENT_THRESHOLD,
    Canary,
    NodeTrust,
    Verifier,
    agreement,
)

REFERENCE = (
    "The capital of Japan is Tokyo, which has been the seat of government "
    "since 1868 when the emperor moved there from Kyoto."
)

CANARIES = [
    Canary(prompt="What is the capital of Japan?", expected=REFERENCE),
    Canary(
        prompt="Define latency.",
        expected=(
            "Latency is the delay between issuing a request and receiving the "
            "first byte of the response, measured end to end."
        ),
    ),
]


def _verifier(rate=1.0, **kw):
    return Verifier(canaries=CANARIES, rate=rate, **kw)


# ── how answers are compared ────────────────────────────────────────────

def test_an_identical_answer_agrees_completely():
    assert agreement(REFERENCE, REFERENCE) == 1.0


def test_reflowed_whitespace_is_not_evidence_of_cheating():
    """Line wrapping differs between backends and means nothing."""
    assert agreement(REFERENCE, REFERENCE.replace(" ", "\n  ")) == 1.0


def test_an_unrelated_answer_scores_far_below_the_threshold():
    score = agreement(REFERENCE, "I am unable to help with that request.")

    assert score < DEFAULT_AGREEMENT_THRESHOLD


def test_an_empty_answer_agrees_with_nothing():
    assert agreement(REFERENCE, "") == 0.0
    assert agreement("", REFERENCE) == 0.0


# ── canaries are not identifiable ───────────────────────────────────────

async def test_canaries_are_spread_through_the_batch_not_left_at_the_end():
    """Tacked on the end, canaries all land on whoever drains the tail - and a
    node that learns that can be honest for the last few units and cheat on
    the rest."""
    store = BatchStore(verifier=_verifier(rate=0.25))
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(20)], "m"
    )

    order = [u.id for u in await store.lease("n1", 100)]
    positions = [order.index(uid) for uid in batch.canary_unit_ids]

    assert positions and max(positions) < len(order) - 1


async def test_a_canary_is_shaped_like_the_work_around_it():
    """Anything distinguishing - a marker, a different max_tokens - would let a
    node answer canaries honestly and cheat on everything else."""
    store = BatchStore(verifier=_verifier())
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": "q"}], "max_tokens": 256}], "m"
    )

    canary = store.units[batch.canary_unit_ids[0]]
    real = store.units[batch.unit_ids[0]]

    assert canary.batch_id == real.batch_id
    assert canary.max_tokens == real.max_tokens
    assert set(canary.payload()) == set(real.payload())


async def test_a_canary_inherits_the_batch_model_so_it_is_not_routed_differently():
    """A canary that named no model could be leased by a node serving another
    one, which would look like a cheat and is just a routing accident."""
    store = BatchStore(verifier=_verifier())
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": "q"}]}], "llama3.2"
    )

    assert store.units[batch.canary_unit_ids[0]].model == "llama3.2"


# ── canaries never reach the client ─────────────────────────────────────

async def test_the_client_never_sees_a_canary_in_its_results():
    store = BatchStore(verifier=_verifier())
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(4)], "m"
    )

    for unit in await store.lease("n1", 100):
        await store.complete(unit.id, "n1", REFERENCE)

    records = store.results(batch.id)
    assert len(records) == 4
    assert [r["index"] for r in records] == [0, 1, 2, 3]


async def test_canaries_are_absent_from_the_counts_the_client_polls():
    store = BatchStore(verifier=_verifier(rate=1.0))
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(4)], "m"
    )

    assert store.counts(batch.id)["total"] == 4


async def test_a_batch_finishes_without_waiting_on_its_canaries():
    """Verification is the operator's business, not a delay the submitter pays."""
    store = BatchStore(verifier=_verifier())
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": "q"}]}], "m"
    )

    real = store.units[batch.unit_ids[0]]
    await store.lease("n1", 100)
    await store.complete(real.id, "n1", "an answer")

    assert store.get_batch(batch.id).state.value == "completed"


async def test_cancelling_a_batch_also_stops_its_canaries():
    store = BatchStore(verifier=_verifier())
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": "q"}]}], "m"
    )

    await store.cancel_batch(batch.id)

    assert store.units[batch.canary_unit_ids[0]].state == UnitState.FAILED


# ── the cheats ──────────────────────────────────────────────────────────

class CheatingNode:
    """A node that lies in one specific way, so detection can be attributed."""

    def __init__(self, style):
        self.style = style

    def answer(self, expected: str) -> str:
        if self.style == "honest":
            return expected
        if self.style == "truncated":          # stops early to save decode steps
            return expected[: len(expected) // 4]
        if self.style == "wrong_model":        # a smaller model, plausible but different
            return "Tokyo is a city in Japan."
        if self.style == "empty":              # returns nothing at all
            return ""
        if self.style == "refusal":            # cheapest possible output
            return "I cannot answer that."
        raise AssertionError(f"unknown style {self.style}")


async def _run_batch(store, node_id, style, n=8):
    """One node takes a whole batch and answers in its own style."""
    await store.create_batch(
        [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(n)], "m"
    )
    node = CheatingNode(style)
    for unit in await store.lease(node_id, 1000):
        canary = store.verifier.planted.get(unit.id)
        text = node.answer(canary.expected) if canary else "a normal answer"
        await store.complete(unit.id, node_id, text)


@pytest.mark.parametrize("style", ["truncated", "wrong_model", "empty", "refusal"])
async def test_every_modelled_cheat_is_detected(style):
    """Detection rate: 100% of these cheats, on every canary they touch."""
    verifier = _verifier()
    store = BatchStore(verifier=verifier)

    await _run_batch(store, "cheat", style)

    assert verifier.trust["cheat"].is_suspect
    assert verifier.trust["cheat"].failed > 0
    assert verifier.trust["cheat"].trust == 0.0


async def test_an_honest_node_is_never_accused():
    """False-positive rate: 0 across every canary, repeated."""
    verifier = _verifier()
    store = BatchStore(verifier=verifier)

    for _ in range(5):
        await _run_batch(store, "honest", "honest")

    assert verifier.checks_run >= 5
    assert verifier.checks_failed == 0
    assert verifier.suspects() == []
    assert verifier.trust["honest"].trust == 1.0


async def test_a_cheat_is_caught_without_the_client_seeing_an_error():
    """The submitter's batch must succeed even though the fleet caught a liar -
    detection is the operator's signal, not the client's problem."""
    verifier = _verifier()
    store = BatchStore(verifier=verifier)
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(4)], "m"
    )

    for unit in await store.lease("cheat", 100):
        await store.complete(unit.id, "cheat", "I cannot answer that.")

    assert verifier.suspects() == ["cheat"]
    assert all(r["status"] == "complete" for r in store.results(batch.id))


async def test_one_node_cheating_does_not_implicate_another():
    verifier = _verifier()
    store = BatchStore(verifier=verifier)

    await _run_batch(store, "good", "honest")
    await _run_batch(store, "bad", "wrong_model")

    assert verifier.suspects() == ["bad"]
    assert verifier.trust["good"].trust == 1.0


# ── replay ──────────────────────────────────────────────────────────────

async def test_replaying_an_earlier_answer_for_a_different_unit_is_caught():
    verifier = _verifier(rate=0.0)  # canaries off, so this is replay alone
    store = BatchStore(verifier=verifier)
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(3)], "m"
    )

    for uid in batch.unit_ids:
        await store.complete(uid, "lazy", REFERENCE)

    assert verifier.replays_caught == 2  # the second and third reuse the first
    assert verifier.trust["lazy"].is_suspect


async def test_a_retried_unit_is_not_mistaken_for_a_replay():
    """The same unit answered twice is the retry path working, not a cheat."""
    verifier = _verifier(rate=0.0)
    store = BatchStore(verifier=verifier)
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": "q"}]}], "m"
    )
    uid = batch.unit_ids[0]

    await store.complete(uid, "n1", REFERENCE)
    await store.complete(uid, "n1", REFERENCE)  # duplicate, ignored upstream

    assert verifier.replays_caught == 0
    assert verifier.suspects() == []


def test_two_units_sharing_a_short_answer_is_not_fraud():
    """Plenty of honest prompts answer 'Yes'. Calling that a replay would
    punish honest nodes for the shape of the question."""
    verifier = _verifier()

    assert verifier.note_replay("u1", "n1", "Yes.", "Is water wet?") is False
    assert verifier.note_replay("u2", "n1", "Yes.", "Is fire hot?") is False
    assert verifier.suspects() == []


def test_the_same_question_answered_the_same_way_twice_is_determinism():
    """This is what greedy decoding *means*. The first version of the replay
    check flagged it, and failed every honest node answering a repeated
    canary - caught by the honest-node false-positive test, not by review."""
    verifier = _verifier()
    prompt = "What is the capital of Japan?"

    assert verifier.note_replay("u1", "n1", REFERENCE, prompt) is False
    assert verifier.note_replay("u2", "n1", REFERENCE, prompt) is False
    assert verifier.suspects() == []


def test_the_same_answer_to_a_different_question_is_still_caught():
    """Fixing the false positive must not blunt the detection it exists for."""
    verifier = _verifier()

    assert verifier.note_replay("u1", "n1", REFERENCE, "What is the capital?") is False
    assert verifier.note_replay("u2", "n1", REFERENCE, "Define latency.") is True


# ── what the operator is told ───────────────────────────────────────────

def test_an_unchecked_node_is_unproven_rather_than_trusted():
    """The assumption this module exists to remove is that silence means honest."""
    assert NodeTrust(node_id="n1").trust is None
    assert NodeTrust(node_id="n1").is_suspect is False


def test_overhead_is_reported_as_a_share_of_real_work():
    verifier = _verifier()
    verifier.checks_run = 5

    assert verifier.overhead_pct(100) == 5.0
    assert verifier.overhead_pct(0) == 0.0


def test_a_batch_too_small_to_round_up_still_gets_a_canary():
    """Rounding to zero would leave exactly the small batch a cheat hides in
    unchecked."""
    assert _verifier(rate=0.02).canary_count(1) == 1


def test_verification_is_off_when_no_canary_has_been_recorded():
    """Canaries need reference answers from a trusted run. Inventing them here
    would mean asserting what a model 'should' say."""
    assert Verifier(canaries=[], rate=0.5).canary_count(100) == 0


def test_the_snapshot_admits_the_threshold_is_not_measured():
    """It is an assumption pending the honest-divergence baseline, and a report
    that hid that would read as a finding."""
    snapshot = _verifier().snapshot()

    assert snapshot["threshold_is_measured"] is False
    assert snapshot["agreement_threshold"] == DEFAULT_AGREEMENT_THRESHOLD


async def test_a_store_without_a_verifier_behaves_exactly_as_before():
    """Verification is opt-in; the default path must be untouched."""
    store = BatchStore()
    batch = await store.create_batch(
        [{"messages": [{"role": "user", "content": "q"}]}], "m"
    )

    assert batch.canary_unit_ids == []
    assert len(store.results(batch.id)) == 1
