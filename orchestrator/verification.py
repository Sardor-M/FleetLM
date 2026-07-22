"""Checking work that ran on a machine nobody controls.

Every result is currently trusted completely. A node can return a truncated
answer, output from a smaller model than it claims to serve, or a result
replayed from an earlier unit, and nothing notices. That is the blocking
problem for running work on contributed hardware.

The layer built here is the cheapest one: **canaries**. A canary is a unit
whose expected answer was recorded from a run the operator trusts. It is
injected into a batch shaped exactly like real work, so a node cannot tell it
apart and cannot cheat selectively; its result never reaches the client.

What this deliberately does not do:

- It does not decide *why* a node failed a check. Wrong model, truncated
  output and outright fabrication all look the same from here, and guessing
  between them would be a claim this cannot support.
- It does not set its own threshold. How much two honest backends disagree on
  identical input is an open measurement (#8), so `agreement_threshold` is a
  stated assumption supplied by the caller, not a constant discovered here.
- It cannot catch a node that computes honestly on canaries and cheats
  elsewhere. Redundant execution and logprob attestation are the layers that
  address that; this one raises the cost of cheating, it does not close it.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher

logger = logging.getLogger("orchestrator.verification")

# Until #8 measures how far honest backends drift on identical input, this is
# an assumption rather than a finding. Deliberately loose: a false accusation
# costs a contributor's machine, a missed cheat costs one unit.
DEFAULT_AGREEMENT_THRESHOLD = 0.85


def agreement(a: str, b: str) -> float:
    """How alike two answers are, from 0 to 1.

    Whitespace-normalised because line wrapping is not evidence of cheating.
    Exact equality short-circuits: at temperature 0 an honest node running the
    same model should reproduce the reference exactly, and the interesting
    cases are the ones that do not.
    """
    left, right = " ".join(a.split()), " ".join(b.split())
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


@dataclass(frozen=True)
class Canary:
    """A prompt whose answer was recorded from a run the operator trusts.

    Reference answers are supplied, never invented here: an LLM's output is not
    knowable in advance, so a canary is only as good as the run it came from.
    """

    prompt: str
    expected: str
    model: str | None = None

    @property
    def id(self) -> str:
        return hashlib.sha256(
            f"{self.model}\x00{self.prompt}\x00{self.expected}".encode()
        ).hexdigest()[:12]

    def as_request(self) -> dict:
        """Shaped exactly like a real unit, so it is indistinguishable to a node."""
        return {
            "messages": [{"role": "user", "content": self.prompt}],
            "model": self.model,
            "max_tokens": 256,
            "temperature": 0.0,  # the reference is only reproducible greedily
        }


@dataclass
class NodeTrust:
    """What a node's record says about it. Evidence, not a reputation system.

    A node starts *unproven* rather than trusted: with no checks against it,
    `trust` is None and callers decide what to do with that. Treating an
    unexamined node as trustworthy is the assumption this whole module exists
    to remove.
    """

    node_id: str
    passed: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def checks(self) -> int:
        return self.passed + self.failed

    @property
    def trust(self) -> float | None:
        return None if not self.checks else self.passed / self.checks

    @property
    def is_suspect(self) -> bool:
        """One failed canary is enough to stop trusting a node's other work.

        Not a statistical judgement: a node that returns a wrong answer to a
        known question has no innocent reading, and the cost of being wrong
        here is one machine being asked to re-prove itself.
        """
        return self.failed > 0

    def snapshot(self) -> dict:
        return {
            "node_id": self.node_id[:8],
            "checks": self.checks,
            "passed": self.passed,
            "failed": self.failed,
            "trust": None if self.trust is None else round(self.trust, 3),
            "suspect": self.is_suspect,
            "recent_failures": self.failures[-5:],
        }


class Verifier:
    """Canary bookkeeping, replay detection, and what the fleet learned from it."""

    def __init__(
        self,
        canaries: list[Canary] | None = None,
        rate: float = 0.02,
        agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
    ):
        self.canaries = list(canaries or [])
        self.rate = max(0.0, min(1.0, rate))
        self.agreement_threshold = agreement_threshold
        self.trust: dict[str, NodeTrust] = {}
        # unit_id -> canary, for units this verifier planted
        self.planted: dict[str, Canary] = {}
        self.checks_run = 0
        self.checks_failed = 0
        self.replays_caught = 0
        # answer fingerprint -> (prompt fingerprint, unit it first came from)
        self._seen: dict[str, tuple[str, str]] = {}

    # ── planting ────────────────────────────────────────────────────────

    def canary_count(self, batch_size: int) -> int:
        """How many canaries to mix into a batch of `batch_size` real units.

        At least one whenever verification is on at all: a batch small enough
        to round down to zero is exactly the batch a cheat would hide in.
        """
        if not self.canaries or self.rate <= 0 or batch_size <= 0:
            return 0
        return max(1, round(batch_size * self.rate))

    def select(self, batch_size: int) -> list[Canary]:
        """Which canaries to mix into a batch, cycled so all of them get used."""
        count = self.canary_count(batch_size)
        return [self.canaries[i % len(self.canaries)] for i in range(count)]

    def register(self, unit_id: str, canary: Canary) -> None:
        self.planted[unit_id] = canary

    def is_canary(self, unit_id: str) -> bool:
        return unit_id in self.planted

    # ── checking ────────────────────────────────────────────────────────

    def check(self, unit_id: str, node_id: str, text: str) -> dict | None:
        """Judge one result. Returns a verdict for canaries, None for real work."""
        canary = self.planted.get(unit_id)
        if canary is None:
            return None

        score = agreement(canary.expected, text)
        passed = score >= self.agreement_threshold
        self.checks_run += 1
        record = self.trust.setdefault(node_id, NodeTrust(node_id=node_id))
        if passed:
            record.passed += 1
        else:
            record.failed += 1
            self.checks_failed += 1
            record.failures.append(f"canary {canary.id} scored {score:.2f}")
            logger.warning(
                f"Node {node_id[:8]} failed canary {canary.id}: "
                f"agreement {score:.2f} < {self.agreement_threshold}"
            )
        return {
            "unit_id": unit_id,
            "canary_id": canary.id,
            "node_id": node_id,
            "agreement": round(score, 4),
            "passed": passed,
        }

    def note_replay(self, unit_id: str, node_id: str, text: str, prompt: str) -> bool:
        """Flag an answer already returned for a *different question*.

        Keyed on the prompt, not the unit. Two units asking the same thing
        should get the same answer - that is what greedy decoding means, and
        the first version of this check called that determinism fraud, failing
        every honest node that answered a repeated canary. A replay is reusing
        an old answer for a *new* question.

        Cheap to check and cheap to evade, so this is a floor rather than a
        defence. Short answers are skipped: plenty of distinct prompts are
        honestly answered "Yes".
        """
        if len(text.strip()) < 40:
            return False
        answer_fp = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
        prompt_fp = hashlib.sha256(" ".join(prompt.split()).encode()).hexdigest()
        first_prompt, first_unit = self._seen.setdefault(
            answer_fp, (prompt_fp, unit_id)
        )
        if first_prompt == prompt_fp:
            return False  # same question, same answer - expected
        self.replays_caught += 1
        record = self.trust.setdefault(node_id, NodeTrust(node_id=node_id))
        record.failed += 1
        record.failures.append(f"replayed the answer already given for {first_unit}")
        logger.warning(
            f"Node {node_id[:8]} answered {unit_id} with text already returned "
            f"for the different prompt in {first_unit}"
        )
        return True

    # ── reporting ───────────────────────────────────────────────────────

    def suspects(self) -> list[str]:
        return [nid for nid, rec in self.trust.items() if rec.is_suspect]

    def overhead_pct(self, real_units: int) -> float:
        """Verification cost as a percentage of the work it rode along with."""
        if real_units <= 0:
            return 0.0
        return round(100 * self.checks_run / real_units, 2)

    def snapshot(self) -> dict:
        return {
            "enabled": bool(self.canaries) and self.rate > 0,
            "canaries_configured": len(self.canaries),
            "sample_rate": self.rate,
            "agreement_threshold": self.agreement_threshold,
            "threshold_is_measured": False,  # pending #8
            "checks_run": self.checks_run,
            "checks_failed": self.checks_failed,
            "replays_caught": self.replays_caught,
            "suspect_nodes": len(self.suspects()),
            "nodes": [rec.snapshot() for rec in self.trust.values()],
        }
