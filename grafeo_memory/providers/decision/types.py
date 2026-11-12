"""Typed questions and decisions for decision models (System One)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CategoricalQuestion:
    """Pick one choice from a bounded set (Jev ``choice`` primitive)."""

    question_id: str
    question: str  # instructions sent to the model
    choices: list[str]
    criteria: dict[str, str | None] = field(default_factory=dict)  # choice -> description


@dataclass(frozen=True)
class ScoreQuestion:
    """A graded position on a numeric scale (Jev ``score`` primitive)."""

    question_id: str
    question: str
    floor: float = 0.0
    ceiling: float = 1.0
    criteria: list[str] | None = None  # level labels; defaults to [floor, ceiling]


Question = CategoricalQuestion | ScoreQuestion


@dataclass(frozen=True)
class Decision:
    """One answer: the value plus the model's probability and calibrated confidence."""

    question_id: str
    value: str | float
    probability: float  # P(value) within the returned distribution
    confidence: float  # provider's calibrated confidence, when available
    # P(runner-up) for categorical questions; used by the PROB_GAP escalation
    # signal (p1 - p2). None when the provider returned no full distribution.
    runner_up: float | None = None
