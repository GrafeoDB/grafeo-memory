"""Decision tables: a rule is a predicate over a judgment's context plus an action.

Hit policy FIRST: rules are tried in priority order (lower first) and the first
whose predicate holds resolves the judgment; no hit means abstain (the next rung
decides). Rules never call a model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field


@dataclass
class FactContext:
    """What a rule sees about one fact and, for post rules, the decision about it."""

    text: str
    role: str | None = None  # role of the message the fact came from
    action: str | None = None  # the engine's decision: add / update / delete / none (post rules only)
    engine: str | None = None
    confidence: float | None = None
    target: dict | None = None  # properties of the targeted memory (status, role, ...)


@dataclass(frozen=True)
class Rule:
    id: str  # "<pack>.<name>", traced on every hit
    stage: str  # "pre" (before the engine, per fact) or "post" (on the engine's decision)
    priority: int
    predicate: Callable[[FactContext], bool]
    action: str  # pre: "reject"; post: "stand_down" (no change) | "park" | "add" (keep both)
    description: str = ""


@dataclass
class RuleHit:
    rule_id: str
    action: str
    detail: dict = field(default_factory=dict)


def evaluate(rules: Sequence[Rule], ctx: FactContext, stage: str) -> RuleHit | None:
    """FIRST hit by priority among the rules of ``stage``; None means abstain."""
    for rule in sorted((r for r in rules if r.stage == stage), key=lambda r: r.priority):
        try:
            if rule.predicate(ctx):
                return RuleHit(rule.id, rule.action)
        except Exception:  # a broken predicate abstains, it never blocks a write
            continue
    return None
