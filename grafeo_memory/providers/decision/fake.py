"""Deterministic in-repo decision model for CI and tests, no network, no key."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from grafeo_memory.providers.decision.types import CategoricalQuestion, Decision, Question


class FakeDecisionModel:
    """A deterministic DecisionModel.

    ``handler(state, questions) -> list[Decision]`` may override behavior;
    the default picks "ADD" when offered (else the first choice) with
    probability = confidence = 1.0, and answers score questions with the
    ceiling. Every call is recorded in ``calls`` so tests can assert how many
    decision calls a pipeline made.
    """

    def __init__(self, handler: Callable[[str, Sequence[Question]], list[Decision]] | None = None):
        self.handler = handler
        self.calls: list[tuple[str, list[Question]]] = []

    async def decide_async(self, state: str, questions: Sequence[Question]) -> list[Decision]:
        self.calls.append((state, list(questions)))
        if self.handler is not None:
            return self.handler(state, questions)
        out: list[Decision] = []
        for q in questions:
            if isinstance(q, CategoricalQuestion):
                value = "ADD" if "ADD" in q.choices else q.choices[0]
                out.append(
                    Decision(question_id=q.question_id, value=value, probability=1.0, confidence=1.0, runner_up=0.0)
                )
            else:  # ScoreQuestion
                out.append(
                    Decision(question_id=q.question_id, value=q.ceiling, probability=1.0, confidence=1.0, runner_up=0.0)
                )
        return out

    def decide(self, state: str, questions: Sequence[Question]) -> list[Decision]:
        from grafeo_memory.common.compat import run_sync

        return run_sync(self.decide_async(state, questions))
