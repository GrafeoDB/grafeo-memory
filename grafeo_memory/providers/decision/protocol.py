"""The DecisionModel protocol, one interface, many providers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from grafeo_memory.providers.decision.types import Decision, Question


@runtime_checkable
class DecisionModel(Protocol):
    """A decision model: state + typed questions in, calibrated decisions out.

    Implementations are discriminative, no prose generation. Providers: Jev
    (hosted, ``grafeo-memory[jev]``), a local classifier, an ensemble, anything
    implementing this protocol, registered in :mod:`.registry`.
    """

    async def decide_async(self, state: str, questions: Sequence[Question]) -> list[Decision]:
        """Answer every question against ``state`` in one pass."""
        ...


async def decide_async(model: DecisionModel, state: str, questions: Sequence[Question]) -> list[Decision]:
    """Convenience wrapper: ``await decide_async(model, state, questions)``."""
    return await model.decide_async(state, questions)
