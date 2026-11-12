"""Layer G: the open-decisions bucket, pure parts (questions, options, answer mapping).

A parked judgment keeps its evidence and a typed question whose options are fixed
strings, so the same question can go to a decision model, an LLM or the user.
Storage and resolution live in ``services/bucket.py``.
"""

from __future__ import annotations

from grafeo_memory.common.types import MemoryAction
from grafeo_memory.providers.decision.types import CategoricalQuestion

# Memory-contradiction options: what to do with the new (provisional) fact and the
# memory it contests.
REPLACE = "replace"  # the new fact supersedes the old memory (UPDATE)
INVALIDATE = "invalidate"  # the old memory is no longer true; keep the new one (DELETE)
KEEP_BOTH = "keep_both"  # both stand (ADD)
DUPLICATE = "duplicate"  # the new fact repeats the old one; drop the new (NONE)
MEMORY_OPTIONS = [REPLACE, INVALIDATE, KEEP_BOTH, DUPLICATE]

MERGE = "merge"
DISTINCT = "distinct"
ENTITY_MERGE_OPTIONS = [MERGE, DISTINCT]
NEW_ENTITY = "NEW"

_ACTION_TO_OPTION = {
    MemoryAction.UPDATE: REPLACE,
    MemoryAction.DELETE: INVALIDATE,
    MemoryAction.ADD: KEEP_BOTH,
    MemoryAction.NONE: DUPLICATE,
}


def option_for_action(action: MemoryAction) -> str:
    return _ACTION_TO_OPTION[action]


def contradiction_question(new_text: str, old_text: str | None) -> str:
    if old_text is None:
        return f'Is "{new_text}" new information, or does it repeat something already known?'
    return f'Does "{new_text}" replace "{old_text}", show that it is no longer true, hold alongside it, or repeat it?'


def entity_question(mention: str, candidates: list[str]) -> str:
    return f'Is "{mention}" the same as one of: {", ".join(candidates)}? Or a new one?'


def merge_question(a: str, b: str) -> str:
    return f'Are "{a}" and "{b}" the same thing?'


def rejudge_question(question_id: str, options: list[str], question: str) -> CategoricalQuestion:
    """The parked question as a decision-model question (options are the choices)."""
    return CategoricalQuestion(question_id=question_id, question=question, choices=list(options))
