"""Question packs: turn (new facts, candidate memories) into typed questions,
and map the decisions back to memory actions.

Both granularities ship; benchmarks decide the default. The state text is built from the
same material the LLM reconciliation prompt uses, so both engines always judge
identical evidence.
"""

from __future__ import annotations

from collections.abc import Sequence

from grafeo_memory.common.types import EscalationSignal, Fact, MemoryAction, ReconciliationDecision, Relation
from grafeo_memory.providers.decision.types import CategoricalQuestion, Decision, Question


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"


def build_state(new_facts: Sequence[Fact], candidates: Sequence[dict], max_chars: int = 100_000) -> str:
    """The state both engines see: numbered facts + candidate memories with IDs/scores.

    Bounded to ``max_chars`` (Jev rejects oversized states with
    ``max_tokens_exceeded``): highest-similarity candidates first, each memory
    text truncated to 500 chars, each fact to 1000 chars.
    """
    facts_text = "\n".join(f"{i + 1}. {_truncate(f.text, 1000)}" for i, f in enumerate(new_facts))
    ordered = sorted(candidates, key=lambda m: m.get("score", 0.0), reverse=True)
    memories_text = "\n".join(
        f'- ID {m["id"]}: "{_truncate(str(m["text"]), 500)}" (similarity: {m.get("score", 0.0):.2f})' for m in ordered
    )
    state = f"New facts:\n{facts_text}\n\nExisting memories:\n{memories_text}\n"
    return _truncate(state, max_chars)


def per_fact_pack(facts: Sequence[Fact], candidates: Sequence[dict]) -> list[Question]:
    """One categorical per fact; choices carry the target memory IDs."""
    questions: list[Question] = []
    for i, _fact in enumerate(facts):
        choices = ["ADD", "NONE"]
        criteria: dict[str, str | None] = {
            "ADD": "The fact is new — store it as a new memory.",
            "NONE": "The fact is already fully covered by an existing memory — do nothing.",
        }
        for m in candidates:
            choices.append(f"UPDATE_{m['id']}")
            criteria[f"UPDATE_{m['id']}"] = f"The fact changes or extends existing memory ID {m['id']}."
        for m in candidates:
            choices.append(f"DELETE_{m['id']}")
            criteria[f"DELETE_{m['id']}"] = f"The fact contradicts and invalidates existing memory ID {m['id']}."
        questions.append(
            CategoricalQuestion(
                question_id=f"fact_{i}",
                question=(
                    f"Given the new fact #{i + 1} and the existing memories in the state, "
                    "choose the single correct memory action."
                ),
                choices=choices,
                criteria=criteria,
            )
        )
    return questions


def pairwise_pack(facts: Sequence[Fact], candidates: Sequence[dict]) -> list[Question]:
    """One categorical per (fact, candidate); argmax over candidates afterwards."""
    questions: list[Question] = []
    for i, _fact in enumerate(facts):
        for m in candidates:
            questions.append(
                CategoricalQuestion(
                    question_id=f"fact_{i}|mem_{m['id']}",
                    question=(
                        f"Does new fact #{i + 1} duplicate, update, invalidate, or merely "
                        f"relate to existing memory ID {m['id']}?"
                    ),
                    choices=["DUPLICATE", "UPDATE", "INVALIDATE", "UNRELATED"],
                    criteria={
                        "DUPLICATE": "The fact restates this memory — no change needed.",
                        "UPDATE": "The fact changes or extends this memory.",
                        "INVALIDATE": "The fact contradicts this memory — it should be deleted.",
                        "UNRELATED": "The fact and this memory are unrelated.",
                    },
                )
            )
    return questions


PACKS = {"per_fact": per_fact_pack, "pairwise": pairwise_pack}


def _parse_target(choice: str) -> tuple[MemoryAction, str | None]:
    if choice.startswith("UPDATE_"):
        return MemoryAction.UPDATE, choice[len("UPDATE_") :]
    if choice.startswith("DELETE_"):
        return MemoryAction.DELETE, choice[len("DELETE_") :]
    if choice == "ADD":
        return MemoryAction.ADD, None
    return MemoryAction.NONE, None


def decisions_to_actions_per_fact(
    facts: Sequence[Fact],
    candidates: Sequence[dict],
    decisions: Sequence[Decision],
) -> list[tuple[ReconciliationDecision, Decision | None]]:
    """Map per-fact categorical decisions back to actions.

    Returns (action, winning Decision) pairs; the action already carries
    engine="decision" and the decision's confidence.
    """
    by_id = {d.question_id: d for d in decisions}
    out: list[tuple[ReconciliationDecision, Decision | None]] = []
    for i, fact in enumerate(facts):
        d = by_id.get(f"fact_{i}")
        if d is None or not isinstance(d.value, str):
            out.append((ReconciliationDecision(action=MemoryAction.ADD, text=fact.text, engine="decision"), None))
            continue
        action, target = _parse_target(d.value)
        out.append(
            (
                ReconciliationDecision(
                    action=action,
                    text=fact.text,
                    target_memory_id=target,
                    engine="decision",
                    confidence=d.confidence,
                ),
                d,
            )
        )
    return out


def decisions_to_actions_pairwise(
    facts: Sequence[Fact],
    candidates: Sequence[dict],
    decisions: Sequence[Decision],
) -> list[tuple[ReconciliationDecision, Decision | None]]:
    """Argmax over each fact's candidate row; all-UNRELATED resolves to ADD.

    Returns (action, deciding Decision) pairs; the action already carries
    engine="decision" and that decision's confidence. For an all-UNRELATED ADD the
    deciding decision is the least confident UNRELATED answer: the ADD is only as sure
    as its weakest pair, and None remains only when no answer came back at all.
    """
    by_id = {d.question_id: d for d in decisions}
    out: list[tuple[ReconciliationDecision, Decision | None]] = []
    for i, fact in enumerate(facts):
        best_action = MemoryAction.ADD
        best_target: str | None = None
        best_prob = -1.0
        best_decision: Decision | None = None
        weakest_unrelated: Decision | None = None
        for m in candidates:
            d = by_id.get(f"fact_{i}|mem_{m['id']}")
            if d is None or not isinstance(d.value, str):
                continue
            cls = d.value
            if cls == "UNRELATED" and (weakest_unrelated is None or d.confidence < weakest_unrelated.confidence):
                weakest_unrelated = d
            if cls in ("DUPLICATE", "UPDATE", "INVALIDATE") and d.probability > best_prob:
                best_prob = d.probability
                best_decision = d
                if cls == "UPDATE":
                    best_action, best_target = MemoryAction.UPDATE, str(m["id"])
                elif cls == "INVALIDATE":
                    best_action, best_target = MemoryAction.DELETE, str(m["id"])
                else:  # DUPLICATE
                    best_action, best_target = MemoryAction.NONE, None
        if best_decision is None:
            best_decision = weakest_unrelated
        out.append(
            (
                ReconciliationDecision(
                    action=best_action,
                    text=fact.text,
                    target_memory_id=best_target,
                    engine="decision",
                    confidence=best_decision.confidence if best_decision is not None else None,
                ),
                best_decision,
            )
        )
    return out


# --- relations: KEEP / DELETE per (new relation, existing relation) -------------


def build_relation_state(
    new_relations: Sequence[Relation], existing_relations: Sequence[dict], max_chars: int = 100_000
) -> str:
    """The state for relation reconciliation, mirroring the LLM prompt's material.
    Bounded to ``max_chars`` like :func:`build_state`."""
    new_text = "\n".join(f"{i + 1}. {r.source} -- {r.relation_type} -- {r.target}" for i, r in enumerate(new_relations))
    existing_text = "\n".join(
        f"- ID {i}: {r['source']} -- {r['relation_type']} -- {r['target']}" for i, r in enumerate(existing_relations)
    )
    return _truncate(f"New relations:\n{new_text}\n\nExisting relations:\n{existing_text}\n", max_chars)


def relations_pack(new_relations: Sequence[Relation], existing_relations: Sequence[dict]) -> list[Question]:
    """One KEEP/DELETE categorical per (new relation, existing relation) pair."""
    questions: list[Question] = []
    for i, _new in enumerate(new_relations):
        for j, _existing in enumerate(existing_relations):
            questions.append(
                CategoricalQuestion(
                    question_id=f"new_{i}|existing_{j}",
                    question=(
                        f"Does new relation #{i + 1} contradict or invalidate existing relation "
                        f"ID {j}? Choose DELETE only if the existing relation must be removed."
                    ),
                    choices=["KEEP", "DELETE"],
                    criteria={
                        "KEEP": "The existing relation still holds.",
                        "DELETE": "The new relation contradicts it — the existing relation should be deleted.",
                    },
                )
            )
    return questions


def relation_verdicts(
    new_relations: Sequence[Relation],
    existing_relations: Sequence[dict],
    decisions: Sequence[Decision],
) -> list[tuple[bool, Decision | None]]:
    """Per existing relation: (delete?, deciding decision). Each (new, existing) pair is
    decided on its own: an existing relation is deleted when any new relation's answer
    for it is DELETE (a confident KEEP from an unrelated new relation does not cancel a
    contradiction). The deciding decision is the most probable DELETE, else the most
    probable KEEP (None: no answer)."""
    by_id = {d.question_id: d for d in decisions}
    out: list[tuple[bool, Decision | None]] = []
    for j in range(len(existing_relations)):
        best_delete: Decision | None = None
        best_keep: Decision | None = None
        for i in range(len(new_relations)):
            d = by_id.get(f"new_{i}|existing_{j}")
            if d is None or not isinstance(d.value, str):
                continue
            if d.value == "DELETE":
                if best_delete is None or d.probability > best_delete.probability:
                    best_delete = d
            elif best_keep is None or d.probability > best_keep.probability:
                best_keep = d
        delete = best_delete is not None
        out.append((delete, best_delete if delete else best_keep))
    return out


def relation_key(r: dict) -> dict:
    return {"source": r["source"], "target": r["target"], "relation_type": r["relation_type"]}


def decisions_to_deletes(
    new_relations: Sequence[Relation],
    existing_relations: Sequence[dict],
    decisions: Sequence[Decision],
) -> list[dict]:
    """The existing relations to delete (see :func:`relation_verdicts`)."""
    verdicts = relation_verdicts(new_relations, existing_relations, decisions)
    return [relation_key(r) for r, (delete, _d) in zip(existing_relations, verdicts, strict=True) if delete]


# --- HYBRID escalation -----------------------------------------------------------


def should_escalate(decision: Decision, threshold: float, signal: EscalationSignal) -> bool:
    """True when a decision is too uncertain and HYBRID should hand it to the LLM.

    PROB_GAP uses p(chosen) - p(runner-up); when the provider returned no full
    distribution (runner_up is None) it falls back to a two-class estimate.
    """
    if signal == EscalationSignal.CONFIDENCE:
        return decision.confidence < threshold
    runner_up = decision.runner_up if decision.runner_up is not None else 1.0 - decision.probability
    gap = decision.probability - runner_up
    if signal == EscalationSignal.PROB_GAP:
        return gap < threshold
    return decision.confidence < threshold or gap < threshold  # COMBINED


# The mapper each built-in pack's answers need (read by reconciliation._resolve_pack).
# A custom pack declares its own with a ``granularity`` attribute.
PACK_GRANULARITY = {per_fact_pack: "per_fact", pairwise_pack: "pairwise"}
