"""Jev (TypeSafe System One) decision-model adapter, optional extra ``grafeo-memory[jev]``.

Hosted API: ``POST https://api.typesafe.ai/v1/systemone`` with
``Authorization: Bearer $TYPESAFE_API_KEY``. One request carries the ``state``
plus a map of typed questions (choice / score / noul); the response carries one
calibrated answer per question. Calls flow through the shared
:class:`~grafeo_memory.common.cache.MemoryCache` (kind ``"decision"``) when one is
provided, so benchmark reruns never re-pay for identical decisions.

Data boundary: enabling ``decision_model="jev"`` means reconciliation state
(facts + candidate memories) leaves the device to TypeSafe. The default
pipeline stays fully local (LLM via your own provider, or no calls at all on
the fast path).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Sequence

from grafeo_memory.common.cache import MemoryCache
from grafeo_memory.providers.decision.types import CategoricalQuestion, Decision, Question, ScoreQuestion

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
# Account or key errors: every further call would fail the same way until someone acts.
_CIRCUIT_STATUSES = {401, 402, 403}


class JevDecisionModel:
    """DecisionModel backed by the TypeSafe Jev API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        cache: MemoryCache | None = None,
        timeout: float = 60.0,
        max_retries: int = 5,
        circuit_cooldown: float = 300.0,
    ):
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self._api_key:
            raise ValueError("JevDecisionModel needs an API key: pass api_key= or set TYPESAFE_API_KEY.")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.timeout = timeout
        self.max_retries = max_retries
        # Token usage of the most recent non-cached call (for usage_callback accounting).
        self.last_usage: dict | None = None
        # Circuit breaker: after a 401/402/403 the API is not called again for
        # ``circuit_cooldown`` seconds; calls fail at once and callers fall back to the LLM.
        self.circuit_cooldown = circuit_cooldown
        self._circuit_open_until = 0.0
        self._circuit_status: int | None = None

    # -- request building ----------------------------------------------------

    def _to_jev_question(self, q: Question) -> dict:
        if isinstance(q, CategoricalQuestion):
            criteria = {c: (q.criteria.get(c) if q.criteria else None) for c in q.choices}
            return {"type": "choice", "instructions": q.question, "criteria": criteria}
        if isinstance(q, ScoreQuestion):
            criteria = q.criteria or [str(q.floor), str(q.ceiling)]
            return {"type": "score", "instructions": q.question, "criteria": criteria}
        raise TypeError(f"Unsupported question type: {type(q).__name__}")

    def _build_payload(self, state: str, questions: Sequence[Question]) -> dict:
        return {
            "model": self.model,
            "state": state,
            "questions": {q.question_id: self._to_jev_question(q) for q in questions},
        }

    # -- response parsing ------------------------------------------------------

    @staticmethod
    def _parse_answer(q: Question, ans: dict) -> Decision:
        raw_conf = ans.get("confidence")
        confidence = 1.0 if raw_conf is None else float(raw_conf)  # 0.0 is a real answer, keep it
        if "choice" in ans:
            value = str(ans["choice"])
            probs = ans.get("probabilities")
            runner_up: float | None = None
            if isinstance(probs, dict):
                probability = float(probs.get(value, 1.0))
                rest = [float(p) for c, p in probs.items() if str(c) != value]
                runner_up = max(rest) if rest else 0.0
            else:
                raw_p = ans.get("probability")
                probability = 1.0 if raw_p is None else float(raw_p)
            return Decision(
                question_id=q.question_id,
                value=value,
                probability=probability,
                confidence=confidence,
                runner_up=runner_up,
            )
        if "noul" in ans:
            p = float(ans["noul"])
            return Decision(question_id=q.question_id, value=p, probability=p, confidence=confidence)
        if "score" in ans:
            # Jev score levels are positional: criteria[0] -> 0, criteria[-1] -> len-1.
            # Normalize onto the question's [floor, ceiling] range.
            n_levels = len(q.criteria) if isinstance(q, ScoreQuestion) and q.criteria else 2
            raw = float(ans["score"])
            span = max(n_levels - 1, 1)
            value = q.floor + (raw / span) * (q.ceiling - q.floor) if isinstance(q, ScoreQuestion) else raw
            return Decision(question_id=q.question_id, value=value, probability=1.0, confidence=confidence)
        raise ValueError(f"Unrecognized Jev answer for {q.question_id}: {sorted(ans)}")

    # -- the call --------------------------------------------------------------

    async def decide_async(self, state: str, questions: Sequence[Question]) -> list[Decision]:
        if not questions:
            return []
        payload = self._build_payload(state, questions)
        cache_key_model = f"jev:{self.model}"
        if self.cache is not None:
            hit = self.cache.get("decision", cache_key_model, payload)
            if hit is not None:
                return [self._parse_answer(q, hit["answers"][q.question_id]) for q in questions]

        if time.monotonic() < self._circuit_open_until:
            raise RuntimeError(f"Jev unavailable (HTTP {self._circuit_status}); not retrying until the cooldown ends")

        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(self.base_url, json=payload, headers=headers)
                if resp.status_code == 429 and attempt < self.max_retries - 1:
                    await asyncio.sleep(min(2**attempt, 30))
                    continue
                if resp.status_code == 400 and "max_tokens_exceeded" in resp.text:
                    if len(questions) > 1:
                        # Request too large: split the question batch and recurse.
                        mid = len(questions) // 2
                        logger.warning(
                            "Jev max_tokens_exceeded with %d questions: splitting into %d + %d",
                            len(questions),
                            mid,
                            len(questions) - mid,
                        )
                        first = await self.decide_async(state, questions[:mid])
                        second = await self.decide_async(state, questions[mid:])
                        return list(first) + list(second)
                    # Single question still too big: shrink the state and retry.
                    state = state[: max(len(state) // 2, 1000)] + "\n... [truncated to fit context]"
                    payload = self._build_payload(state, questions)
                    logger.warning("Jev max_tokens_exceeded: halved state to %d chars and retrying", len(state))
                    continue
                if resp.status_code >= 400:
                    # Log the status and the API's error code only: the body can echo the
                    # request, which holds memory text.
                    try:
                        err = resp.json().get("error")
                        code = err.get("code") or err.get("type") if isinstance(err, dict) else err
                    except Exception:
                        code = None
                    logger.warning(
                        "Jev HTTP %s (%s) for model=%s | questions=%d state_len=%d",
                        resp.status_code,
                        code,
                        self.model,
                        len(questions),
                        len(state),
                    )
                    if resp.status_code in _CIRCUIT_STATUSES:
                        self._circuit_status = resp.status_code
                        self._circuit_open_until = time.monotonic() + self.circuit_cooldown
                resp.raise_for_status()
                body = resp.json()
                usage = body.get("usage") or {}
                # Current API: {"input_tokens": n, "output_tokens": n}; tolerate older names.
                self.last_usage = {
                    "prompt_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                    "completion_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
                }
                answers = body.get("answers", {})
                decisions = [self._parse_answer(q, answers[q.question_id]) for q in questions]
                if self.cache is not None:
                    self.cache.put("decision", cache_key_model, payload, {"answers": answers})
                return decisions
            except Exception as e:
                last_exc = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                transient = status == 429 or "Timeout" in type(e).__name__ or "Connect" in type(e).__name__
                if transient and attempt < self.max_retries - 1:
                    await asyncio.sleep(min(2**attempt, 30))
                    continue
                raise
        raise RuntimeError(f"Jev request failed after {self.max_retries} attempts: {last_exc}")

    def decide(self, state: str, questions: Sequence[Question]) -> list[Decision]:
        from grafeo_memory.common.compat import run_sync

        return run_sync(self.decide_async(state, questions))
