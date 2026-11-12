"""Shipped rule packs: compliance, temporal, entrenchment, lifecycle.

- compliance: credentials and payment or identity numbers are never stored.
- temporal: disjoint validity intervals never conflict (the pre-model temporal rule;
  enabling the pack turns on the reconciliation pre-check).
- entrenchment: certified content outranks inferred, user-stated outranks
  assistant-stated. A model decision that would overwrite either stands down, or
  parks when the bucket is on and conservative mode is set.
- lifecycle: a re-stated fact counts as an observation of the memory it repeats;
  N observations promote that memory to certified (applied by the manager).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from grafeo_memory.core.policy.rules import FactContext, Rule

SECRET_PATTERNS: tuple[str, ...] = (
    r"\b(?:sk|pk|rk)[-_][A-Za-z0-9_\-]{16,}",  # API keys (OpenAI / Stripe style)
    r"\bAKIA[0-9A-Z]{16}\b",  # AWS access key id
    r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",  # GitHub tokens
    r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b",  # Slack tokens
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"(?i)\b(?:password|passwd|pwd|passcode|pin code|api key|secret key)\s*(?:is|:|=)\s*\S{4,}",
    r"\b(?:\d[ -]?){12,18}\d\b",  # card-like digit runs, kept only when they pass Luhn
    r"\b\d{3}-\d{2}-\d{4}\b",  # US SSN
    r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}\b",  # IBAN
)

_CARD = re.compile(r"(?:\d[ -]?){12,18}\d")


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def find_secret(text: str, extra: Sequence[str] = ()) -> str | None:
    """The first secret-looking span in ``text`` (None when clean)."""
    for pat in (*SECRET_PATTERNS, *extra):
        for m in re.finditer(pat, text):
            span = m.group(0)
            if _CARD.fullmatch(span) and not _luhn(re.sub(r"\D", "", span)):
                continue
            return span
    return None


def _overwrites(ctx: FactContext) -> bool:
    return ctx.action in ("update", "delete") and ctx.target is not None


def build_pre_rules(packs: Sequence[str], extra_secret_patterns: Sequence[str] = ()) -> list[Rule]:
    rules: list[Rule] = []
    if "compliance" in packs:
        rules.append(
            Rule(
                "compliance.secret",
                "pre",
                0,
                lambda c: find_secret(c.text, extra_secret_patterns) is not None,
                "reject",
                "It is forbidden that a credential or a payment or identity number is stored.",
            )
        )
    return rules


def build_post_rules(packs: Sequence[str], conservative: bool = True) -> list[Rule]:
    rules: list[Rule] = []
    if "entrenchment" in packs:
        rules.append(
            Rule(
                "entrenchment.certified",
                "post",
                10,
                lambda c: _overwrites(c) and (c.target or {}).get("status") == "certified" and c.engine != "user",
                "park" if conservative else "stand_down",
                "Certified content is not overwritten by an inferred decision.",
            )
        )
        rules.append(
            Rule(
                "entrenchment.user_stated",
                "post",
                20,
                lambda c: _overwrites(c) and (c.target or {}).get("role") == "user" and c.role == "assistant",
                "add",
                "A user-stated memory is not overwritten by an assistant-stated fact; both stand.",
            )
        )
    return rules


def observed_candidate(fact_text: str, candidates: Sequence[dict], min_overlap: float = 0.6) -> str | None:
    """Lifecycle: the candidate a NONE fact repeats (token Jaccard >= min_overlap), if any."""
    words = set(re.findall(r"[a-z0-9]+", fact_text.lower()))
    if not words:
        return None
    best, best_score = None, 0.0
    for m in candidates:
        other = set(re.findall(r"[a-z0-9]+", str(m.get("text", "")).lower()))
        if not other:
            continue
        j = len(words & other) / len(words | other)
        if j > best_score:
            best, best_score = str(m["id"]), j
    return best if best_score >= min_overlap else None
