"""Layer F: the policy cascade. Rules resolve or abstain before any model is asked.

Rules are data (decision tables with a FIRST hit policy by priority), every hit is
traced with its rule id, and shapes guard every outcome whatever rung produced it.
"""

from __future__ import annotations

from grafeo_memory.core.policy.packs import (
    SECRET_PATTERNS,
    build_post_rules,
    build_pre_rules,
    find_secret,
    observed_candidate,
)
from grafeo_memory.core.policy.rules import FactContext, Rule, RuleHit, evaluate
from grafeo_memory.core.policy.shapes import ShapeVerdict, check_memory_shape, check_relation_shape

__all__ = [
    "SECRET_PATTERNS",
    "FactContext",
    "Rule",
    "RuleHit",
    "ShapeVerdict",
    "build_post_rules",
    "build_pre_rules",
    "check_memory_shape",
    "check_relation_shape",
    "evaluate",
    "find_secret",
    "observed_candidate",
]
