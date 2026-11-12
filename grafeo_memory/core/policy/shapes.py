"""SHACL-style shapes: validate an outcome whatever rung produced it.

A shape returns a verdict and the configured severity decides what happens:
"violation" rejects the write, "warning" writes it flagged with lowered
confidence, "info" only logs.
"""

from __future__ import annotations

from dataclasses import dataclass

from grafeo_memory.core.ontology.types import is_violation


@dataclass
class ShapeVerdict:
    shape: str
    ok: bool
    severity: str = "info"
    message: str = ""

    @property
    def rejects(self) -> bool:
        return not self.ok and self.severity == "violation"

    @property
    def flags(self) -> bool:
        return not self.ok and self.severity == "warning"


def check_relation_shape(rel: str, source_type: str, target_type: str, severity: str) -> ShapeVerdict:
    """relation_domain_range: a controlled relation type must fit its domain and range."""
    bad = is_violation(rel, source_type, target_type)
    msg = f"{rel}: {source_type} -> {target_type} is outside the relation's domain or range" if bad else ""
    return ShapeVerdict("relation_domain_range", not bad, severity, msg)


def check_memory_shape(props: dict, severity: str) -> ShapeVerdict:
    """memory_provenance: a memory carries its provenance (a session, a run or a source)."""
    ok = bool(props.get("session_id") or props.get("run_id") or props.get("source"))
    return ShapeVerdict("memory_provenance", ok, severity, "" if ok else "memory has no session or run provenance")
