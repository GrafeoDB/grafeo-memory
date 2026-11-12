"""Telemetry: LLM usage accounting. The manager's ``usage_callback`` plugs in here."""

from __future__ import annotations

import sys
from collections.abc import Callable


def stderr_usage(operation: str, usage) -> None:
    """Print one usage line per LLM op to stderr (debugging/CI logs)."""
    print(f"[usage] {operation}: {usage}", file=sys.stderr)


class UsageTracker:
    """usage_callback sink: per-operation call counts and token totals."""

    def __init__(self):
        self.ops: dict[str, dict] = {}

    def __call__(self, op: str, usage) -> None:
        e = self.ops.setdefault(op, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        e["calls"] += 1
        e["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
        e["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)

    def stats(self) -> dict:
        total_in = sum(e["input_tokens"] for e in self.ops.values())
        total_out = sum(e["output_tokens"] for e in self.ops.values())
        return {"ops": self.ops, "total_input_tokens": total_in, "total_output_tokens": total_out}


UsageCallback = Callable[[str, object], None]
