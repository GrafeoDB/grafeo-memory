"""Zero-cost OpenTelemetry tracing helpers for grafeo-memory."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Any


@contextlib.contextmanager
def trace_operation(
    name: str,
    *,
    enabled: bool = False,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Context manager that creates an OTel span when enabled, or no-ops.

    Zero overhead when ``enabled=False`` — the opentelemetry import is deferred.
    """
    if not enabled:
        yield None
        return

    try:
        from opentelemetry.trace import get_tracer
    except ImportError:
        yield None
        return

    tracer = get_tracer("grafeo-memory")
    with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
        yield span
