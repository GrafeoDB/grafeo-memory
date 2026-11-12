"""Registry for decision models.

Built-ins register here; third-party providers plug in via the
``grafeo_memory.decision_models`` entry-point group (same pattern as the
embedder extras), e.g. in their pyproject::

    [project.entry-points."grafeo_memory.decision_models"]
    mymodel = "mypkg.decisions:MyDecisionModel"
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Callable[..., Any]] = {}
_ENTRYPOINT_GROUP = "grafeo_memory.decision_models"


def register_decision_model(name: str, factory: Callable[..., Any]) -> None:
    """Register a factory (class or callable) under ``name``."""
    _REGISTRY[name] = factory


def _load_entrypoints() -> None:
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group=_ENTRYPOINT_GROUP)
    except Exception:
        logger.warning("listing decision-model entry points failed", exc_info=True)
        return
    for ep in eps:
        if ep.name in _REGISTRY:
            continue
        try:
            # ep.load() returns the provider class/factory; register that, not the loader.
            _REGISTRY[ep.name] = ep.load()
        except Exception:  # one broken plugin must not hide the others
            logger.warning("failed to load decision-model entry point %r", ep.name, exc_info=True)


def _instantiate(factory: Callable[..., Any], kwargs: dict) -> Any:
    """Call ``factory(**kwargs)``, dropping kwargs the factory doesn't accept
    (so callers can pass shared context like cache=/api_key= to any provider)."""
    import inspect

    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):  # no introspectable signature (some builtins/C callables)
        return factory(**kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return factory(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in params}
    return factory(**accepted)


def get_decision_model(spec: str | Any, **kwargs) -> Any:
    """Resolve ``spec`` to a DecisionModel.

    ``spec`` may be a DecisionModel instance (returned as-is), a DecisionModel
    class (instantiated with ``kwargs``), a registered name (``"jev"``,
    ``"fake"``), or an entry-point name.
    """
    if isinstance(spec, type) and hasattr(spec, "decide_async"):
        return _instantiate(spec, kwargs)
    if hasattr(spec, "decide_async"):
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"decision_model must be a str or DecisionModel, got {type(spec).__name__}")
    if spec not in _REGISTRY:
        _load_entrypoints()
    if spec in _REGISTRY:
        return _instantiate(_REGISTRY[spec], kwargs)
    raise ValueError(
        f"Unknown decision model {spec!r}. Registered: {sorted(_REGISTRY)}. "
        f"Install a provider extra (e.g. grafeo-memory[jev]) or register via "
        f"register_decision_model()."
    )


# Built-ins. "fake" is always available (tests and CI). "jev" imports httpx only when it
# calls the API (pydantic-ai already depends on httpx); it is also the entry point in
# pyproject.toml.
from grafeo_memory.providers.decision.fake import FakeDecisionModel  # noqa: E402
from grafeo_memory.providers.decision.jev import JevDecisionModel  # noqa: E402

register_decision_model("fake", FakeDecisionModel)
register_decision_model("jev", JevDecisionModel)
