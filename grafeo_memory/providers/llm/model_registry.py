"""Model-string registry: "provider:name" specs -> pydantic-ai models.

pydantic-ai accepts plain model-name strings directly, so resolution is mostly
pass-through; the registry exists to (a) name the provider explicitly and
(b) build provider-specific model objects when an extra is installed.
"""

from __future__ import annotations

from pydantic_ai.models import Model

ModelType = str | Model


def provider_of(spec: str) -> str:
    """The provider prefix of a model spec: "mistral:x" -> "mistral", "gpt-4o" -> "openai"."""
    if ":" in spec:
        return spec.split(":", 1)[0]
    return "openai"


def resolve_model(spec: str | Model) -> ModelType:
    """Resolve a model spec to something pydantic-ai Agents accept.

    Plain strings pass through (pydantic-ai resolves them per its own registry);
    already-built Model objects pass through untouched.
    """
    if isinstance(spec, str):
        return spec
    return spec
