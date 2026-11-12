"""Provider retry policy: transient failures (timeouts, connection errors, 429, 5xx) are
retried with exponential backoff instead of failing the step that made the call.

The backoff lives in ``common.timing`` (base layer). :class:`RetryingModel` applies it to
every request of a pydantic-ai model, so a dropped connection during entity extraction
retries the call instead of storing a fact without its entities.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.wrapper import WrapperModel

from grafeo_memory.common.timing import TRANSIENT_MARKERS, is_transient, retry_async, retry_sync

__all__ = ["TRANSIENT_MARKERS", "RetryingModel", "is_transient", "retry_async", "retry_sync", "with_retries"]


class RetryingModel(WrapperModel):
    """Retry a model's requests on transient errors; any other error is raised at once."""

    def __init__(self, wrapped: Model, retries: int = 3):
        super().__init__(wrapped)
        self.retries = max(1, retries)

    async def request(self, messages, model_settings, model_request_parameters):
        return await retry_async(
            lambda: self.wrapped.request(messages, model_settings, model_request_parameters),
            retries=self.retries,
        )


def with_retries(model: Any, retries: int) -> Any:
    """``model`` wrapped in :class:`RetryingModel` when ``retries`` > 0. A model string
    is resolved first; a value pydantic-ai cannot resolve is returned unchanged."""
    if retries <= 0 or isinstance(model, RetryingModel):
        return model
    if isinstance(model, str):
        try:
            model = infer_model(model)
        except Exception:
            return model
    if not isinstance(model, Model):
        return model
    return RetryingModel(model, retries=retries + 1)  # retries after the first attempt
