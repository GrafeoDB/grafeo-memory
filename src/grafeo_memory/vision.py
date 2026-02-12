"""Vision support: describe images via LLM for memory extraction."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_ai import Agent, ImageUrl

from ._compat import run_sync
from .messages import ImageContent
from .prompts import IMAGE_DESCRIBE_SYSTEM

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage

logger = logging.getLogger(__name__)

_DESCRIBE_USER = "Describe this image concisely."


async def describe_images_async(
    model: object,
    images: list[ImageContent],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> list[str]:
    """Convert images to text descriptions via a vision-capable LLM.

    Each image is described individually. Failed descriptions fall back to
    a placeholder string so the pipeline can continue.
    """
    if not images:
        return []

    agent: Agent[None, str] = Agent(model, system_prompt=IMAGE_DESCRIBE_SYSTEM, output_type=str)
    descriptions: list[str] = []

    for image in images:
        try:
            image_part = ImageUrl(url=image.url) if image.url else None
            if image_part is None:
                descriptions.append("[image: undescribed]")
                continue
            result = await agent.run([_DESCRIBE_USER, image_part])
            if _on_usage is not None:
                _on_usage("describe_image", result.usage())
            descriptions.append(result.output)
        except Exception:
            logger.warning("Image description failed", exc_info=True)
            descriptions.append("[image: undescribed]")

    return descriptions


def describe_images(
    model: object,
    images: list[ImageContent],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> list[str]:
    """Convert images to text descriptions via a vision-capable LLM (sync)."""
    return run_sync(describe_images_async(model, images, _on_usage=_on_usage))
