"""Message parsing: normalize str | dict | list[dict] input to text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class Message(TypedDict, total=False):
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    name: str  # optional actor identifier


@dataclass
class ImageContent:
    """An image extracted from a multimodal message.

    Either ``url`` (for remote images or data URIs) or ``data`` (raw bytes)
    will be populated, never both.
    """

    url: str | None = None
    data: bytes | None = None
    media_type: str = "image/png"


def parse_messages(input: str | dict | list[dict]) -> tuple[str, list[Message], list[ImageContent]]:
    """Normalize message input to (concatenated_text, parsed_messages, images).

    Accepts:
        - A plain string (treated as a single user message).
        - A single message dict with at least a "content" key.
        - A list of message dicts.

    When ``content`` is a list (OpenAI multimodal format), text parts are
    concatenated and ``image_url`` parts are collected as ``ImageContent``.

    Returns:
        A 3-tuple of (text for extraction, list of parsed messages, list of images).
    """
    if isinstance(input, str):
        return input, [{"role": "user", "content": input}], []

    if isinstance(input, dict):
        msgs: list[dict] = [input]
    elif isinstance(input, list):
        msgs = input
    else:
        raise TypeError(f"Expected str, dict, or list[dict], got {type(input).__name__}")

    if not msgs:
        return "", [], []

    parts: list[str] = []
    parsed: list[Message] = []
    images: list[ImageContent] = []

    for msg in msgs:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        name = msg.get("name")

        # Handle OpenAI multimodal content format: content is a list of parts
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype == "text":
                    text_parts.append(part.get("text", ""))
                elif ptype == "image_url":
                    image_url_obj = part.get("image_url", {})
                    url = image_url_obj.get("url", "") if isinstance(image_url_obj, dict) else ""
                    if url:
                        images.append(ImageContent(url=url))
            content = " ".join(text_parts)

        prefix = f"{name} ({role})" if name else role
        parts.append(f"{prefix}: {content}")

        entry: Message = {"role": role, "content": content}
        if name:
            entry["name"] = name
        parsed.append(entry)

    return "\n".join(parts), parsed, images
