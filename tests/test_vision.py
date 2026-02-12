"""Tests for vision / multimodal support."""

from __future__ import annotations

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from grafeo_memory import AddResult, ImageContent, MemoryConfig, MemoryManager
from grafeo_memory.vision import describe_images
from tests.mock_llm import MockEmbedder, make_test_model


def _vision_model() -> FunctionModel:
    """FunctionModel that returns image descriptions as plain text."""

    def handler(messages: list, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="A whiteboard with architecture diagrams")])

    return FunctionModel(function=handler)


def _make_manager(model, **config_kwargs):
    embedder = MockEmbedder(dims=16)
    config = MemoryConfig(**config_kwargs)
    return MemoryManager(model, config, embedder=embedder)


# --- Unit tests for describe_images ---


class TestDescribeImages:
    def test_describe_single_image(self):
        model = _vision_model()
        images = [ImageContent(url="https://example.com/img.jpg")]
        descriptions = describe_images(model, images)
        assert len(descriptions) == 1
        assert "whiteboard" in descriptions[0].lower()

    def test_describe_multiple_images(self):
        model = _vision_model()
        images = [
            ImageContent(url="https://example.com/a.jpg"),
            ImageContent(url="https://example.com/b.jpg"),
        ]
        descriptions = describe_images(model, images)
        assert len(descriptions) == 2

    def test_describe_empty_list(self):
        model = _vision_model()
        descriptions = describe_images(model, [])
        assert descriptions == []

    def test_describe_image_no_url(self):
        """ImageContent with no URL returns placeholder."""
        model = _vision_model()
        images = [ImageContent()]  # no url, no data
        descriptions = describe_images(model, images)
        assert len(descriptions) == 1
        assert "undescribed" in descriptions[0]

    def test_describe_usage_callback(self):
        model = _vision_model()
        images = [ImageContent(url="https://example.com/img.jpg")]
        calls: list[tuple[str, RunUsage]] = []
        describe_images(model, images, _on_usage=lambda op, u: calls.append((op, u)))
        assert len(calls) == 1
        assert calls[0][0] == "describe_image"
        assert isinstance(calls[0][1], RunUsage)

    def test_describe_error_returns_placeholder(self):
        """LLM failure returns placeholder, doesn't raise."""

        def bad_handler(messages: list, info: AgentInfo) -> ModelResponse:
            raise ValueError("Vision model error")

        model = FunctionModel(function=bad_handler)
        images = [ImageContent(url="https://example.com/img.jpg")]
        descriptions = describe_images(model, images)
        assert len(descriptions) == 1
        assert "undescribed" in descriptions[0]


# --- Integration tests with manager ---


class TestVisionIntegration:
    def test_add_with_vision_enabled(self):
        """When enable_vision=True and images present, descriptions are added to text."""
        # Vision model for image description, then extraction model for fact/entity extraction
        call_idx = [0]

        def handler(messages: list, info: AgentInfo) -> ModelResponse:
            idx = call_idx[0]
            call_idx[0] += 1
            tool_name = info.output_tools[0].name if info.output_tools else None

            if tool_name is None:
                # Vision description call (output_type=str → no tools)
                return ModelResponse(parts=[TextPart(content="A photo of Alice at Acme Corp office")])

            # Extraction calls
            if idx == 1:  # extract_facts
                from pydantic_ai.messages import ToolCallPart

                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool_name, args={"facts": ["alice is at acme corp office"]})]
                )
            if idx == 2:  # extract_entities
                from pydantic_ai.messages import ToolCallPart

                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=tool_name,
                            args={
                                "entities": [{"name": "alice", "entity_type": "PERSON"}],
                                "relations": [],
                            },
                        )
                    ]
                )
            # reconcile (fast path - no existing memories)
            return ModelResponse(parts=[TextPart(content="")])

        model = FunctionModel(function=handler)
        manager = _make_manager(model, enable_vision=True)

        result = manager.add(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here's a photo of my office"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/office.jpg"}},
                ],
            },
            user_id="u1",
        )

        assert isinstance(result, AddResult)
        assert len(result) >= 1
        manager.close()

    def test_add_vision_disabled_ignores_images(self):
        """When enable_vision=False (default), images are silently dropped."""
        model = make_test_model(
            [
                {"facts": ["user mentioned a photo"]},
                {"entities": [], "relations": []},
            ]
        )
        manager = _make_manager(model, enable_vision=False)

        result = manager.add(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here's a photo"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                ],
            },
            user_id="u1",
        )

        assert isinstance(result, AddResult)
        # Text was still extracted and processed normally
        assert len(result) >= 1
        manager.close()

    def test_add_plain_text_unaffected_by_vision_config(self):
        """Vision config has no effect on plain text adds."""
        model = make_test_model(
            [
                {"facts": ["alice likes hiking"]},
                {"entities": [], "relations": []},
            ]
        )
        manager = _make_manager(model, enable_vision=True)

        result = manager.add("Alice likes hiking", user_id="u1")
        assert isinstance(result, AddResult)
        assert len(result) >= 1
        manager.close()

    def test_vision_usage_tracked(self):
        """Vision LLM calls should appear in usage tracking."""

        def handler(messages: list, info: AgentInfo) -> ModelResponse:
            tool_name = info.output_tools[0].name if info.output_tools else None
            if tool_name is None:
                return ModelResponse(parts=[TextPart(content="A landscape photo")])
            from pydantic_ai.messages import ToolCallPart

            if "facts" in str(info.output_tools[0]):
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool_name, args={"facts": ["landscape photo seen"]})]
                )
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args={"entities": [], "relations": []})])

        model = FunctionModel(function=handler)
        calls: list[tuple[str, RunUsage]] = []
        manager = _make_manager(
            model,
            enable_vision=True,
            usage_callback=lambda op, u: calls.append((op, u)),
        )

        manager.add(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Photo"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                ],
            },
            user_id="u1",
        )

        ops = [c[0] for c in calls]
        assert "describe_image" in ops
        manager.close()
