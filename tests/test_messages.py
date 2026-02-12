"""Tests for message parsing."""

from grafeo_memory.messages import ImageContent, parse_messages


class TestParseMessages:
    def test_string_input(self):
        text, msgs, images = parse_messages("hello world")
        assert text == "hello world"
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello world"
        assert images == []

    def test_single_dict(self):
        text, msgs, images = parse_messages({"role": "assistant", "content": "I can help"})
        assert text == "assistant: I can help"
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert images == []

    def test_list_of_dicts(self):
        text, msgs, images = parse_messages(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        )
        assert "user: Hi" in text
        assert "assistant: Hello" in text
        assert len(msgs) == 2
        assert images == []

    def test_dict_with_name(self):
        text, msgs, images = parse_messages({"role": "user", "content": "I like hiking", "name": "alice"})
        assert text == "alice (user): I like hiking"
        assert msgs[0]["name"] == "alice"
        assert images == []

    def test_empty_list(self):
        text, msgs, images = parse_messages([])
        assert text == ""
        assert msgs == []
        assert images == []

    def test_dict_defaults_to_user_role(self):
        text, msgs, images = parse_messages({"content": "no role"})
        assert msgs[0]["role"] == "user"
        assert "user: no role" in text
        assert images == []

    def test_invalid_type_raises(self):
        import pytest

        with pytest.raises(TypeError, match="Expected str"):
            parse_messages(42)


class TestMultimodalMessages:
    def test_image_url_extracted(self):
        text, _msgs, images = parse_messages(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here's a photo"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                ],
            }
        )
        assert "Here's a photo" in text
        assert len(images) == 1
        assert images[0].url == "https://example.com/img.jpg"

    def test_data_uri_extracted(self):
        data_uri = "data:image/png;base64,iVBORw0KGgo="
        text, _msgs, images = parse_messages(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "My diagram"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        )
        assert "My diagram" in text
        assert len(images) == 1
        assert images[0].url == data_uri

    def test_multiple_images(self):
        _text, _msgs, images = parse_messages(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Two images"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/b.jpg"}},
                ],
            }
        )
        assert len(images) == 2

    def test_text_only_multimodal_format(self):
        """Content list with only text parts — no images extracted."""
        text, _msgs, images = parse_messages(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "just text"},
                ],
            }
        )
        assert text == "user: just text"
        assert images == []

    def test_image_only_no_text(self):
        """Content list with only an image — text is empty."""
        text, _msgs, images = parse_messages(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                ],
            }
        )
        assert text == "user: "
        assert len(images) == 1

    def test_multimodal_across_messages(self):
        """Images collected across multiple messages in a list."""
        text, _msgs, images = parse_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/1.jpg"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Second"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/2.jpg"}},
                    ],
                },
            ]
        )
        assert "First" in text
        assert "Second" in text
        assert len(images) == 2

    def test_mixed_string_and_multimodal(self):
        """List with both plain string content and multimodal content."""
        text, _msgs, images = parse_messages(
            [
                {"role": "user", "content": "plain text"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "with image"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                    ],
                },
            ]
        )
        assert "plain text" in text
        assert "with image" in text
        assert len(images) == 1

    def test_image_content_defaults(self):
        ic = ImageContent(url="https://example.com/img.jpg")
        assert ic.url == "https://example.com/img.jpg"
        assert ic.data is None
        assert ic.media_type == "image/png"

    def test_empty_image_url_skipped(self):
        """Empty URL string should not produce an ImageContent."""
        _text, _msgs, images = parse_messages(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": ""}},
                ],
            }
        )
        assert images == []
