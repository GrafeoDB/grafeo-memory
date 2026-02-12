"""Tests for the CLI tool."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from grafeo_memory.cli import _build_parser, main
from grafeo_memory.types import MemoryAction, MemoryEvent, SearchResult

# --- Fixtures ---


@dataclass
class FakeManager:
    """Minimal stand-in for MemoryManager to capture CLI calls."""

    calls: list = None

    def __post_init__(self):
        self.calls = []

    def add(self, text, *, memory_type="semantic", infer=True):
        self.calls.append(("add", text, memory_type, infer))
        return [MemoryEvent(action=MemoryAction.ADD, memory_id="1", text=text)]

    def search(self, query, *, k=10, memory_type=None):
        self.calls.append(("search", query, k, memory_type))
        return [SearchResult(memory_id="1", text="alice works at acme", score=0.95, user_id="u1")]

    def get_all(self, *, memory_type=None):
        self.calls.append(("get_all", memory_type))
        return [SearchResult(memory_id="1", text="alice works at acme", score=1.0, user_id="u1")]

    def update(self, memory_id, text):
        self.calls.append(("update", memory_id, text))
        return MemoryEvent(action=MemoryAction.UPDATE, memory_id=memory_id, text=text, old_text="old text")

    def delete(self, memory_id):
        self.calls.append(("delete", memory_id))
        return True

    def delete_all(self):
        self.calls.append(("delete_all",))
        return 3

    def history(self, memory_id):
        self.calls.append(("history", memory_id))
        return [{"event": "ADD", "new_text": "alice works at acme", "timestamp": 1000}]

    def summarize(self, *, preserve_recent=5, batch_size=20):
        self.calls.append(("summarize", preserve_recent, batch_size))
        return [
            MemoryEvent(action=MemoryAction.ADD, memory_id="10", text="consolidated"),
            MemoryEvent(action=MemoryAction.DELETE, memory_id="1", old_text="old1"),
        ]

    def close(self):
        pass


@pytest.fixture
def fake_manager():
    return FakeManager()


def _run(argv, fake_manager):
    """Run CLI with mocked manager."""
    with patch("grafeo_memory.cli._create_manager", return_value=fake_manager):
        main(argv)


# --- Parser tests ---


class TestParser:
    def test_add_args(self):
        parser = _build_parser()
        args = parser.parse_args(["add", "hello world"])
        assert args.command == "add"
        assert args.text == "hello world"
        assert args.memory_type == "semantic"
        assert not args.no_infer

    def test_add_procedural(self):
        parser = _build_parser()
        args = parser.parse_args(["add", "use dark mode", "--type", "procedural"])
        assert args.memory_type == "procedural"

    def test_search_args(self):
        parser = _build_parser()
        args = parser.parse_args(["search", "Where does Alice work?", "-k", "5"])
        assert args.command == "search"
        assert args.query == "Where does Alice work?"
        assert args.k == 5

    def test_delete_all_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["delete", "--all", "--yes"])
        assert args.command == "delete"
        assert args.delete_all is True
        assert args.yes is True

    def test_global_flags(self):
        parser = _build_parser()
        argv = ["--model", "anthropic:claude-3-5-haiku", "--db", "/tmp/test.db", "--user", "bob", "--json", "list"]
        args = parser.parse_args(argv)
        assert args.model == "anthropic:claude-3-5-haiku"
        assert args.db == "/tmp/test.db"
        assert args.user == "bob"
        assert args.output_json is True

    def test_yolo_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--yolo", "list"])
        assert args.yolo is True

    def test_no_yolo_by_default(self):
        parser = _build_parser()
        args = parser.parse_args(["list"])
        assert args.yolo is False

    def test_summarize_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["summarize"])
        assert args.preserve_recent == 5
        assert args.batch_size == 20


# --- Command tests ---


class TestAddCommand:
    def test_add_text(self, capsys, fake_manager):
        _run(["add", "Alice works at Acme"], fake_manager)
        assert fake_manager.calls[0] == ("add", "Alice works at Acme", "semantic", True)
        out = capsys.readouterr().out
        assert "ADD" in out

    def test_add_no_infer(self, capsys, fake_manager):
        _run(["add", "raw text", "--no-infer"], fake_manager)
        assert fake_manager.calls[0] == ("add", "raw text", "semantic", False)

    def test_add_procedural(self, capsys, fake_manager):
        _run(["add", "use dark mode", "-t", "procedural"], fake_manager)
        assert fake_manager.calls[0] == ("add", "use dark mode", "procedural", True)

    def test_add_json_output(self, capsys, fake_manager):
        _run(["--json", "add", "hello"], fake_manager)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "events" in data
        assert data["events"][0]["action"] == "add"

    def test_add_stdin(self, capsys, fake_manager, monkeypatch):
        monkeypatch.setattr("sys.stdin", MagicMock(read=MagicMock(return_value="from stdin\n")))
        _run(["add", "-"], fake_manager)
        assert fake_manager.calls[0][1] == "from stdin"


class TestSearchCommand:
    def test_search(self, capsys, fake_manager):
        _run(["search", "Alice"], fake_manager)
        assert fake_manager.calls[0] == ("search", "Alice", 10, None)
        out = capsys.readouterr().out
        assert "0.950" in out
        assert "alice works at acme" in out

    def test_search_json(self, capsys, fake_manager):
        _run(["--json", "search", "Alice"], fake_manager)
        data = json.loads(capsys.readouterr().out)
        assert data["results"][0]["score"] == 0.95

    def test_search_with_type(self, capsys, fake_manager):
        _run(["search", "query", "-t", "procedural"], fake_manager)
        assert fake_manager.calls[0] == ("search", "query", 10, "procedural")


class TestListCommand:
    def test_list(self, capsys, fake_manager):
        _run(["list"], fake_manager)
        assert fake_manager.calls[0] == ("get_all", None)
        out = capsys.readouterr().out
        assert "alice works at acme" in out

    def test_list_json(self, capsys, fake_manager):
        _run(["--json", "list"], fake_manager)
        data = json.loads(capsys.readouterr().out)
        assert "memories" in data


class TestUpdateCommand:
    def test_update(self, capsys, fake_manager):
        _run(["update", "42", "new text"], fake_manager)
        assert fake_manager.calls[0] == ("update", "42", "new text")
        out = capsys.readouterr().out
        assert "UPDATE" in out
        assert "old text" in out


class TestDeleteCommand:
    def test_delete_by_id(self, capsys, fake_manager):
        _run(["delete", "42"], fake_manager)
        assert fake_manager.calls[0] == ("delete", "42")
        out = capsys.readouterr().out
        assert "Deleted memory 42" in out

    def test_delete_all_with_yes(self, capsys, fake_manager):
        _run(["delete", "--all", "--yes"], fake_manager)
        assert fake_manager.calls[0] == ("delete_all",)
        out = capsys.readouterr().out
        assert "Deleted 3 memories" in out

    def test_delete_no_args_exits(self, fake_manager):
        with pytest.raises(SystemExit, match="1"):
            _run(["delete"], fake_manager)

    def test_delete_json(self, capsys, fake_manager):
        _run(["--json", "delete", "42"], fake_manager)
        data = json.loads(capsys.readouterr().out)
        assert data["deleted"] is True


class TestHistoryCommand:
    def test_history(self, capsys, fake_manager):
        _run(["history", "42"], fake_manager)
        assert fake_manager.calls[0] == ("history", "42")
        out = capsys.readouterr().out
        assert "ADD" in out

    def test_history_json(self, capsys, fake_manager):
        _run(["--json", "history", "42"], fake_manager)
        data = json.loads(capsys.readouterr().out)
        assert data["history"][0]["event"] == "ADD"


class TestSummarizeCommand:
    def test_summarize(self, capsys, fake_manager):
        _run(["summarize"], fake_manager)
        assert fake_manager.calls[0] == ("summarize", 5, 20)
        out = capsys.readouterr().out
        assert "consolidated" in out.lower() or "ADD" in out

    def test_summarize_custom_args(self, capsys, fake_manager):
        _run(["summarize", "--preserve-recent", "10", "--batch-size", "50"], fake_manager)
        assert fake_manager.calls[0] == ("summarize", 10, 50)


# --- Version and help ---


class TestVersionAndHelp:
    def test_version(self, capsys):
        main(["--version"])
        out = capsys.readouterr().out
        assert "grafeo-memory" in out
        assert "0.8.0" in out

    def test_no_command_shows_help(self, capsys):
        main([])
        out = capsys.readouterr().out
        assert "usage" in out.lower() or "grafeo-memory" in out


# --- Error handling ---


class TestErrorHandling:
    def test_manager_error_human(self, capsys, fake_manager):
        fake_manager.search = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(SystemExit, match="1"):
            _run(["search", "fail"], fake_manager)
        err = capsys.readouterr().err
        assert "boom" in err

    def test_manager_error_json(self, capsys, fake_manager):
        fake_manager.search = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(SystemExit, match="1"):
            _run(["--json", "search", "fail"], fake_manager)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["error"] == "boom"
