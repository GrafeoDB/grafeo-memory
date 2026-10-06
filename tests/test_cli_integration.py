"""The CLI against a real MemoryManager and a database file, one process call per command.

Each ``main()`` call opens the database, runs one command and closes it, as
the installed ``grafeo-memory`` command does, so these tests also cover
persistence across close and reopen. test_cli.py covers parsing and output
with a fake manager.
"""

from __future__ import annotations

import json

import pytest
from mock_llm import MockEmbedder

from grafeo_memory import cli


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    db_path = str(tmp_path / "cli.grafeo")
    monkeypatch.setattr(cli, "_create_embedder", lambda model: MockEmbedder(16))
    for var in ("GRAFEO_MEMORY_DB", "GRAFEO_MEMORY_USER", "GRAFEO_MEMORY_MODEL"):
        monkeypatch.delenv(var, raising=False)

    def _run(*args: str, json_output: bool = True, expect_exit: int | None = None):
        argv = ["--model", "test", "--db", db_path]
        if json_output:
            argv.append("--json")
        argv.extend(args)
        if expect_exit is None:
            cli.main(argv)
        else:
            with pytest.raises(SystemExit) as exc:
                cli.main(argv)
            assert exc.value.code == expect_exit
        out = capsys.readouterr()
        if json_output and out.out.strip():
            return json.loads(out.out)
        return out

    _run.db_path = db_path  # type: ignore[attr-defined]
    return _run


class TestCliAgainstFileDatabase:
    def test_memory_lifecycle_across_invocations(self, run):
        mid = run("add", "--no-infer", "alice likes tea")["events"][0]["memory_id"]

        assert [m["text"] for m in run("list")["memories"]] == ["alice likes tea"]

        event = run("update", mid, "alice likes green tea")["events"][0]
        assert event["old_text"] == "alice likes tea"

        history = run("history", mid)["history"]
        assert [h["event"] for h in history] == ["ADD", "UPDATE"]

        assert run("delete", mid) == {"deleted": True, "memory_id": mid}
        assert run("delete", mid) == {"deleted": False, "memory_id": mid}

        text = run("history", mid, json_output=False).out
        assert "[ADD]" in text and "[UPDATE]" in text and "[DELETE]" in text
        assert run("stats")["total_memories"] == 0

    def test_search_after_reopen(self, run):
        run("add", "--no-infer", "alice likes tea")
        run("add", "--no-infer", "bob plays chess")
        results = run("search", "alice likes tea", "-k", "1")["results"]
        assert [r["text"] for r in results] == ["alice likes tea"]

    def test_update_of_missing_memory_fails(self, run):
        out = run("update", "424242", "x", json_output=False, expect_exit=1)
        assert "not found" in out.err

    def test_users_are_isolated_and_delete_all(self, run):
        run("--user", "alix", "add", "--no-infer", "alix note")
        run("--user", "gus", "add", "--no-infer", "gus note")
        assert [m["text"] for m in run("--user", "alix", "list")["memories"]] == ["alix note"]
        assert run("--user", "alix", "delete", "--all", "--yes") == {"deleted_count": 1}
        assert run("--user", "alix", "list")["memories"] == []
        assert [m["text"] for m in run("--user", "gus", "list")["memories"]] == ["gus note"]

    def test_db_from_environment(self, run, monkeypatch, capsys):
        run("add", "--no-infer", "from flag")
        monkeypatch.setenv("GRAFEO_MEMORY_DB", run.db_path)
        cli.main(["--model", "test", "--json", "list"])
        assert [m["text"] for m in json.loads(capsys.readouterr().out)["memories"]] == ["from flag"]
