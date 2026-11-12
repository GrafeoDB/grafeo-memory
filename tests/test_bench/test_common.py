"""Offline checks for the shared bench plumbing (resume, answer cache, single-flight calls)."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from grafeo_memory import CacheConfig
from grafeo_memory.bench.common import Cache, call_mistral, ingest_complete, remove_db
from grafeo_memory.common.cache import MemoryCache


def test_answer_cache_put_leaves_only_valid_json(tmp_path):
    p = tmp_path / "answers_cache.json"
    c = Cache(p)
    c.put("a", "x")
    c.put("b", "y")
    assert json.loads(p.read_text()) == {"a": "x", "b": "y"}
    assert [f.name for f in tmp_path.iterdir()] == ["answers_cache.json"]
    assert Cache(p).get("b") == "y"


def test_ingest_resume_skips_only_clean_logs(tmp_path):
    log = tmp_path / "ingest_x.json"
    assert not ingest_complete(log)
    log.write_text(json.dumps({"errors": [{"session": "D1"}]}))
    assert not ingest_complete(log)
    log.write_text(json.dumps({"errors": []}))
    assert ingest_complete(log)


def test_remove_db_drops_sidecars_but_not_neighbours(tmp_path):
    db = tmp_path / "db_conv-1.grafeo"
    db.write_text("x")
    (tmp_path / "db_conv-1.grafeo.wal").mkdir()
    (tmp_path / "db_conv-1.grafeo.wal" / "0001").write_text("x")
    (tmp_path / "db_conv-10.grafeo").write_text("keep")
    remove_db(db)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["db_conv-10.grafeo"]


def test_call_mistral_single_flight(tmp_path):
    calls = []

    def complete(**kw):
        calls.append(kw)
        time.sleep(0.2)
        msg = SimpleNamespace(content=" ok ")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(complete=complete))
    cache = MemoryCache(CacheConfig(persist_path=str(tmp_path / "c.jsonl"), max_entries=100))
    out: list[str] = []
    threads = [
        threading.Thread(target=lambda: out.append(call_mistral(client, "m", "same prompt", cache=cache)))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == ["ok"] * 4
    assert len(calls) == 1
