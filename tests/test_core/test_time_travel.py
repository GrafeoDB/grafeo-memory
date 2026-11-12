"""Time travel (grafeo 0.6.2 history retention): MemoryConfig.history_retention,
MemoryManager.as_of / execute_as_of / get_memory_as_of.

The manager opens its database with the configured retention, so as_of reads
can reach back. A database file stores the retention, so a reopen keeps it.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import grafeo
import pytest
from conftest import make_manager

from grafeo_memory import MemoryConfig
from grafeo_memory.common.types import MEMORY_LABEL
from grafeo_memory.core.history import HistoryEntry, record_history


def _now() -> datetime:
    return datetime.now(UTC)


class TestConfig:
    def test_defaults_to_none(self):
        assert MemoryConfig().history_retention is None

    @pytest.mark.parametrize("value", ["off", "OFF", "forever", "Forever"])
    def test_keywords(self, value):
        assert MemoryConfig(history_retention=value).history_retention == value.lower()

    def test_seconds_become_a_timedelta(self):
        assert MemoryConfig(history_retention=300).history_retention == timedelta(seconds=300)

    def test_timedelta_passes_through(self):
        assert MemoryConfig(history_retention=timedelta(hours=1)).history_retention == timedelta(hours=1)

    @pytest.mark.parametrize("value", ["sometimes", -5, 0, True, timedelta(seconds=-1)])
    def test_invalid(self, value):
        with pytest.raises(ValueError, match="history_retention"):
            MemoryConfig(history_retention=value)


class TestTimeTravel:
    def test_execute_as_of_sees_earlier_state(self):
        manager = make_manager([], history_retention="forever")
        db = manager._db
        db.execute(f"INSERT (:{MEMORY_LABEL} {{text: 'first', user_id: 'test_user'}})")
        point = _now()
        time.sleep(0.05)
        db.execute(f"INSERT (:{MEMORY_LABEL} {{text: 'second', user_id: 'test_user'}})")
        db.execute(f"MATCH (m:{MEMORY_LABEL} {{text: 'first'}}) SET m.text = 'first!'")

        rows = manager.execute_as_of(f"MATCH (m:{MEMORY_LABEL}) RETURN m.text AS t ORDER BY t", point)
        assert rows == [{"t": "first"}]
        rows = manager.execute_as_of(f"MATCH (m:{MEMORY_LABEL}) RETURN m.text AS t ORDER BY t", _now())
        assert rows == [{"t": "first!"}, {"t": "second"}]
        manager.close()

    def test_execute_as_of_with_params_and_epoch(self):
        manager = make_manager([], history_retention="forever")
        db = manager._db
        db.execute(f"INSERT (:{MEMORY_LABEL} {{text: 'a'}}), (:{MEMORY_LABEL} {{text: 'b'}})")
        epoch = db.epoch_as_of(_now())
        time.sleep(0.05)
        db.execute(f"INSERT (:{MEMORY_LABEL} {{text: 'c'}})")
        rows = manager.execute_as_of(
            f"MATCH (m:{MEMORY_LABEL}) WHERE m.text = $t RETURN m.text AS t", epoch, {"t": "c"}
        )
        assert rows == [], "the memory 'c' did not exist at that epoch"
        manager.close()

    def test_get_memory_as_of(self):
        manager = make_manager([], history_retention="forever")
        db = manager._db
        node = db.create_node([MEMORY_LABEL], {"text": "before", "user_id": "test_user"})
        point = _now()
        time.sleep(0.05)
        db.set_node_property(node.id, "text", "after")

        old = manager.get_memory_as_of(str(node.id), point)
        assert old is not None and old["properties"]["text"] == "before" and old["labels"] == [MEMORY_LABEL]
        assert manager.get_memory_as_of(str(node.id), _now())["properties"]["text"] == "after"
        manager.close()

    def test_get_memory_as_of_unknown_or_not_yet_existing(self):
        manager = make_manager([], history_retention="forever")
        db = manager._db
        point = _now()
        time.sleep(0.05)
        node = db.create_node([MEMORY_LABEL], {"text": "new"})
        assert manager.get_memory_as_of(str(node.id), point) is None, "created after the point"
        assert manager.get_memory_as_of("not-a-number", point) is None
        manager.close()

    def test_as_of_view_is_read_only(self):
        manager = make_manager([], history_retention="forever")
        db = manager._db
        db.execute(f"INSERT (:{MEMORY_LABEL} {{text: 'kept'}})")
        point = _now()
        time.sleep(0.05)
        with manager.as_of(point) as view:
            assert list(view.execute(f"MATCH (m:{MEMORY_LABEL}) RETURN m.text AS t")) == [{"t": "kept"}]
            with pytest.raises(grafeo.GrafeoError):
                view.execute(f"INSERT (:{MEMORY_LABEL})")
        assert db.node_count == 1
        manager.close()

    def test_without_retention_the_engine_refuses(self):
        manager = make_manager([])  # default: retention off
        manager._db.execute(f"INSERT (:{MEMORY_LABEL} {{text: 'x'}})")
        with pytest.raises(grafeo.GrafeoError, match="history"):
            manager.execute_as_of("MATCH (m) RETURN count(m) AS c", _now())
        manager.close()

    def test_external_db_without_as_of_gets_a_clear_error(self):
        manager = make_manager([], history_retention="forever")
        manager._db = object()  # type: ignore[assignment]
        with pytest.raises(TypeError, match="as_of"):
            manager.as_of(_now())
        with pytest.raises(TypeError, match="execute_as_of"):
            manager.execute_as_of("MATCH (n) RETURN n", _now())

    def test_file_database_stores_retention(self, tmp_path):
        path = str(tmp_path / "mem.grafeo")
        manager = make_manager([], db_path=path, history_retention="forever")
        manager._db.execute(f"INSERT (:{MEMORY_LABEL} {{text: 'persisted'}})")
        point = _now()
        time.sleep(0.05)
        manager._db.execute(f"INSERT (:{MEMORY_LABEL} {{text: 'later'}})")
        manager.close()

        # Reopened without asking for retention: the file kept it.
        reopened = make_manager([], db_path=path)
        rows = reopened.execute_as_of(f"MATCH (m:{MEMORY_LABEL}) RETURN m.text AS t ORDER BY t", point)
        assert rows == [{"t": "persisted"}]
        reopened.close()

    def test_history_entry_timestamp_as_a_point(self):
        # history() entries carry wall-clock ms; convert to an aware datetime for as_of.
        manager = make_manager([], history_retention="forever")
        db = manager._db
        node = db.create_node([MEMORY_LABEL], {"text": "v1", "user_id": "test_user"})
        entry = HistoryEntry(event="ADD", new_text="v1")
        record_history(db, node.id, entry, user_id="test_user")
        time.sleep(0.05)
        db.set_node_property(node.id, "text", "v2")

        point = datetime.fromtimestamp(entry.timestamp / 1000, UTC)
        assert manager.get_memory_as_of(str(node.id), point)["properties"]["text"] == "v1"
        manager.close()


class TestAsyncFacade:
    def test_async_manager_exposes_time_travel(self):
        import asyncio

        from mock_llm import MockEmbedder, make_test_model

        from grafeo_memory import AsyncMemoryManager

        async def _run():
            config = MemoryConfig(user_id="test_user", embedding_dimensions=16, history_retention="forever")
            manager = AsyncMemoryManager(make_test_model([]), config, embedder=MockEmbedder(16))
            manager._db.execute(f"INSERT (:{MEMORY_LABEL} {{text: 'async'}})")
            point = _now()
            time.sleep(0.05)
            manager._db.execute(f"INSERT (:{MEMORY_LABEL} {{text: 'later'}})")

            rows = await manager.execute_as_of(f"MATCH (m:{MEMORY_LABEL}) RETURN m.text AS t ORDER BY t", point)
            assert rows == [{"t": "async"}]
            with manager.as_of(point) as view:
                assert list(view.execute(f"MATCH (m:{MEMORY_LABEL}) RETURN count(m) AS c")) == [{"c": 1}]
            manager.close()

        asyncio.run(_run())
