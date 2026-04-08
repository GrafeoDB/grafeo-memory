"""Tests for bi-temporal model: valid_at/invalid_at, temporal annotation, point-in-time search."""

from __future__ import annotations

from datetime import UTC, datetime

from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import MemoryConfig, MemoryManager, detect_temporal_hints
from grafeo_memory.extraction.temporal import _parse_date_to_epoch_ms


def _make_manager(outputs, dims=16, **config_kwargs):
    model = make_test_model(outputs)
    embedder = MockEmbedder(dims)
    db = config_kwargs.pop("db", None)
    defaults = {"db_path": None, "user_id": "test_user", "embedding_dimensions": dims}
    defaults.update(config_kwargs)
    config = MemoryConfig(**defaults)  # type: ignore[invalid-argument-type]
    return MemoryManager(model, config, embedder=embedder, db=db) if db else MemoryManager(model, config, embedder=embedder)


def _extraction_output(facts, entities=None, relations=None):
    return {
        "facts": facts,
        "entities": entities or [],
        "relations": relations or [],
    }


def _temporal_annotation_output(annotations):
    return {"annotations": annotations}


# ---------------------------------------------------------------------------
# _parse_date_to_epoch_ms (pure function tests)
# ---------------------------------------------------------------------------


class TestParseDateToEpochMs:
    def test_iso_date(self):
        ms = _parse_date_to_epoch_ms("2024-01-15")
        expected = int(datetime(2024, 1, 15, tzinfo=UTC).timestamp() * 1000)
        assert ms == expected

    def test_iso_datetime(self):
        ms = _parse_date_to_epoch_ms("2024-06-15T10:30:00")
        expected = int(datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC).timestamp() * 1000)
        assert ms == expected

    def test_year_only(self):
        ms = _parse_date_to_epoch_ms("2024")
        expected = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
        assert ms == expected

    def test_none_returns_none(self):
        assert _parse_date_to_epoch_ms(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_date_to_epoch_ms("") is None

    def test_unparseable_returns_none(self):
        assert _parse_date_to_epoch_ms("not a date") is None

    def test_whitespace_stripped(self):
        ms = _parse_date_to_epoch_ms("  2024-03-01  ")
        expected = int(datetime(2024, 3, 1, tzinfo=UTC).timestamp() * 1000)
        assert ms == expected


# ---------------------------------------------------------------------------
# Bi-temporal memory creation
# ---------------------------------------------------------------------------


class TestBiTemporalAdd:
    def test_add_with_bitemporal_sets_valid_at(self):
        """When enable_bitemporal=True, temporal annotation runs and sets valid_at."""
        jan_15 = int(datetime(2024, 1, 15, tzinfo=UTC).timestamp() * 1000)
        manager = _make_manager(
            [
                # 1. Combined extraction
                _extraction_output(["alice started at acme corp in january 2024"]),
                # 2. Temporal annotation
                _temporal_annotation_output([{"fact_index": 0, "valid_at": "2024-01-15", "invalid_at": None}]),
                # 3. Reconciliation (no existing memories -> ADD)
                {"decisions": [{"action": "add", "text": "alice started at acme corp in january 2024"}]},
                # 4. Entity extraction for graph search during reconciliation
                {"entities": [], "relations": []},
            ],
            enable_bitemporal=True,
        )
        events = manager.add("Alice started at Acme Corp in January 2024")
        assert len(events) >= 1
        assert events[0].valid_at == jan_15

        # Verify the node has the property
        memories = manager.get_all()
        assert len(memories) >= 1
        assert memories[0].valid_at == jan_15
        manager.close()

    def test_add_without_bitemporal_no_valid_at(self):
        """When enable_bitemporal=False (default), no temporal annotation runs."""
        manager = _make_manager(
            [
                _extraction_output(["alice works at acme"]),
                {"decisions": [{"action": "add", "text": "alice works at acme"}]},
                {"entities": [], "relations": []},
            ],
        )
        events = manager.add("Alice works at Acme")
        assert len(events) >= 1
        assert events[0].valid_at is None

        memories = manager.get_all()
        assert len(memories) >= 1
        assert memories[0].valid_at is None
        manager.close()

    def test_update_sets_invalid_at(self):
        """UPDATE via reconciliation sets invalid_at on the old memory when bitemporal."""
        jan_15 = int(datetime(2024, 1, 15, tzinfo=UTC).timestamp() * 1000)
        march_1 = int(datetime(2024, 3, 1, tzinfo=UTC).timestamp() * 1000)

        # We need a two-phase setup: first add creates the memory, then we
        # build a second manager whose reconciliation mock references the real ID.
        import grafeo

        db = grafeo.GrafeoDB()

        manager1 = _make_manager(
            [
                _extraction_output(["alice works at acme"]),
                _temporal_annotation_output([{"fact_index": 0, "valid_at": "2024-01-15", "invalid_at": None}]),
                {"decisions": [{"action": "add", "text": "alice works at acme"}]},
            ],
            enable_bitemporal=True,
            db=db,
        )
        events1 = manager1.add("Alice works at Acme")
        assert len(events1) >= 1
        old_memory_id = events1[0].memory_id
        manager1.close()

        # Second add targets the real memory ID so UPDATE path executes.
        # reconciliation_threshold=0.0 ensures search_similar finds the first memory,
        # so the reconciliation LLM mock is used instead of the fast-path ADD.
        manager2 = _make_manager(
            [
                _extraction_output(["alice now works at globex"]),
                _temporal_annotation_output([{"fact_index": 0, "valid_at": "2024-03-01", "invalid_at": None}]),
                {"decisions": [{"action": "update", "text": "alice now works at globex", "target_memory_id": old_memory_id}]},
            ],
            enable_bitemporal=True,
            reconciliation_threshold=0.0,
            db=db,
        )
        events2 = manager2.add("Alice now works at Globex")
        assert len(events2) >= 1
        assert events2[0].action.value == "update"
        assert events2[0].valid_at == march_1

        # Verify invalid_at was set on the old (expired) memory
        memories = manager2.get_all(include_expired=True)
        old = [m for m in memories if m.memory_id == old_memory_id]
        assert len(old) == 1
        assert old[0].invalid_at == march_1
        assert old[0].valid_at == jan_15
        manager2.close()


# ---------------------------------------------------------------------------
# Point-in-time search
# ---------------------------------------------------------------------------


class TestPointInTimeSearch:
    def _setup_two_memories(self):
        """Create a manager with two memories at different valid_at times."""
        import grafeo

        db = grafeo.GrafeoDB()

        # Create manager with raw adds (no LLM) to control valid_at directly
        model = make_test_model(
            [
                # search will need entity extraction
                {"entities": [], "relations": []},
                {"entities": [], "relations": []},
                {"entities": [], "relations": []},
            ]
        )
        embedder = MockEmbedder(16)
        config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16, enable_bitemporal=True)
        manager = MemoryManager(model, config, embedder=embedder, db=db)

        # Manually create memories with specific valid_at
        jan = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)
        jun = int(datetime(2024, 6, 1, tzinfo=UTC).timestamp() * 1000)

        emb1 = embedder.embed(["alice works at acme"])[0]
        emb2 = embedder.embed(["alice works at globex"])[0]

        mid1 = manager._create_memory(
            "alice works at acme",
            emb1,
            "test_user",
            None,
            None,
            jan,
            valid_at=jan,
        )
        mid2 = manager._create_memory(
            "alice works at globex",
            emb2,
            "test_user",
            None,
            None,
            jun,
            valid_at=jun,
        )
        # Set invalid_at on first memory (she left Acme when she joined Globex)
        db.set_node_property(int(mid1), "invalid_at", jun)

        return manager, mid1, mid2

    def test_point_in_time_includes_valid(self):
        """Memory with valid_at before query time and no invalid_at is returned."""
        manager, mid1, mid2 = self._setup_two_memories()
        # Query at July 2024: only mid2 should match (mid1 was invalidated in June)
        jul = int(datetime(2024, 7, 1, tzinfo=UTC).timestamp() * 1000)
        results = manager.search("alice work", point_in_time=jul)
        result_ids = {r.memory_id for r in results}
        assert mid2 in result_ids
        # mid1 has invalid_at=jun which is before jul, so excluded
        assert mid1 not in result_ids
        manager.close()

    def test_point_in_time_excludes_future(self):
        """Memory with valid_at after query time is excluded."""
        manager, mid1, mid2 = self._setup_two_memories()
        # Query at March 2024: mid2 (valid_at=June) should be excluded
        mar = int(datetime(2024, 3, 1, tzinfo=UTC).timestamp() * 1000)
        results = manager.search("alice work", point_in_time=mar)
        result_ids = {r.memory_id for r in results}
        assert mid2 not in result_ids
        # mid1 was valid in March (valid Jan, invalid Jun)
        assert mid1 in result_ids
        manager.close()

    def test_point_in_time_none_permissive(self):
        """Memory with valid_at=None is always included (backward compat)."""
        import grafeo

        db = grafeo.GrafeoDB()
        model = make_test_model([{"entities": [], "relations": []}])
        embedder = MockEmbedder(16)
        config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)
        manager = MemoryManager(model, config, embedder=embedder, db=db)

        emb = embedder.embed(["some fact"])[0]
        mid = manager._create_memory("some fact", emb, "test_user", None, None, 1000)
        # No valid_at set

        results = manager.search("some fact", point_in_time=500)
        assert any(r.memory_id == mid for r in results)
        manager.close()


# ---------------------------------------------------------------------------
# SearchResult fields
# ---------------------------------------------------------------------------


class TestSearchResultFields:
    def test_get_all_includes_valid_at(self):
        """get_all() populates valid_at and invalid_at from node properties."""
        import grafeo

        db = grafeo.GrafeoDB()
        model = make_test_model([])
        embedder = MockEmbedder(16)
        config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)
        manager = MemoryManager(model, config, embedder=embedder, db=db)

        emb = embedder.embed(["test"])[0]
        mid = manager._create_memory("test", emb, "test_user", None, None, 1000, valid_at=2000)
        db.set_node_property(int(mid), "invalid_at", 3000)

        memories = manager.get_all()
        assert len(memories) == 1
        assert memories[0].valid_at == 2000
        assert memories[0].invalid_at == 3000
        manager.close()


# ---------------------------------------------------------------------------
# Temporal hints: point-in-time detection
# ---------------------------------------------------------------------------


class TestPointInTimeHints:
    def test_as_of_detected(self):
        hints = detect_temporal_hints("What was Alice's job as of 2023?")
        assert hints.point_in_time_hint
        assert hints.is_temporal
        assert hints.include_expired

    def test_back_then_detected(self):
        hints = detect_temporal_hints("What did the team look like back then?")
        assert hints.point_in_time_hint

    def test_at_the_time_detected(self):
        hints = detect_temporal_hints("Who was the manager at the time?")
        assert hints.point_in_time_hint

    def test_in_year_detected(self):
        hints = detect_temporal_hints("What projects were active in 2022?")
        assert hints.point_in_time_hint

    def test_no_false_positive(self):
        hints = detect_temporal_hints("Where does Alice work now?")
        assert not hints.point_in_time_hint
