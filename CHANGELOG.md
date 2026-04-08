# Changelog

All notable changes to grafeo-memory are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 0.2.2

Bi-temporal validity tracking, episode-based provenance, and community summaries for richer temporal knowledge graphs.

### Added

- **Bi-temporal model** (opt-in): every Memory node can now carry `valid_at` (when the fact became true in reality) and `invalid_at` (when it ceased being true), orthogonal to the existing system-time properties `created_at`/`expired_at`. Enable with `enable_bitemporal=True`
- **Temporal annotation extraction**: new LLM step parses real-world dates from message context (ISO-8601, year-only, relative dates) and attaches them to extracted facts. New module `extraction/temporal.py` with `annotate_temporal_async()`
- **Point-in-time queries**: `search(query, point_in_time=epoch_ms)` filters to facts that were valid at a specific moment. Memories with no `valid_at` are included permissively (backward compatible)
- **Point-in-time hint detection**: `detect_temporal_hints()` now recognizes "as of", "at that time", "back then", "in YYYY" patterns and sets `point_in_time_hint=True` with automatic expired-memory inclusion
- **Episode provenance** (opt-in): `add()` creates `:Episode` nodes storing the raw message text, linked to produced memories via `PRODUCED` edges and to extracted entities via `MENTIONS` edges. Enable with `enable_episodes=True`
- **Session replay via episodes**: `NEXT_EPISODE` edges chain episodes within a run. When episodes are enabled, `LEADS_TO` edges between memories are no longer created (episodes handle session ordering)
- **`get_episodes()`**: retrieve all episodes for a user, optionally filtered by `session_id`
- **`get_provenance(memory_id)`**: trace any memory back to the raw episode that produced it
- **`episode_chain(episode_id, direction)`**: follow `NEXT_EPISODE` edges forward, backward, or both for session replay
- **Community summaries** (opt-in): after Louvain community detection, creates `:Community` nodes with LLM-generated `name` and `summary` for entity clusters (>= 2 members). Lazy materialization on next `search()`. Enable with `enable_community_summaries=True` (requires `enable_graph_algorithms=True`)
- **`get_communities()`**: retrieve all detected communities and their summaries
- **Community context in search**: relevant community summaries are discoverable via `get_community_context()` for entity-based lookups
- New node labels: `Episode`, `Community`
- New edge types: `PRODUCED`, `MENTIONS`, `NEXT_EPISODE`, `HAS_MEMBER`
- New config options: `enable_bitemporal`, `enable_episodes`, `enable_community_summaries`
- New types: `EpisodeResult`, `CommunityInfo`, `TemporalAnnotation`, `TemporalAnnotationOutput`, `CommunitySummaryOutput`
- New exports: `EPISODE_LABEL`, `PRODUCED_EDGE`, `MENTIONS_EDGE`, `NEXT_EPISODE_EDGE`, `COMMUNITY_LABEL`, `HAS_MEMBER_EDGE`, `EpisodeResult`, `CommunityInfo`
- New modules: `extraction/temporal.py`, `communities.py`
- New prompts: `TEMPORAL_ANNOTATION_SYSTEM/USER`, `COMMUNITY_SUMMARY_SYSTEM/USER`

### Changed

- `SearchResult` gains two new fields: `valid_at`, `invalid_at` (both default `None`, backward compatible)
- `MemoryEvent` gains `valid_at` field (default `None`, backward compatible)
- `ExtractionResult` gains `temporal_annotations` field (default empty dict, backward compatible)
- `MemoryStats` gains `episode_count` and `community_count` fields (both default 0)
- `search()`, `explain()` gain `point_in_time` keyword parameter on both `MemoryManager` and `AsyncMemoryManager`
- `_expire_memory()` sets `invalid_at` on the expired node when bi-temporal mode is enabled
- `_execute_decisions()` passes `valid_at` from temporal annotations to new memories and sets `invalid_at` on superseded memories
- `_stats_impl()` now counts Episode and Community nodes
- `db_info` in stats includes `episode_node_count` and `community_node_count`
- MCP `memory_search` tool gains `point_in_time` parameter

## [0.2.1] - 2026-04-07

Test quality, error surfacing, documentation improvements, and shared database support.

### Added

- **Shared database support**: `MemoryManager` and `AsyncMemoryManager` accept an optional `db=` keyword argument to inject an existing `GrafeoDB` instance instead of creating one internally. `close()` respects ownership and leaves externally provided databases open ([#18](https://github.com/GrafeoDB/grafeo-memory/issues/18), [#19](https://github.com/GrafeoDB/grafeo-memory/pull/19), thanks [@Michaelzag](https://github.com/Michaelzag))
- **Multi-tenant safety**: all internal operations are now scoped to Memory and Entity nodes, making it safe to share a database with non-memory data

### Fixed

- **Graph metrics pollution**: `_recompute_graph_metrics()` now writes `_pagerank`, `_betweenness`, `_community` only to Memory/Entity nodes instead of every node in the database
- **Unlabeled Cypher queries**: `temporal_chain()` and `_get_node_relations()` now require `:Memory` labels on match patterns, preventing traversal from foreign nodes
- **N+1 entity lookup**: `_find_or_create_entity` and `graph_search` replaced global `find_nodes_by_property` + Python filtering with single label-scoped Cypher queries, reducing FFI crossings
- **Stats data leak**: `_stats_impl()` no longer exposes raw `db.info()`, returning only memory-scoped counts. MCP `memory://stats` resource now delegates to the manager's scoped stats
- **Error surfacing tests**: broken embedder on add/search now verified to raise, not silently return empty results
- **Persistence lifecycle tests**: write, close, reopen round-trip with `tmp_path`
- **Multi-session regression tests**: sequential open/close/open guards against event loop corruption
- **Concurrency tests**: concurrent async adds for same user (lock verification) and different users (isolation)
- **Reconciliation boundary test**: controlled embeddings at threshold boundary verify UPDATE triggers
- **History ordering test**: LEADS_TO chain verified for session-order retrieval
- **Delete cascade test**: orphaned entity behavior after memory deletion documented
- **Memory type filter test**: search with `memory_type` filter verified
- **Summarize test**: non-empty summary from related memories
- **MCP tool smoke tests**: all tool functions imported and called with mock context

### Changed

- `test_search_with_broken_embedder` now uses `pytest.raises` instead of silently passing
- Concurrency tests use `async with` context manager for guaranteed cleanup
- Error surfacing tests wrapped in `try/finally` for manager cleanup
- Shared `make_manager` helper extracted to `conftest.py`
- README: added model compatibility table (OpenAI, Anthropic, Mistral, Groq, Google)
- README: documented reconciliation threshold tuning and `history()` return type
- 435 tests passing

## [0.2.0] - 2026-03-26

Temporal reasoning, multi-hop search and engine integration for better long running memory performance.

### Added

- **Soft expiry**: reconciliation DELETE now sets `expired_at` instead of removing nodes. Old facts remain queryable for point-in-time reconstruction. `include_expired` parameter on `search()` and `get_all()` to access expired memories explicitly
- **Soft-expiry UPDATE**: creates a new memory node and a `SUPERSEDES` edge to the expired original, preserving full lineage
- **Time-range search**: `time_before` and `time_after` parameters on `search()` filter results by `created_at` timestamp
- **Temporal keyword detection**: rule-based heuristics detect "when", "first", "used to", "how many days", etc. in queries. Auto-expands result limits, includes expired memories, and sorts chronologically when appropriate. New `detect_temporal_hints()` utility and `TemporalHints` dataclass
- **`LEADS_TO` edges**: sequential `add()` calls within a session (`run_id` or `session_id`) create temporal ordering edges, forming causal chains across a conversation
- **`temporal_chain()` method**: follows `LEADS_TO` edges forward, backward, or both, with configurable max depth. Available on `MemoryManager` and `AsyncMemoryManager`
- **2-hop graph traversal**: `graph_search_depth=2` config option enables `Memory -> Entity -> RELATION -> Entity -> Memory` paths, surfacing indirectly connected memories. 2-hop results score at 0.7x to prefer direct matches
- **Graph algorithm scoring**: `enable_graph_algorithms=True` runs PageRank, betweenness centrality, and Louvain community detection after each `add()`. Results cached as node properties (`_pagerank`, `_betweenness`, `_community`), recomputed only when the graph changes
- **Cross-session entity reinforcement**: `cross_session_factor` config option boosts memories connected to high-PageRank/betweenness nodes, promoting cross-session hubs
- **MMR diverse search**: `search(query, diverse=True)` uses the engine's `mmr_search()` for Maximal Marginal Relevance retrieval, avoiding same-session clustering. `mmr_lambda` config tunes relevance vs. diversity (default 0.5)
- **Session-grouped results**: `search(query, grouped=True)` returns results organized by `session_id`, chronologically sorted within each group
- **Property indexes**: automatic `create_property_index()` on `user_id`, `created_at`, `memory_type`, and `name` at startup for O(1) filtered lookups
- **Batch node creation**: `_raw_add_batch()` uses `batch_create_nodes_with_props()` for bulk ingestion, with per-node fallback when unavailable
- **`learned_at` timestamp**: every memory now stores when it was first learned, distinct from `created_at` (which tracks node creation). Populated on `SearchResult`
- **`session_id` on SearchResult**: search results now carry `session_id` for downstream grouping
- **Native CDC history**: `record_history()` is a no-op when the engine exposes `node_history()` (change data capture). `get_history()` prefers engine-native CDC events, falling through to legacy `:History` nodes
- New edge types: `SUPERSEDES_EDGE`, `LEADS_TO_EDGE`
- New config options: `graph_search_depth`, `mmr_lambda`, `cross_session_factor`, `enable_graph_algorithms`
- New exports: `SUPERSEDES_EDGE`, `LEADS_TO_EDGE`, `TemporalHints`, `detect_temporal_hints`, `apply_cross_session_boost`
- New module: `temporal.py`
- 57 new tests across `test_temporal.py`, `test_multi_hop.py`, `test_new_features.py`, and `test_history.py`

### Changed

- **BREAKING: reconciliation DELETE is now soft expiry**. Nodes are preserved with `expired_at` set. Use `manager.delete()` for hard deletion. Existing code calling `get_all()` or `search()` is unaffected (expired memories excluded by default)
- **BREAKING: reconciliation UPDATE creates a new node** instead of mutating in place. The old node is expired and linked via `SUPERSEDES`. `MemoryEvent.memory_id` now points to the new node
- `SearchResult` gains four new fields: `created_at`, `learned_at`, `session_id`, `expired_at` (all default `None`, backward compatible)
- `search()` and `explain()` gain `time_before`, `time_after`, `include_expired`, `diverse`, and `grouped` keyword parameters
- `get_all()` gains `include_expired` keyword parameter
- `_get_memories_with_timestamps()` now excludes expired memories (affects `summarize()`)
- History module prefers engine CDC when available, reducing graph clutter from `:History` nodes
- 387 tests passing, 78% coverage

## [0.1.5] - 2026-03-14

Guardrails, observability, and search quality: config validation, database introspection, search pipeline tracing, result provenance, score filtering, agreement bonus, threshold semantics fix, and concurrency safety.

### Added

- **`MemoryConfig.__post_init__` validation**: rejects invalid config at construction. Range checks on all weights, thresholds, and dimensions. Warns when importance weights do not sum to ~1.0
- **`manager.stats()`**: returns a `MemoryStats` dataclass with memory counts (total, by type), entity count, relation count, and database info. No LLM calls, pure database reads. Available on both `MemoryManager` and `AsyncMemoryManager`
- **`manager.explain(query)`**: runs a search and returns an `ExplainResult` with a step-by-step pipeline trace (embedding, hybrid search, entity extraction, graph search, merge, optional boosts). Helps users understand why results rank the way they do
- **Search result provenance**: `SearchResult.source` field tracks where each result came from: `"vector"`, `"graph"`, `"both"`, or `None` (for `get_all()`). Preserved through scoring, reranking, and topology boost stages
- **Minimum score filtering**: `search_min_score` config option (default 0.0) and `min_score` parameter on `search()` filter out low-quality results. Per-call parameter overrides the config default
- **Agreement bonus**: `agreement_bonus` config option (default 0.1) gives a 10% score boost when both vector and graph search find the same memory, rewarding cross-source agreement
- **Embedding dimension validation**: `_create_memory()` now raises `ValueError` when embedding dimensions don't match the vector index, catching model mismatches at write time instead of producing silent search corruption
- **Per-user locking**: `asyncio.Lock` per user_id serializes the search-reconcile-store critical section in `add()`, preventing race conditions from concurrent calls
- **CLI subcommands**: `stats`, `explain`, `--min-score` on search
- **MCP tools**: `memory_stats`, `memory_explain_search`, `min_score` parameter on `memory_search`
- New types: `MemoryStats`, `ExplainStep`, `ExplainResult`
- 60+ new tests (config validation, stats, explain, search provenance, min-score filtering, agreement bonus, threshold semantics, dimension validation, locking)

### Fixed

- **Silent exception in `scoring.py`**: bare `pass` on access stats update replaced with `logger.debug()` (importance scoring no longer silently loses access counts)
- **Silent exception in `history.py`**: `get_history()` query failures now logged with `logger.warning()` instead of returning empty list silently
- **Entity duplication in `manager.py`**: narrowed try-block in `_find_or_create_entity` so property access errors propagate instead of creating duplicate Entity nodes
- **DERIVED_FROM edge logging**: upgraded from `debug` to `warning` level (lineage loss is now visible)
- **Usage callback error context**: log message now includes the callback function name for easier debugging

### Changed

- **BREAKING: `similarity_threshold` renamed to `reconciliation_threshold`** with corrected semantics: now means minimum similarity (0.0-1.0), not maximum distance. Default changed from 0.7 to 0.3 (same effective behavior: old `distance <= 0.7` equals `similarity >= 0.3`)
- Search merge now uses agreement-aware dedup instead of naive max-score dedup. The `explain()` trace includes `agreement_count` in the merge step
- `search_similar()` threshold parameter now uses similarity semantics (higher = stricter) instead of distance semantics
- 330 tests passing, 79% coverage

## [0.1.4] - 2026-02-28

Episodic memory, built-in MCP server, OpenTelemetry tracing, and topology-aware consolidation.

### Added

- **Episodic memory type**: new `memory_type="episodic"` for interaction events and reasoning context (e.g. "user asked X, found Y"). Dedicated extraction prompts, filterable in `search()` and `get_all()`
- **Built-in MCP server** (`grafeo-memory-mcp`): 9 tools, 2 resources, 2 prompts exposing the high-level memory API to AI agents (Claude Desktop, Cursor, etc.). Install with `uv add grafeo-memory[mcp]`
- **OpenTelemetry instrumentation** (opt-in): set `instrument=True` in `MemoryConfig` to trace all LLM calls via pydantic-ai's `Agent.instrument_all()`. Supports custom `InstrumentationSettings` for tracer provider and content filtering
- **Topology-aware consolidation**: `consolidation_protect_threshold` config option prevents `summarize()` from merging well-connected hub memories. Memories with topology scores above the threshold are preserved
- New `MemoryConfig` options: `consolidation_protect_threshold`, `instrument`
- New export: `InstrumentationSettings` from `grafeo_memory`
- MCP tools: `memory_add`, `memory_add_batch`, `memory_search`, `memory_update`, `memory_delete`, `memory_delete_all`, `memory_list`, `memory_summarize`, `memory_history`
- MCP resources: `memory://config`, `memory://stats`
- MCP prompts: `manage_memories`, `knowledge_capture`
- 18 new tests (episodic memory, tracing, MCP tools)

### Changed

- Dropped Groq from default provider list (Mistral preferred)
- CI now installs all extras (`uv sync --all-extras`) to cover MCP tests
- All `pip install` references in docs, examples, and error messages changed to `uv add` (with pip as alternative)

## [0.1.3] - 2026-02-27

Bug fixes, provenance tracking, and topology-boosted search.

### Added

- **Provenance edges**: `summarize()` creates `DERIVED_FROM` edges linking summary memories to the originals they replaced
- **Topology boost** (opt-in): lightweight search re-ranking that promotes well-connected memories. Enable with `enable_topology_boost=True`, tune with `topology_boost_factor` (default 0.2). No LLM call, purely structural
- **Extraction fallback**: when combined extraction fails (e.g. Mistral JSON mode), automatically falls back to separate fact + entity extraction calls instead of returning empty
- New config options: `enable_topology_boost`, `topology_boost_factor`
- 8 new tests (context manager reuse, extraction fallback, DERIVED_FROM edges, topology boost)

### Fixed

- **Context manager reuse**: closing a `MemoryManager` and opening a new one in the same process no longer corrupts the async event loop. The `asyncio.Runner` is now process-scoped and cleaned up via `atexit`
- **Combined extraction traceback noise**: downgraded from `logger.warning` to `logger.debug` since the fallback is handled gracefully
- **`history()` return type**: now returns `list[HistoryEntry]` instead of `list[dict]`, matching the exported type
- **Reconciliation logging**: fast-path ADD (no existing memories found) now logs at debug level for easier diagnosis

### Changed

- `MemoryManager.close()` no longer calls `shutdown()` on the async runner
- `history()` return type: `list[dict]` -> `list[HistoryEntry]` (breaking if code accessed dict keys)
- CLI `history` command updated for `HistoryEntry` attribute access
- README API reference rewritten with correct return types and iteration examples

## [0.1.2] - 2026-02-27

Performance and quality release: fewer LLM calls per operation, smarter memory extraction, and topology-aware scoring.

### Added

- **Combined extraction**: fact + entity + relation extraction now runs in a single LLM call instead of two sequential calls, saving ~1 LLM call per `add()`. New `ExtractionOutput` schema and `COMBINED_EXTRACTION_SYSTEM` / `COMBINED_PROCEDURAL_EXTRACTION_SYSTEM` prompts
- **Topology-aware scoring** (opt-in): graph-connectivity score based on entity sharing between memories. Enable with `weight_topology > 0` in `MemoryConfig`. Inspired by VimRAG
- **Structural decay modulation** (opt-in): foundational memories (those reinforced by newer related memories) resist temporal decay. Enable with `enable_structural_decay=True` and tune `structural_feedback_gamma`. Inspired by VimRAG Eq. 7
- **New `MemoryConfig` options**: `weight_topology` (default 0.0), `enable_structural_decay` (default False), `structural_feedback_gamma` (default 0.3)
- 20 new topology scoring tests (`test_topology_scoring.py`)
- 2 new extraction coverage tests (combined extraction error path, vector_search embedding fallback)

### Improved

- **1 fewer embedding call per search**: query embedding is now computed once in `_search()` and shared across both `hybrid_search()` and `graph_search()`, via new `query_embedding` parameter on both functions
- **Fact grouping prompt**: extraction prompts now instruct the LLM to group closely related details into a single fact (e.g., "marcus plays guitar, is learning jazz, and focuses on Wes Montgomery's style" instead of three separate facts), producing fewer but richer memories
- **Reconciliation temporal reasoning**: reconciliation prompt now includes explicit guidance for temporal updates ("now works at X" → UPDATE), state changes ("car is fixed" → UPDATE "car is broken"), and accumulative facts ("also likes sushi" → ADD alongside "likes pizza")
- **Type annotations**: `run_sync()` now uses generic `[T]` syntax with proper `Coroutine` typing instead of `object -> object`
- **Windows safety net**: `_ProactorBasePipeTransport.__del__` monkey-patch uses `contextlib.suppress(RuntimeError)` instead of bare try/except

### Fixed

- **Search deadlock on Windows**: `_search()` no longer triggers nested `run_sync()` calls. Entity extraction for graph search is now performed asynchronously within the already-running event loop, then passed to `graph_search()` via the new `_entities` parameter. This fixes the `RuntimeError: Event loop is closed` / hang when calling `search()` on Windows with Python 3.13+
- **`graph_search()` nested `run_sync()`**: accepts pre-extracted `_entities` to avoid calling `extract_entities()` (which internally calls `run_sync()`) from within an async context

### Changed

- `extract_async()` now makes 1 LLM call (combined) instead of 2 (facts → entities). The standalone `extract_facts_async()` and `extract_entities_async()` functions remain unchanged for independent use (e.g., search query entity extraction)
- `graph_search()` signature: added `_entities` and `query_embedding` keyword-only parameters (backward compatible, both default to `None`)
- `vector_search()` and `hybrid_search()` signatures: added `query_embedding` keyword-only parameter (backward compatible, defaults to `None`)
- `compute_composite_score()` signature: added `topology` and `reinforcement` keyword-only parameters (backward compatible, both default to 0.0)
- Removed local `grafeo` path dependency from `pyproject.toml` (`[tool.uv.sources]` section)
- Configured `ty` checker: added `extra-paths = ["tests"]` and downgraded rules that produce false positives from Rust-extension deps

## [0.1.1] - 2026-02-12

### Fixed

- CI configuration and failing tests
- Type checking fixes for `ty`

### Changed

- Documentation pass on README
- Lock file updates

## [0.1.0] - 2026-02-12

Initial release.

### Added

- **`MemoryManager`** (sync) and **`AsyncMemoryManager`** (async): full memory CRUD with `add()`, `search()`, `update()`, `delete()`, `get_all()`, `history()`
- **LLM-driven extraction pipeline**: fact extraction, entity/relation extraction via pydantic-ai structured output
- **LLM-driven reconciliation**: ADD / UPDATE / DELETE / NONE decisions for new facts against existing memories, plus relation reconciliation for graph edges
- **Hybrid search**: BM25 + vector similarity with RRF fusion, falling back to vector-only when hybrid is unavailable
- **Graph search**: entity extraction from queries, graph traversal via HAS_ENTITY edges, cosine similarity scoring
- **Importance scoring** (opt-in): composite scoring with configurable weights for similarity, recency, frequency, and importance
- **Memory summarization**: LLM-driven consolidation of old memories into fewer, richer entries
- **Procedural memory**: separate memory type for instructions, preferences, and behavioral rules with dedicated extraction prompts
- **Vision / multimodal** (opt-in): describe-first approach for image inputs via LLM vision
- **Actor tracking**: optional `actor_id` and `role` on messages for multi-agent scenarios
- **Scoping**: `user_id`, `agent_id`, `run_id` for multi-tenant memory isolation
- **Usage tracking** (opt-in): per-step LLM usage callbacks via `usage_callback`
- **CLI**: `grafeo-memory add`, `search`, `list`, `update`, `delete`, `history`, `summarize` with JSON output mode
- **Graph-native history**: HAS_HISTORY edges tracking all memory mutations with actor and timestamp
- **Windows async compatibility**: persistent `asyncio.Runner` and `ProactorEventLoop` safety net for Python 3.13+
- 230 tests, 83% coverage

[Unreleased]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/GrafeoDB/grafeo-memory/releases/tag/v0.1.0
