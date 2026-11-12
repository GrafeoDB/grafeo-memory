# Changelog

All notable changes to grafeo-memory, an embedded AI memory layer on GrafeoDB.

## [0.3.0] - 2026-11-12

Engine and architecture release: grafeo-memory runs on grafeo 0.6.2, its modules move into layers, and each reconciliation stage can run on the LLM, a decision model, or both. Before upgrading: the first open of a grafeo 0.5 database migrates it one way (the old copy stays as `<path>.pre-0.6`), every `db_path` is a single file, and pre-0.3 module paths still import but warn until 0.7. Every new toggle and layer is off by default, which keeps the 0.2 pipeline; the release also carries 0.2.3, which never shipped on its own.

### Added

- **Decision engines per pipeline stage**: `MemoryConfig.pipeline` (`PipelineConfig`, `StageConfig`) runs the `memories`, `relations` and `search` stages on `Engine.LLM` (default), `Engine.DECISION` or `Engine.HYBRID`, with the model named by `MemoryConfig.decision_model` (`"jev"`, `"fake"` or an entry point in `grafeo_memory.decision_models`). `HYBRID` asks the decision model first and sends the facts (for relations, the KEEP/DELETE verdicts) below `confidence_threshold` to one LLM call; the extraction stages stay on the LLM, and another engine there raises `ValueError`.
- **Jev decision model** (extra `jev`): `JevDecisionModel` calls TypeSafe's hosted System One API with `TYPESAFE_API_KEY`. It sends the reconciliation state (new facts and their candidate memories) to TypeSafe; the default pipeline sends nothing there.
- **Decision-model protocol**: `DecisionModel`, `CategoricalQuestion`, `ScoreQuestion`, `Decision`, `register_decision_model()`, `get_decision_model()` and the offline `FakeDecisionModel` are exported, with per-fact and pairwise question packs (`Granularity`) and three escalation signals (`EscalationSignal`).
- **Temporal pre-check** (`PipelineConfig.temporal_precheck`): a new fact whose validity interval overlaps no candidate memory is added without a model call, and the other facts reconcile against their overlapping candidates only. Only stated validity counts ("since 2019" has no end, "until 2022", "2020-2022", "in 2024", "on 7 May 2023", ISO dates); a fact without one, or a bare number such as "2048 followers", always reaches the engine.
- **Ontology normalization** (`PipelineConfig.ontology`): before graph writes, extracted entity names resolve to existing entities of the same type by exact label or alias (relations follow the rename), and a relation of a controlled type whose endpoint types break its domain or range is not written. Types match case-insensitively (`person`, `works at`). Free-text relation types and unknown entity types pass unchanged.
- **`compile()`** on `MemoryManager` and `AsyncMemoryManager` (also `acompile()`): an offline pass that merges entities of the same type and `graph_name` scope whose names embed at cosine 0.92 or more into the older node, moving their edges with their properties and keeping the merged name in `merged_from` (so `ontology` resolves it later), and reports relations that break the controlled vocabulary. It makes no LLM call, and the manager never runs it on its own.
- **Layer toggles on `PipelineConfig`**: `bitemporal` and `episodes` mirror `MemoryConfig.enable_bitemporal` and `enable_episodes`, `compile` is read by the benchmark harnesses, and `taxonomy`, `policy` and `bucket` switch on the layers below.
- **Open-decisions bucket** (`MemoryConfig.bucket`, `BucketConfig`): a decision-model answer below `floor` (default 0.5) is parked instead of applied or escalated. The fact is stored with `provisional=True`, the contested memory stays, and `list_open_decisions()` / `resolve_open_decision(id, answer)` let a person settle it; a later write touching the evidence, or `compile()`, re-judges it, and entries past `ttl_days` or beyond `max_open` expire to `unknown`.
- **Policy rule packs** (`MemoryConfig.policy`, `PolicyConfig`): `compliance` keeps credentials and card, IBAN and SSN numbers out of the store, `temporal` turns on the pre-check, `entrenchment` stops inferred decisions from overwriting certified or user-stated memories, and `lifecycle` certifies a memory after `compilation.promote_after` re-statements and applies `retention_days`. Each event carries the rung that decided it (`MemoryEvent.rung`), and `stats().layers` reports rule hits and the deterministic share.
- **Subject and instance scopes** (`MemoryConfig.taxonomy`, `TaxonomyConfig`): new memories get subject tags from ten designed roots, entities linked to enough memories become scopes, and searches boost (`mode="soft"`) or keep only (`"filter"`) the memories in the scopes a query routes to. `search(subjects=, scopes=)` and `add(scopes=)` pin scopes, `hard_roots` wall a root off, `scopes()` and `set_root_scheme()` show and edit the roots (retiring one never deletes memories), and `compile()` grows generated children.
- **Retrieval knobs** (`MemoryConfig.retrieval`, `RetrievalConfig`): `order="deterministic"` boosts and pulls in the memories of the entities a query names and their relation neighbours (`expansion_depth`), skipping hub entities such as conversation speakers that sit on more than `hub_share` of the memories, `temporal_filter` favors memories whose stated interval overlaps the query's, and `certified_boost` ranks certified memories higher.
- **Ontology knobs** (`MemoryConfig.ontology`, `OntologyConfig`): write-time embedding match of entity names, one decision-model question for the ambiguous rest (`rung4`), and `interval_closing`, which ends a superseded fact's validity at the new fact's stated start.
- **Working, procedural and episodic tiers** (`MemoryConfig.tiers`, `TiersConfig`): `assemble(query, session_id, budget)` returns a `ContextBlock` with the session window, instructions, facts and events within a token budget, each line with its provenance; `end_session()` promotes a session (with `defer_extraction`, `add()` only buffers and extraction runs once per session); `add_instruction()` stores certified procedural memories and `auto_triggers` picks up instruction turns.
- **Certification and compile**: `certify(memory_id)` marks a memory certified; `compile()` also returns merge proposals (cosine 0.85 to 0.92, open decisions when the bucket is on), and runs every `compilation.every_n_adds` adds when set.
- **MCP tools** `memory_open_decisions`, `memory_resolve_decision`, `memory_manage_roots`, `memory_assemble` and `memory_add_instruction`.
- **Store schema version**: a store that uses a layer records `schema_version` 2 and runs pending migrations on open; a store that never turns a layer on is left unchanged.
- **Benchmark overrides**: the harnesses take `--set KEY=VALUE` for any `MemoryConfig` field by dotted path (`--set bucket.floor=0.4`), and `matrix` has rounds 14 to 19, one per layer, and round 20 with the read-path knobs at `k=20`.
- **LLM call cache** (`MemoryConfig.cache`, `True` or `CacheConfig`): LLM, embedding and decision-model calls are answered from an in-memory LRU and, for a file database, from `<db_path>.llmcache.jsonl`, so an identical call reaches the provider once. Off by default; `cached_model()`, `cached_embedder()` and `MemoryCache` wrap a model or embedder directly, and `MemoryCache.clear()` removes the entries and the file.
- **Settings from the environment**: `GrafeoMemorySettings` reads `GRAFEO_MEMORY_*` variables and `.env` for the CLI and MCP server (model, database, user, cache, engine limits, decision model), and provider keys as secrets that are never logged. An empty variable counts as unset. The CLI and MCP server log to stderr at `GRAFEO_MEMORY_LOG_LEVEL` (default `WARNING`) with secret values redacted; `grafeo_memory.common.install_redaction()` adds the same filter to an application's handlers.
- **Transient LLM errors are retried**: timeouts, connection errors, 429 and 5xx from the LLM provider are retried with exponential backoff (`MemoryConfig.llm_retries`, default 3; `0` disables). A dropped connection used to fail the step silently, so entity extraction stored a fact without its entities.
- **Engine memory limits**: `MemoryConfig.db_memory_limit_bytes` and `db_spill_limit_bytes` (`GRAFEO_MEMORY_DB_MEMORY_LIMIT_MB`, `GRAFEO_MEMORY_DB_SPILL_LIMIT_MB`) are passed to grafeo when the manager opens its database.
- **Time travel** (grafeo 0.6.2 history retention): `MemoryConfig.history_retention` (`"off"`, `"forever"`, a `timedelta` or seconds) is passed when the manager opens its database, and a database file keeps it. `as_of(point)`, `execute_as_of(query, point, params)` and `get_memory_as_of(memory_id, point)` on both managers read the database, a query or a memory as it was at an epoch, an aware datetime or a `timedelta` ago (`None` when the memory did not exist yet).
- **Microsoft Agent Framework** (extra `agent-framework`): `grafeo_memory.agent_framework.GrafeoVectorStore` adapts a manager's database to the framework's `BaseVectorStore`, so its collection CRUD, search and agent tools (`create_upsert_tool`, `create_vector_search_tool`) store and find memories in grafeo. `eq` filters become grafeo's exact-match `filters=`; other operators raise `NotImplementedError`.
- **Benchmark harnesses** (extra `bench`): `python -m grafeo_memory.bench.locomo_eval`, `lme_eval`, `beam_eval` and `matrix` run LOCOMO, LongMemEval and BEAM with every engine and toggle as a flag and a shared call cache. See `docs/benchmarks.md`.
- **Python 3.15**: CI tests Python 3.13 to 3.15, and the package lists them in its classifiers.
- **Upgrade tests** (`tests/test_services/test_upgrade_from_05.py`): databases written on grafeo 0.5.44 by grafeo-memory 0.2.2 and 0.2.3, as a `.grafeo` file and as a WAL directory, open through the manager, the CLI and the MCP server, migrate once and keep every memory, embedding, history entry, entity, relation, graph metric and community. Vector search finds every memory and later writes persist.
- **Engine contract tests for grafeo 0.6**: floats stay floats, `hybrid_search(filters=...)` filters before ranking, non-finite vectors are refused, a path without extension is one file, an empty directory is refused, `detach_delete_node()` exists, and the 0.6.2 type rules hold (a string never equals a number, wrong-kind operands raise, procedure arguments are validated, lists with a null compare as unknown).
- Tests: 641 to 1071, with a regression test for each fix and end-to-end tests per layer.

### Changed

Upgrade notes first: what breaks or behaves differently from 0.2.x.

- **The first open of a grafeo 0.5 database migrates it**: a `MemoryManager`, the CLI or the MCP server converts it at the same path and keeps the old file or WAL directory, unchanged, as `<path>.pre-0.6`; grafeo 0.5 cannot open the result. Stop every process that uses the database first and keep the `.pre-0.6` copy until going back is ruled out (README, "Upgrading Grafeo (0.5 to 0.6)"); for grafeo 0.5, stay on grafeo-memory 0.2.x.
- **Every `db_path` is a database file**: a path without the `.grafeo` extension no longer creates a WAL directory, and opening a directory that is not a grafeo 0.5 database (an empty one included) raises `GrafeoError`. Point `db_path`, `--db` or `GRAFEO_MEMORY_DB` at a file such as `./memories/memory.grafeo`.
- **Embeddings with NaN or infinite values raise `ValueError`** before anything is written: grafeo 0.5 stored such a vector and vector search never found it, grafeo 0.6 refuses it. Check the embedder if `add()`, `update()` or `search()` raises "the embedder must return finite floats".
- **Modules live in layers**: the source moved from `src/grafeo_memory/` to `grafeo_memory/` with `common/`, `providers/`, `core/`, `services/` and the entry points `cli/`, `mcp/` and `agent_framework.py`; lint and a test enforce the import order. `from grafeo_memory import ...` is unchanged (see Deprecated for the old module paths).
- **Relation reconciliation sees at most 50 candidates**: the existing relations shown to the LLM are those sharing an entity with a new relation (only those can conflict), ranked same source and relation type first, then same target and type or the same two entities, then the rest, newest first within a rank. 0.2.x showed every relation of the user, a prompt that grew without bound.
- **Numbers keep their type**: scores, similarities, importance and PageRank and betweenness values stay `float`, and timestamps `int`. grafeo 0.6 reads a GQL literal such as `0.5` as an exact decimal, so grafeo-memory writes such values only as parameters or through the direct API.
- **Deletes use `db.detach_delete_node()`** when the engine has it (grafeo 0.6.0), and a `DETACH DELETE` query otherwise.
- **Checked and kept on grafeo 0.6**: embeddings are converted to floats (a list of ints is still stored as a plain list and left out of vector search), and history stays in `:History` nodes (change data capture is still off by default and kept in memory only).
- **More LOCOMO questions answered**: on conversations 0 to 2 (385 questions, Mistral Small 4 for extraction, answers and judge, `k=100`) the default pipeline scores 0.9013 against 0.8727 for 0.2.2 under the same prompts. On all 10 conversations it scores 0.8753 (`docs/benchmarks.md`).
- **README**: new sections on decision engines and pipeline toggles (including what leaves the device), the call cache and the module layout; Persistence and "Upgrading Grafeo (0.5 to 0.6)" describe what grafeo 0.6 does with files, directories and 0.5 databases.
- **CI**: the type check installs the optional extras, so optional imports resolve; coverage leaves out the benchmark harnesses (they need datasets and API keys). `actions/checkout@v7`, `astral-sh/setup-uv@v10.3.0` (setup-uv no longer publishes major tags), `actions/upload-artifact@v7`, `actions/download-artifact@v8`, `codecov/codecov-action@v7`; `ty check` runs with the configured excludes.

### Deprecated

- **Pre-0.3 module paths**: `grafeo_memory.manager`, `.types`, `.schemas`, `.embedding`, `.history`, `.messages`, `.prompts`, `.protocol`, `.reranker`, `.scoring`, `.vision`, `.communities`, `.temporal`, `.extraction.*`, `.reconciliation.*`, `.search.*`, `._compat`, `._graph` and `._tracing` resolve to the modules at their new homes and raise `DeprecationWarning` naming the new path. They are removed in 0.7.

### Removed

- **Python 3.12**: grafeo-memory 0.3.0 requires Python 3.13 or later (`requires-python = ">=3.13"`). On Python 3.12, stay on 0.2.x.

### Fixed

- **Searches in a shared database could miss a user's memories**: the default hybrid search let the engine pick the best `k` memories of all users and filtered by `user_id`, `agent_id`, `run_id`, `graph_name` and `filters` afterwards, so other users' memories could take every slot. The scope filters now go to grafeo 0.6's `hybrid_search(filters=...)`, which filters before it ranks.
- **A refused embedding left a memory without one**: the memory node was created before its embedding was checked and set, so a wrong-sized or non-finite embedding left a memory that vector search never finds, and `update()` changed the text before re-embedding it. The embedding is now checked before anything is written.
- **Mistral embeddings with mistralai 2 and later**: the CLI and the MCP server imported `from mistralai import Mistral`, which mistralai 2 moved to `mistralai.client`, and exited with "mistralai package not installed". `grafeo_memory.providers.embedding.mistral_client_class()` finds the client in either layout; the README and examples import it from `mistralai.client`.
- **Type of `search()`**: overloads on `grouped` give `SearchResponse` (and `dict[str, list[SearchResult]]` with `grouped=True`) instead of a union, so type checkers accept iterating the results.

### Dependencies

- **grafeo 0.6.2** (`grafeo>=0.6.2,<0.7`; 0.2.3 required `>=0.5.44,<0.6`).
- **pydantic-ai-slim 1.96 or later** (`>=1.96`, was unversioned): grafeo-memory reads `AgentRunResult.usage` as a property, which 1.96 introduced and pydantic-ai 2 requires. pydantic-ai 2 is supported; the lock has 2.66.
- **pydantic-settings and python-dotenv** are new runtime dependencies (`pydantic-settings>=2.6`, `python-dotenv>=1.0`), for the settings above.
- **mcp 2** for the `mcp` extra (`mcp>=2.2,<3`; was `>=1.26,<2`): the server uses the mcp 2.x API (`mcp.server.mcpserver.MCPServer`, `Context[AppContext, ServerSession]`).
- **mistralai 2 or later** for the `mistral`, `bench` and `all` extras (`mistralai>=2`): pydantic-ai's Mistral support already required 3.1, so 1.x never resolved; the client is imported from `mistralai.client`.
- **New extras**: `jev` (`httpx>=0.27`), `agent-framework` (`agent-framework-core>=1.25,<2`) and `bench` (`mistralai`, `nltk`, `numpy`, `pyarrow`).
- **Lockfile**: openai 3.49, anthropic 2.7, mistralai 3.7, google-genai 3.0, groq 1.7; development tools pytest 9.2, ruff 0.17 and ty 0.0.100 (its renamed rules `import-not-found` and `missing-attribute` are used in `pyproject.toml`).

## [0.2.3] - Unreleased (ships in 0.3.0)

Works with grafeo 0.5.44: deletes work again, memory history is stored durably, and an engine contract suite runs weekly against the newest grafeo.

### Added

- **Engine contract tests** (`tests/test_core/test_grafeo_contract.py`, 37 tests): every grafeo behavior grafeo-memory relies on (direct API calls and their errors, query shapes and result format, `DETACH DELETE`, vector, text, hybrid and MMR search, PageRank, betweenness and Louvain output, single-file and directory persistence, CDC defaults). A failure names the behavior that changed and the code that depends on it.
- **Weekly engine compatibility workflow** (`.github/workflows/engine-compat.yml`, also manual): runs the contract tests and the full suite on Linux and Windows against the newest grafeo the requirement accepts and the newest grafeo on PyPI, pre-releases included.
- Tests for delete cascades through every edge type, history on every write path and across close and reopen (single-file and WAL-directory databases), shared databases with and without CDC, the async manager, the MCP tools and server lifespan against a real manager, and the CLI against a database file; regression tests for each fix above (540 to 641 tests).

### Changed

- **History storage**: `:History` nodes carry `memory_id`, `event`, `timestamp` (epoch ms), the texts, `actor_id`, `role`, `user_id` and `graph_name`, and have no edges, so they do not change graph algorithm scores. `record_history()` always writes and returns the node id; new `record_history_batch()` (used by `add_batch(infer=False)`) and `adopt_legacy_history()`. `get_history()` gains a `graph_name` filter, still reads the `HAS_HISTORY` entries of grafeo-memory 0.1.x, and raises database errors instead of returning an empty list. Engine CDC is not used, and the CDC setting of a database passed in with `db=` is left alone.
- `history()` is scoped to the manager's `graph_name`.
- `delete()` records a DELETE history entry; `summarize()` records it through the same path.
- `update()` raises `ValueError` when `memory_id` is not a memory of the manager (it used to return an UPDATE event and change nothing). The MCP `memory_update` tool returns the error, the CLI exits with status 1.
- `memory_id` joins the property indexes created at startup.
- README, docstrings and `.env.example` use single-file `.grafeo` database paths; README has new Persistence (including upgrading to grafeo 0.6) and Memory History sections.
- Development dependencies moved from the deprecated `[tool.uv] dev-dependencies` to `[dependency-groups]`.

### Fixed

- **Deletes failed on grafeo 0.5.44**: the engine now refuses to delete a node that still has edges, and memories almost always have some (`HAS_ENTITY`, `LEADS_TO`, `SUPERSEDES`, `DERIVED_FROM`, `PRODUCED`). `delete()`, `delete_all()` and `summarize()` raised `GrafeoError`, and dissolved communities were silently kept. Nodes are now removed with `DETACH DELETE` (new helper `_graph.detach_delete_node()`); entities, relations and episodes stay.
- **`history()` was always empty on grafeo 0.5.32 and later**: history was read from engine change data capture, which became opt-in in 0.5.32 and is kept in memory only (lost on close, pruned by retention, with hybrid logical clock timestamps instead of epoch milliseconds), while `record_history()` wrote nothing whenever the engine had CDC. grafeo-memory now stores every change as a `:History` node in the same database (see Changed). History survives closing and reopening the database and a hard delete of the memory.
- **Search results lost fields**: the vector/graph merge, `enable_topology_boost`, `enable_importance`, the cross-session boost and `LLMReranker` rebuilt `SearchResult` by hand and dropped `created_at`, `session_id`, `valid_at`, `invalid_at`, `learned_at` (and the reranker also `memory_type`). Results found by both vector and graph search then failed `time_after`/`time_before`, ignored `point_in_time`, sorted wrongly for temporal queries, and `grouped=True` put them under `"default"`.
- **Embeddings that were not float lists were never found**: the engine indexes a property as a vector only when every element is a float, so an embedder returning ints stored memories that vector search skipped. Embeddings are now converted to floats before they are stored or searched.
- **Reconciliation acted on any node id the LLM returned**: an UPDATE or DELETE whose `target_memory_id` was not one of the candidates shown to the LLM (hallucinated, injected, another user's memory or a foreign node in a shared database) expired that node. Such decisions now lose their target: UPDATE becomes ADD and DELETE is skipped, with a warning.
- **`delete()`, `update()` and `set_importance()` accepted any node id**: in a shared database they could delete or change foreign nodes, or memories of another `graph_name`. They now act only on `:Memory` nodes in the manager's scope.
- **`delete_all()` left expired memories behind**: memories superseded by reconciliation (and their text) stayed in the database. They are now deleted too.
- **Session and episode chains forked within one millisecond**: `LEADS_TO` and `NEXT_EPISODE` picked the previous node by `created_at` alone, so quick successive adds all linked to the same node. Ties are now broken by node id.

### Dependencies

- **grafeo 0.5.44** (`grafeo>=0.5.44,<0.6`; was `>=0.5.27`): grafeo 0.5.44 changed the direct-write API (writes to missing nodes and deletes of connected nodes raise). The upper bound keeps this release off grafeo 0.6 and its one-way file migration; grafeo-memory 0.3.0 moves to grafeo 0.6, see the [versioning policy](https://grafeo.dev/versioning/).

## [0.2.2] - 2026-04-08

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
- **`graph_name` scoping**: optional `graph_name` on `MemoryConfig` isolates memories and entities per subgraph when multiple callers share a database via `db=`. Stamped on all Memory and Entity creates, filtered on all read paths (`_build_filters`, `graph_search`, `temporal_chain`, session linking, `stats`). No behavior change when unset ([#23](https://github.com/GrafeoDB/grafeo-memory/issues/23), [#24](https://github.com/GrafeoDB/grafeo-memory/pull/24), thanks [@Michaelzag](https://github.com/Michaelzag))
- `graph_name` added to property index list for fast scoped lookups
- 15 new tests covering `graph_name` write stamping, read isolation, entity separation, stats scoping, temporal chain boundaries, and backward compatibility

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

---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[0.3.0]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/GrafeoDB/grafeo-memory/releases/tag/v0.1.0
