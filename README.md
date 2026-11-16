<picture>
  <source media="(max-width: 556px) and (prefers-color-scheme: dark), (min-width: 768px) and (max-width: 844px) and (prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GrafeoDB/grafeo-memory/main/.github/readme/header-narrow-dark.svg">
  <source media="(max-width: 556px), (min-width: 768px) and (max-width: 844px)" srcset="https://raw.githubusercontent.com/GrafeoDB/grafeo-memory/main/.github/readme/header-narrow-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GrafeoDB/grafeo-memory/main/.github/readme/header-dark.svg">
  <img alt="grafeo-memory: Memory for agents, in one file: a graph with vectors and history, with any LLM." src="https://raw.githubusercontent.com/GrafeoDB/grafeo-memory/main/.github/readme/header-light.svg" width="100%">
</picture>

[![CI](https://github.com/GrafeoDB/grafeo-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/GrafeoDB/grafeo-memory/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GrafeoDB/grafeo-memory/graph/badge.svg)](https://codecov.io/gh/GrafeoDB/grafeo-memory)
[![PyPI](https://img.shields.io/pypi/v/grafeo-memory.svg)](https://pypi.org/project/grafeo-memory/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/GrafeoDB/grafeo-memory/blob/main/LICENCE)

AI memory layer powered by [GrafeoDB](https://github.com/GrafeoDB/grafeo), an embedded graph database with native vector search.

No servers, no Docker: one `.grafeo` file and one LLM (and an optional decision model).

## Install

```bash
uv add grafeo-memory                     # base (bring your own LLM + embedder)
uv add "grafeo-memory[mistral]"          # + Mistral models and embeddings
uv add "grafeo-memory[anthropic]"        # + Anthropic models
uv add "grafeo-memory[openai]"           # + OpenAI models and embeddings, also local models through Ollama
uv add "grafeo-memory[mcp,mistral]"      # + MCP server for AI agents (with the provider extra of its model)
uv add "grafeo-memory[jev]"              # + TypeSafe Jev decision model (hosted, opt-in)
uv add "grafeo-memory[agent-framework]"  # + Microsoft Agent Framework vector store
uv add "grafeo-memory[bench]"            # + LOCOMO / LongMemEval / BEAM harnesses
uv add "grafeo-memory[all]"              # all LLM providers
```

## Quick Start

Set `MISTRAL_API_KEY` first. Richard has just left Hooli to start Pied Piper:

```python
from mistralai.client import Mistral
from grafeo_memory import MemoryManager, MemoryConfig, MistralEmbedder

embedder = MistralEmbedder(Mistral())
config = MemoryConfig(db_path="./pied-piper.grafeo", user_id="richard")

with MemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as memory:
    for event in memory.add(
        "I just left Hooli to start Pied Piper, a compression company built on middle-out. "
        "We work out of the Hacker Hostel in Palo Alto",
        speaker="Richard",
    ):
        print(event.action.value, event.text)
    # for example:
    # add Richard left Hooli to start Pied Piper, a compression company built on middle-out.
    # add Pied Piper operates out of the Hacker Hostel in Palo Alto.

    memory.add("We moved Pied Piper out of the Hacker Hostel into our own office in Mountain View", speaker="Richard")
    # for example: update Pied Piper operates out of their own office in Mountain View.

    for result in memory.search("Where is Pied Piper based?", k=2):
        print(round(result.score, 2), result.text)
    # for example:
    # 0.96 Pied Piper operates out of their own office in Mountain View.
    # 0.85 Richard left Hooli to start Pied Piper, a compression company built on middle-out.
```

The exact wording depends on the model. `user_id` partitions the store and never reaches the model; `speaker`
names who is talking (without it, facts say "the user"), and `source="notes"` marks notes or documents, which have
no speaker. Other models, a local one included, are in [Supported LLM Providers](#supported-llm-providers).

## How It Works

grafeo-memory implements the **reconciliation loop**, the intelligence layer that decides what to remember:

1. **Extract** facts from conversation text (LLM call)
2. **Extract** entities and relationships (LLM tool call)
3. **Search** existing memory for related facts (vector + graph)
4. **Reconcile** new facts against existing memory (LLM decides ADD/UPDATE/DELETE/NONE)
5. **Execute** the decisions against GrafeoDB

```
┌──────────────────────────────────────────┐
│             grafeo-memory                │
│                                          │
│  Extractor -> Reconciler -> Executor     │
│  (LLM)       (LLM)        (GrafeoDB)     │
└──────────────────┬───────────────────────┘
                   │
         ┌─────────┴──────────┐
         │      GrafeoDB      │
         │  Graph + Vector    │
         │  + Text (optional) │
         │ single .grafeo file│
         └────────────────────┘
```

## Documentation

- [docs/architecture.md](https://github.com/GrafeoDB/grafeo-memory/blob/main/docs/architecture.md): the package layers and the import rules between them
- [docs/benchmarks.md](https://github.com/GrafeoDB/grafeo-memory/blob/main/docs/benchmarks.md): LOCOMO, LongMemEval and BEAM harnesses, matrix rounds, test gates

The modules live in layers (`grafeo_memory.common`, `.providers`, `.core`, `.services`); import the public API
with `from grafeo_memory import ...`. Older module paths such as `grafeo_memory.manager` or
`grafeo_memory.search.vector` still import, with a `DeprecationWarning` naming the new path, until 0.7.

## Multi-User Isolation

```python
config = MemoryConfig(db_path="./pied-piper.grafeo")

with MemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as memory:
    # Each user's memories are isolated
    built = memory.add(
        "I built Son of Anton, an AI that fixes bugs by deleting the code", user_id="gilfoyle", speaker="Gilfoyle"
    )
    memory.add("Gilfoyle is my rival", user_id="dinesh", speaker="Dinesh")

    memory.search("Son of Anton", user_id="gilfoyle")
    # -> Gilfoyle built Son of Anton, an AI designed to fix bugs by deleting code.

    memory.search("Son of Anton", user_id="dinesh")
    # -> Dinesh has a rival named Gilfoyle. (Dinesh's own memories only)
```

`user_id` on each call is a partition, not a permission: the unbound manager serves every user, and a call by
id (`update`, `delete`, `history`) reaches any memory in its scope. A server that acts for one user at a time
should use a view:

```python
dinesh = memory.for_user("dinesh")  # shares the database and caches
dinesh.add("I built PiperChat, a video chat app", speaker="Dinesh")
dinesh.delete(built[0].memory_id)  # False: Gilfoyle's memory is out of reach
dinesh.search("Son of Anton", user_id="gilfoyle")  # ValueError: the view is bound to dinesh
```

grafeo-memory does not authenticate anyone: the host application verifies who the user is and passes that id.
The CLI and the MCP server act through such a view for their `user_id` (default `GRAFEO_MEMORY_USER`).

### Sessions

```python
sid = memory.start_session(anonymous=True)  # a visitor: turns stay in memory only
memory.add("I want to put my money into a compression company", session_id=sid)
context = memory.assemble("compression", sid)  # the session window, nothing long-term

russ = memory.for_user("russ")
russ.start_session(sid)  # Russ signs in: the session joins russ, one way, and its turns are stored
russ.end_session(sid)  # closes the session
```

An anonymous session never writes long-term memory; `end_session()` discards it, and it expires
`tiers.working.session_ttl` seconds (default 3600) after its last turn. Binding stores its turns for the user once:
when it binds, or at `end_session()` with `tiers.working.defer_extraction`. Sessions live in the process, not in the
database file.

## Persistence

```python
config = MemoryConfig(db_path="./memory.grafeo")  # a database file
config = MemoryConfig()  # in memory, gone when the process ends
```

- A `db_path` is always a single database file, whatever its extension. While it is open, Grafeo keeps its
  write-ahead log in a `memory.grafeo.wal/` directory next to it, and removes it at close; a `memory.grafeo.spill/`
  directory may appear when a query spills to disk. The directory that holds the file must be writable.
- A `db_path` must not be an existing directory: Grafeo 0.6 refuses to open one (an empty one included) unless it is
  a Grafeo 0.5 database (see below). Name a file inside it instead, such as `./memories/memory.grafeo`.
- One process at a time can open a database for writing. Close your app, the CLI or the MCP server before another
  one opens the same path.
- Indexes are (re)created whenever a manager opens a database.

### Upgrading Grafeo (0.5 to 0.6)

grafeo-memory 0.3 and later require Grafeo 0.6.2 or later (`grafeo>=0.6.2,<0.7`); for Grafeo 0.5, stay on grafeo-memory 0.2.x.
Grafeo 0.6 writes a new database format, and the switch is one way:

1. **Before the first open with 0.6**, stop every process that uses the database (your app, `grafeo-memory`,
   `grafeo-memory-mcp`). The migration needs free disk space for a copy of the database.
2. **The first open with Grafeo 0.6** (any `MemoryManager`, the CLI or the MCP server; they all open the database
   for writing) migrates it, at the same path:
   - a database file written by 0.5.x (`memory.grafeo`) is rewritten in the new format, and the old file is kept
     next to it, unchanged, as `memory.grafeo.pre-0.6` (with `memory.grafeo.pre-0.6.wal/` if it had a
     write-ahead log left);
   - a 0.5.x WAL-directory database (a `db_path` without `.grafeo`, such as `./memory.db/`) becomes a database
     file at that path, and the old directory is kept, unchanged, as `memory.db.pre-0.6/`. Its commits are replayed
     into the file. Grafeo 0.5 did not log index definitions in that directory; grafeo-memory creates its indexes
     again on open.
3. **Later opens** find a 0.6 file: nothing migrates again and the `.pre-0.6` copy is left alone. Keep it until you
   are sure you will not go back.
4. **Grafeo 0.5 cannot open the migrated database.** To go back, stop the 0.6 processes, move the migrated file
   away, rename the `.pre-0.6` file (or directory) to the original name and pin both packages back:
   `uv add "grafeo-memory<0.3" "grafeo<0.6"` (grafeo-memory 0.2.2 alone would resolve Grafeo 0.6). Changes made
   after the migration are only in the migrated file.
5. Grafeo 0.7 will not read 0.5.x databases, so open each one once with 0.6 before upgrading further.

Your memories, embeddings, entities, relations, communities and history are ordinary graph data, so they move with
the database: grafeo-memory needs no migration step of its own, and vector search works on the migrated database
right away. Grafeo's own guide is [Upgrading to 0.6](https://docs.grafeo.dev/get-started/upgrading/0.6/);
release notes are in [CHANGELOG.md](https://github.com/GrafeoDB/grafeo-memory/blob/main/CHANGELOG.md).

## Memory History

`history(memory_id)` returns every change grafeo-memory made to a memory (ADD, UPDATE, DELETE), oldest first:

```python
mid = memory.add("Richard indents with tabs", infer=False)[0].memory_id
memory.update(mid, "Richard indents with tabs, never spaces")
memory.delete(mid)

for entry in memory.history(mid):
    print(entry.event, entry.old_text, entry.new_text, entry.timestamp)
# ADD None Richard indents with tabs ...
# UPDATE Richard indents with tabs Richard indents with tabs, never spaces ...
# DELETE Richard indents with tabs, never spaces None ...
```

- History is stored as `:History` nodes in the same database, keyed by the memory id and not connected by edges,
  so it survives closing and reopening the database and a `delete()` of the memory, and it does not change graph
  algorithm scores.
- grafeo-memory does not need Grafeo's change data capture (CDC is off by default and kept in memory only). A
  database you pass in with `db=` keeps its CDC setting.
- Only changes made through grafeo-memory are history; writes made directly with the Grafeo API are not.
- `delete()` and `delete_all()` keep the history as an audit trail. Entries carry the `user_id`, so with the
  database at hand (pass it in with `db=`) you can remove a user's history as well:

```python
import grafeo

db = grafeo.GrafeoDB("./pied-piper.grafeo")
memory = MemoryManager("mistral:mistral-small-latest", config, embedder=embedder, db=db)

memory.delete_all(user_id="richard")
db.execute("MATCH (h:History) WHERE h.user_id = $uid DELETE h", {"uid": "richard"})
```

## Supported LLM Providers

grafeo-memory uses [pydantic-ai](https://ai.pydantic.dev) model strings, so any provider pydantic-ai supports works out of the box:

```python
# Mistral: MistralEmbedder for embeddings
MemoryManager("mistral:mistral-small-latest", config, embedder=MistralEmbedder(Mistral()))

# A local model through Ollama (the [openai] extra; OLLAMA_BASE_URL=http://localhost:11434/v1)
MemoryManager("ollama:ministral-3:3b", config, embedder=embedder)

# Anthropic: pair with MistralEmbedder or a custom embedder
MemoryManager("anthropic:claude-sonnet-4-5-20250929", config, embedder=embedder)

# OpenAI: OpenAIEmbedder for embeddings
from openai import OpenAI
from grafeo_memory import OpenAIEmbedder

MemoryManager("openai:gpt-4o-mini", config, embedder=OpenAIEmbedder(OpenAI()))
```

### Built-in Embedders

| Class | Provider | Default Model | Install Extra |
| --- | --- | --- | --- |
| `MistralEmbedder` | Mistral | `mistral-embed` | `[mistral]` |
| `OpenAIEmbedder` | OpenAI | `text-embedding-3-small` | `[openai]` |

Both accept an optional `model` parameter to override the default. A local embedding model plugs in through the
`EmbeddingClient` protocol (see [Custom Embeddings](#custom-embeddings)).

### Which model works best?

| Model | Status |
| --- | --- |
| Mistral (`mistral-small-latest`) | Tested: the benchmarks and these examples run on it, one extraction call per `add()` |
| Local through Ollama (`ministral-3:3b`) | Tested with the Quick Start; a small model keeps fewer details |
| Anthropic (`claude-sonnet`) | Works; not tested in CI |
| OpenAI (`gpt-4o-mini`) | Works; tested in earlier releases |

grafeo-memory tries a single "combined" LLM call first (facts + entities + relations in one schema). If the model
cannot handle the combined schema, it falls back to separate calls automatically, which means 3 API calls per
`add()` instead of 1. The fallback is silent (logged at DEBUG level, no traceback printed).

## Custom Embeddings

Implement the `EmbeddingClient` protocol to use any embedding provider:

```python
from grafeo_memory import EmbeddingClient


class MyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        # Call your embedding API
        return [...]

    @property
    def dimensions(self) -> int:
        return 1024  # your model's output dimensions


memory = MemoryManager("mistral:mistral-small-latest", config, embedder=MyEmbedder())
```

## Decision Engines and Pipeline Toggles

Every judgment in the write path can run on the LLM (the default), on a decision model that answers typed
questions with a calibrated confidence, or on both: `hybrid` asks the decision model first and sends only the
facts it is unsure of to one LLM call. Extraction always uses the LLM.

```python
from grafeo_memory import Engine, MemoryConfig, PipelineConfig, StageConfig

config = MemoryConfig(
    db_path="./memory.grafeo",
    decision_model="jev",  # or "fake" (offline), or a provider registered under "grafeo_memory.decision_models"
    pipeline=PipelineConfig(
        memories=StageConfig(engine=Engine.HYBRID, confidence_threshold=0.75),
        relations=StageConfig(engine=Engine.HYBRID),
        temporal_precheck=True,  # a fact whose stated validity overlaps no candidate is added without a model call
        ontology=True,  # entity names resolve to existing entities; relations that break domain/range are not written
    ),
)
```

| Stage | Engines | What it decides |
| --- | --- | --- |
| `memories` | `llm`, `decision`, `hybrid` | ADD / UPDATE / DELETE / NONE for each new fact |
| `relations` | `llm`, `decision`, `hybrid` | Which existing relations a new relation invalidates |
| `search` | `llm` (no decision rerank), `decision`, `hybrid` | Rerank score blended by `rerank_blend_weight`; `hybrid` keeps the retrieval score where the model is unsure |
| `facts`, `entities`, `temporal` | `llm` only | Extraction (generates text) |

Custom providers implement `DecisionModel.decide_async(state, questions)` and register with
`register_decision_model(name, factory)`; `FakeDecisionModel` is a deterministic stand-in for tests.

**Data leaving the device.** The default pipeline sends text only to the LLM and embedder you configure. With
`decision_model="jev"` (extra `jev`, key in `TYPESAFE_API_KEY`), the reconciliation state (the new facts and the
candidate memories they are compared with) is also sent to TypeSafe's hosted API. `temporal_precheck`, `ontology`
and `compile()` run locally and send nothing.

All `PipelineConfig` toggles default to off:

| Toggle | Effect |
| --- | --- |
| `temporal_precheck` | Pure-code interval check before reconciliation |
| `ontology` | Deterministic entity normalization and relation-type checks before graph writes |
| `bitemporal`, `episodes` | Same as `MemoryConfig.enable_bitemporal` / `enable_episodes` |
| `compile` | Read by the benchmark harnesses only; applications call `memory.compile()` |
| `taxonomy`, `policy`, `bucket` | Switch on the scoping, policy and open-decisions layers below (same as their `enabled` flags) |

`memory.compile()` (also `await amemory.acompile()`) is an offline pass to run after write batches: entities of the
same type and scope whose names embed nearly identically (cosine >= 0.92) merge into the older node, which keeps
their edges, and relations that break the controlled vocabulary are reported. With the layers below on, it also
proposes near-merges, grows the subject taxonomy, applies retention and re-judges open decisions.

### Memory layers

Each layer has its own config object on `MemoryConfig`, and every one is off by default.

| Config | What it adds |
| --- | --- |
| `bucket` (`BucketConfig`) | A decision-model answer below `floor` is parked, not forced: the fact is kept as `provisional` and an open decision records the question. `list_open_decisions()` and `resolve_open_decision(id, answer)` let a person answer; a later write that touches the evidence, or `compile()`, re-judges it (`resolve_engine`); stale entries expire to an explicit unknown. |
| `policy` (`PolicyConfig`) | Rule packs that decide before any model: `compliance` (credentials and card numbers are never stored), `temporal` (the pre-check), `entrenchment` (certified and user-stated memories are not overwritten by inferred decisions), `lifecycle` (re-stated facts certify a memory, retention TTLs). Relation domain/range checks run as shapes with a severity per shape. `stats().layers` reports which rung decided (rule, decision model, LLM) and the deterministic share. |
| `taxonomy` (`TaxonomyConfig`) | Subject scopes: ten designed roots (career, health, ...) tag every new memory by keyword; prominent entities act as instance scopes. Searches route to the scopes a query names and boost (`mode="soft"`) or keep only (`"filter"`) those memories; `hard_roots` wall a root off. `search(subjects=..., scopes=...)` and `add(scopes=...)` pin scopes; `scopes()` and `set_root_scheme()` show and edit the tree; `compile()` grows generated children. |
| `retrieval` (`RetrievalConfig`) | `order="deterministic"`: memories of the entities a query names (and their relation neighbours) are boosted and pulled in before vector ranking; entities on more than `hub_share` (0.3) of a user's memories, such as the speakers, are skipped. `temporal_filter`: a query that names a time favors memories whose interval overlaps it. `certified_boost` ranks certified memories higher. |
| `ontology` (`OntologyConfig`) | Write-time embedding match of new entity names (`write_embedding_rung`), one question to the decision model for the ambiguous rest (`rung4`), and UPDATE closing the old fact's interval at the new fact's start (`interval_closing`). |
| `compilation` (`CompileConfig`) | Merge and proposal thresholds, automatic compile every N adds, and the observations that certify a memory. |
| `tiers` (`TiersConfig`) | `working`: a per-session window and rolling summary; `assemble(query, session_id, budget)` returns a `ContextBlock` (session, instructions, facts, events, each line with its provenance) and `end_session()` promotes the session. `procedural`: `add_instruction()` stores certified instructions, `auto_triggers` picks up "always ...", "no, use ..." turns, and procedural memories rank higher. `episodic`: recency-weighted ranking. |

```python
from grafeo_memory import BucketConfig, MemoryConfig, PolicyConfig, RetrievalConfig, TaxonomyConfig

config = MemoryConfig(
    decision_model="jev",
    bucket=BucketConfig(enabled=True, floor=0.5),
    policy=PolicyConfig(enabled=True),
    taxonomy=TaxonomyConfig(enabled=True, mode="soft"),
    retrieval=RetrievalConfig(order="deterministic"),
)
```

The rules, the scoping and the retrieval order run locally. Re-judging an open decision sends its question and
evidence to the LLM (or to the decision model with `resolve_engine="decision"`); `rung4` and
`taxonomy.router_engine="decision"` send entity names or fact text to the decision model. The benchmark harnesses
set any of these fields with `--set`, for example `--set bucket.enabled=true --set bucket.floor=0.4`.

### LLM call cache

`MemoryConfig(cache=True)` (or `CacheConfig(...)`; for the CLI and MCP server, `GRAFEO_MEMORY_CACHE_ENABLED=true`) caches LLM, embedding
and decision-model calls in memory and, for a file database, in `<db_path>.llmcache.jsonl`, so an identical call is
answered without reaching the provider. The cache file holds prompts and responses, which include memory text;
delete it together with the database, or call `MemoryCache.clear()`.

## CLI

```bash
export GRAFEO_MEMORY_DB="$HOME/.grafeo/memory.grafeo"   # an absolute path
grafeo-memory add "I indent with tabs, never spaces" --speaker Richard
grafeo-memory add "Middle-out compression scores 5.2 on the Weissman scale" --source notes
grafeo-memory search "Weissman score" -k 5
grafeo-memory list
grafeo-memory history 42
grafeo-memory delete 42
grafeo-memory stats
```

Global flags: `--model`, `--db`, `--user` (each overrides its environment variable, see
[Environment Variables](#environment-variables)) and `--json` for machine-readable output. `add` takes `--speaker`,
`--source chat|notes`, `--type procedural` and `--no-infer`; `-` reads the text from stdin. Other commands:
`update`, `summarize`, `explain`.

## MCP Server

grafeo-memory includes a built-in MCP server so AI agents (Claude Desktop, Cursor, etc.) can use it as a tool.

```bash
uv add "grafeo-memory[mcp,mistral]"   # GRAFEO_MEMORY_MODEL=mistral:mistral-small-latest
uv add "grafeo-memory[mcp,openai]"    # the default model, openai:gpt-4o-mini
```

The server needs the provider extra of its model too: Mistral embeddings for a `mistral:` model, OpenAI embeddings
otherwise. Without `GRAFEO_MEMORY_MODEL` it uses `openai:gpt-4o-mini`.

Give the server an absolute database path: a client starts it in a working directory of its choosing, so
`./memory.grafeo` lands in a different place per client.

### Claude Code

```bash
claude mcp add grafeo-memory \
  -e GRAFEO_MEMORY_MODEL=mistral:mistral-small-latest \
  -e GRAFEO_MEMORY_DB=/home/me/.grafeo/memory.grafeo \
  -e MISTRAL_API_KEY=... \
  -- grafeo-memory-mcp
```

Or keep the keys in a file and pass `-e GRAFEO_MEMORY_ENV_FILE=/home/me/.grafeo/.env`.

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "grafeo-memory": {
      "command": "grafeo-memory-mcp",
      "env": {
        "GRAFEO_MEMORY_MODEL": "mistral:mistral-small-latest",
        "GRAFEO_MEMORY_DB": "/Users/me/.grafeo/memory.grafeo",
        "MISTRAL_API_KEY": "..."
      }
    }
  }
}
```

### Available Tools

| Tool | Description |
| ---- | ----------- |
| `memory_add` | Add a memory by extracting facts from text |
| `memory_add_batch` | Add multiple memories in one batch |
| `memory_search` | Search memories by semantic similarity and graph context |
| `memory_update` | Update an existing memory's text |
| `memory_delete` | Delete a single memory |
| `memory_delete_all` | Delete all memories for a user |
| `memory_list` | List all stored memories |
| `memory_summarize` | Consolidate old memories into topic-grouped summaries |
| `memory_history` | Show change history for a memory |
| `memory_open_decisions` | List questions the memory parked because it was unsure |
| `memory_resolve_decision` | Answer a parked question |
| `memory_manage_roots` | Show or edit the subject roots that scope memories |
| `memory_assemble` | Build a prompt-ready context block with provenance |
| `memory_add_instruction` | Store a standing instruction, certified |
| `memory_session_start` | Start a session (anonymous, or bound to a user) |
| `memory_session_end` | End a session: promote a bound one, discard an anonymous one |

`memory_add` treats its text as an agent's notes (`source="notes"`) unless told `source="chat"`, and takes
`speaker` and `session_id`. Every tool acts for its `user_id` argument, else the server's default user, and reaches
that user's memories only.

### Environment Variables

The CLI and the MCP server read these from the environment, from `.env` in the working directory, and from the
file named by `GRAFEO_MEMORY_ENV_FILE`; a variable that is already set wins, and an empty one counts as unset.

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `GRAFEO_MEMORY_MODEL` | `openai:gpt-4o-mini` | pydantic-ai model string |
| `GRAFEO_MEMORY_DB` | *(in-memory)* | Database file path; use an absolute path |
| `GRAFEO_MEMORY_USER` | `default` | Default user ID |
| `GRAFEO_MEMORY_ENV_FILE` | *(none)* | A `.env` file to read keys and settings from |
| `MISTRAL_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY` | *(none)* | Provider keys, never logged |
| `GRAFEO_MEMORY_QUERY_ENTITIES` | `lexical` | `llm`: search asks the model for the entities a query names (one call per search) |
| `GRAFEO_MEMORY_CACHE_ENABLED` | `false` | Cache LLM, embedding and decision calls |
| `GRAFEO_MEMORY_CACHE_MAX_ENTRIES`, `GRAFEO_MEMORY_CACHE_MAX_MEMORY_MB`, `GRAFEO_MEMORY_CACHE_MAX_STORAGE_MB` | `10000`, `128`, `512` | Cache limits |
| `GRAFEO_MEMORY_CACHE_PERSIST_PATH` | *(next to the database)* | Cache file |
| `GRAFEO_MEMORY_DB_MEMORY_LIMIT_MB`, `GRAFEO_MEMORY_DB_SPILL_LIMIT_MB` | *(grafeo default)* | Engine memory and spill caps |
| `GRAFEO_MEMORY_DECISION_MODEL`, `TYPESAFE_API_KEY` | *(none)* | Decision model name (`jev`) and its key |
| `GRAFEO_MEMORY_LOG_LEVEL` | `WARNING` | Log level on stderr, secrets redacted |
| `GRAFEO_MEMORY_YOLO` | *(off)* | Set to `1` for all features |

### Transport

Supports stdio (default), SSE and streamable HTTP:

```bash
grafeo-memory-mcp              # stdio (default)
grafeo-memory-mcp sse          # SSE
grafeo-memory-mcp streamable-http
```

> **Note:** This is different from [grafeo-mcp](https://github.com/GrafeoDB/grafeo-mcp), which exposes the raw GrafeoDB database. grafeo-memory-mcp wraps the high-level memory API (extract, reconcile, search, summarize).

## Observability

grafeo-memory supports OpenTelemetry instrumentation via pydantic-ai. When enabled, all LLM calls (extraction, reconciliation, summarization, reranking) are traced automatically.

```python
config = MemoryConfig(instrument=True)  # uses global OTel provider
```

For custom providers:

```python
from grafeo_memory import InstrumentationSettings

config = MemoryConfig(
    instrument=InstrumentationSettings(
        tracer_provider=my_tracer_provider,
        include_content=False,
    )
)
```

## Why grafeo-memory?

- **One file**: memories, embeddings, entities and history live in a single `.grafeo` file, with no server or
  container to run.
- **Small install**: about 16 MB with `uv add`.
- **Offline and at the edge**: the database runs inside the process; only model calls leave the device, and none
  do with a local model through Ollama.
- **Graph and vector in one engine**: search combines vector similarity with graph context.
- **Any LLM**: pydantic-ai model strings (Mistral, Ollama, Anthropic, OpenAI and more).
- **Any embedder**: Mistral and OpenAI are built in; others plug in through the `EmbeddingClient` protocol.

## API Reference

### `MemoryManager`

- `MemoryManager(model, config=None, *, embedder)`: create memory manager. `model` is a pydantic-ai model string (e.g. `"mistral:mistral-small-latest"`)
- `.add(messages, user_id=None, session_id=None, metadata=None, *, infer=True, importance=1.0, memory_type="semantic", speaker=None, source="chat")` → `AddResult` (list of `MemoryEvent`)
- `.for_user(user_id)` → a view bound to one user (see [Multi-User Isolation](#multi-user-isolation))
- `.start_session(session_id=None, *, anonymous=False)` → `str`; `.assemble(query, session_id, budget)` → `ContextBlock`; `.end_session(session_id)` → `AddResult` (see [Sessions](#sessions))
- `.compile()` → `CompileResult`: the offline pass (entity merges, graph metrics, community summaries, layer upkeep)
- `.search(query, user_id=None, k=10, *, filters=None, rerank=True, memory_type=None)` → `SearchResponse` (list of `SearchResult`)
- `.update(memory_id, text)` → `MemoryEvent`: update a memory's text directly. Raises `ValueError` when `memory_id` is not a memory of this manager
- `.get_all(user_id=None, memory_type=None)` → `list[SearchResult]`
- `.delete(memory_id)` → `bool`: delete a memory with its edges (links to entities, session chain, provenance). `False` when `memory_id` is not a memory of this manager; other nodes in a shared database are never touched
- `.delete_all(user_id=None)` → `int` (count deleted, expired memories included)
- `.summarize(user_id=None, *, preserve_recent=5, batch_size=20)` → `AddResult`
- `.history(memory_id)` → `list[HistoryEntry]`: returns change history sorted by timestamp (oldest first), also for deleted memories (see [Memory History](#memory-history))
- `.set_importance(memory_id, importance)` → `bool`
- `.close()`: close the database

Use as a context manager: `with MemoryManager(...) as memory:`. Multiple sessions in the same process are supported.

### `MemoryConfig`

- `db_path`: path to the database, preferably ending in `.grafeo` (None for in-memory, see [Persistence](#persistence))
- `user_id`: default user scope (default `"default"`)
- `session_id`: default session scope
- `agent_id`: default agent scope
- `reconciliation_threshold`: minimum cosine similarity (0.0 to 1.0) for a memory to be considered a reconciliation candidate during `add()`. Lower values find more candidates, higher values require closer matches. At 0.3 (default), loosely related memories are considered. At 0.7+, only very similar memories trigger UPDATE/DELETE decisions
- `search_min_score`: minimum score for search results, 0.0 returns everything (default 0.0)
- `agreement_bonus`: score boost when both vector and graph find the same memory (default 0.1)
- `embedding_dimensions`: vector dimensions (default 1536)
- `enable_importance`: enable composite scoring with recency/frequency/importance (default False)
- `weight_topology`: topology score weight for graph-connected memories (default 0.0, requires `enable_importance`)
- `enable_topology_boost`: re-rank search results by graph connectivity, no LLM call (default False)
- `topology_boost_factor`: strength of topology boost (default 0.2)
- `consolidation_protect_threshold`: protect well-connected memories from summarize (default 0.0, off)
- `instrument`: OpenTelemetry instrumentation, `True` or `InstrumentationSettings` (default False)
- `decision_model`: decision model for non-LLM pipeline stages, a name or a `DecisionModel` (default None)
- `pipeline`: `PipelineConfig`, the engine per stage and the layer toggles (default: LLM everywhere, toggles off)
- `cache`: `True` or `CacheConfig` to cache LLM, embedding and decision calls (default False)
- `history_retention`: grafeo history retention for time travel: `"off"`, `"forever"`, a `timedelta` or seconds (default None, grafeo's default)
- `db_memory_limit_bytes`, `db_spill_limit_bytes`: grafeo memory and spill caps (default None, grafeo's machine default)

### `EmbeddingClient` (Protocol)

- `.embed(texts: list[str]) -> list[list[float]]`: generate embeddings for a batch of texts
- `.dimensions -> int`: return the embedding vector dimensionality

### Return Types

- **`AddResult`**: list subclass of `MemoryEvent`, with `.usage` for LLM token counts
- **`SearchResponse`**: list subclass of `SearchResult`, with `.usage` for LLM token counts
- **`MemoryEvent`**: `.action` (ADD/UPDATE/DELETE/NONE), `.memory_id`, `.text`, `.old_text`
- **`SearchResult`**: `.memory_id`, `.text`, `.score`, `.user_id`, `.metadata`, `.relations`, `.memory_type`
- **`HistoryEntry`**: `.event` (ADD/UPDATE/DELETE), `.old_text`, `.new_text`, `.timestamp` (epoch ms), `.actor_id`, `.role`. Returned sorted by timestamp ascending (oldest first)

### Iteration

```python
# AddResult is iterable:
for event in memory.add("text"):
    print(event.action, event.text)

# SearchResponse is iterable:
for result in memory.search("query"):
    print(result.text, result.score)
```

## Ecosystem

grafeo-memory is part of the GrafeoDB ecosystem:

- **[grafeo](https://github.com/GrafeoDB/grafeo)**: Core graph database engine (Rust)
- **grafeo-memory-mcp** (built-in): MCP server for the memory API (`uv add "grafeo-memory[mcp,mistral]"`)
- **[grafeo-mcp](https://github.com/GrafeoDB/grafeo-mcp)**: MCP server for raw GrafeoDB access
- **[grafeo-langchain](https://github.com/GrafeoDB/grafeo-langchain)**: LangChain integration
- **[grafeo-llamaindex](https://github.com/GrafeoDB/grafeo-llamaindex)**: LlamaIndex integration

All packages share the same database file. Build memories with grafeo-memory, query them with grafeo-langchain, expose them via MCP.

Docs and the other Grafeo AI tools: [grafeo.ai](https://grafeo.ai).

## Requirements

- Python 3.13+
- [grafeo](https://pypi.org/project/grafeo/) `>=0.6.2,<0.7` (installed automatically)

## License

Apache-2.0
