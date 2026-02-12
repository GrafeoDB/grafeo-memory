[![CI](https://github.com/GrafeoDB/grafeo-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/GrafeoDB/grafeo-memory/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GrafeoDB/grafeo-memory/graph/badge.svg)](https://codecov.io/gh/GrafeoDB/grafeo-memory)
[![PyPI](https://img.shields.io/pypi/v/grafeo-memory.svg)](https://pypi.org/project/grafeo-memory/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENCE)

# grafeo-memory

AI memory layer powered by [GrafeoDB](https://github.com/GrafeoDB/grafeo), an embedded graph database with native vector search.

No servers, no Docker, no Neo4j, no Qdrant. One `.db` file + one LLM.

```
Typical memory stack: Containers with Neo4j + Qdrant, Embedding API + LLM
grafeo-memory stack:  grafeo (single file) + LLM
```

## Install

```bash
uv add grafeo-memory                   # base (bring your own LLM + embedder)
uv add grafeo-memory[openai]           # + OpenAI embeddings
uv add grafeo-memory[anthropic]        # + Anthropic
uv add grafeo-memory[groq]             # + Groq
uv add grafeo-memory[all]              # all providers
```

Or with pip:

```bash
pip install grafeo-memory[openai]
```

## Quick Start

```python
from openai import OpenAI
from grafeo_memory import MemoryManager, MemoryConfig, OpenAIEmbedder

embedder = OpenAIEmbedder(OpenAI())
config = MemoryConfig(db_path="./memory.db", user_id="alice")

with MemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
    # Add memories from conversation
    events = memory.add("I just started a new job at Acme Corp as a data scientist")
    # -> [ADD "alice works at acme_corp", ADD "alice is a data_scientist"]

    events = memory.add("I've been promoted to senior data scientist at Acme")
    # -> [UPDATE "alice is a senior data scientist at acme_corp"]

    events = memory.add("I left Acme and joined Beta Inc")
    # -> [DELETE "alice works at acme_corp", ADD "alice works at beta_inc"]

    # Search
    results = memory.search("Where does Alice work?")
    # -> [SearchResult(text="alice works at beta_inc", score=0.92, ...)]
```

## How It Works

grafeo-memory implements the **reconciliation loop** &mdash; the intelligence layer that decides what to remember:

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
         │  single .db file   │
         └────────────────────┘
```

## Multi-User Isolation

```python
config = MemoryConfig(db_path="./chat_memory.db")

with MemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
    # Each user's memories are isolated
    memory.add("I love hiking in the mountains", user_id="bob")
    memory.add("I prefer beach vacations", user_id="carol")

    bob_results = memory.search("vacation preferences", user_id="bob")
    # -> hiking, mountains

    carol_results = memory.search("vacation preferences", user_id="carol")
    # -> beach vacations
```

## Supported LLM Providers

grafeo-memory uses [pydantic-ai](https://ai.pydantic.dev) model strings, so any provider pydantic-ai supports works out of the box:

```python
# OpenAI
MemoryManager("openai:gpt-4o-mini", config, embedder=embedder)

# Anthropic
MemoryManager("anthropic:claude-sonnet-4-5-20250929", config, embedder=embedder)

# Groq
MemoryManager("groq:llama-3.3-70b-versatile", config, embedder=embedder)

# Mistral
MemoryManager("mistral:mistral-large-latest", config, embedder=embedder)

# Google
MemoryManager("google-gla:gemini-2.0-flash", config, embedder=embedder)
```

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

memory = MemoryManager("openai:gpt-4o-mini", config, embedder=MyEmbedder())
```

## Why grafeo-memory?

| | Traditional stack | grafeo-memory |
|---|---|---|
| Infrastructure | Neo4j + Qdrant (Docker) | **Single .db file** |
| Install size | ~750MB (Docker images) | **~16MB** (pip install) |
| Offline/edge | Requires servers | **Yes** |
| Graph + vector | Separate services | **Unified engine** |
| LLM providers | Varies | **pydantic-ai** (OpenAI, Anthropic, Mistral, Groq, Google) |
| Embeddings | External API required | **Protocol-based** (any provider) |

## API Reference

### `MemoryManager`

- `MemoryManager(model, config=None, *, embedder)` &mdash; create memory manager. `model` is a pydantic-ai model string (e.g. `"openai:gpt-4o-mini"`)
- `.add(text, user_id=None, session_id=None, metadata=None)` &mdash; extract and store memories
- `.search(query, user_id=None, k=10)` &mdash; semantic + graph search
- `.get_all(user_id=None)` &mdash; retrieve all memories for a user
- `.delete(memory_id)` &mdash; delete a specific memory
- `.delete_all(user_id=None)` &mdash; delete all memories for a user
- `.history(memory_id)` &mdash; get update history for a memory (requires CDC feature)
- `.close()` &mdash; close the database

### `MemoryConfig`

- `db_path` &mdash; path to database file (None for in-memory)
- `user_id` &mdash; default user scope (default `"default"`)
- `session_id` &mdash; default session scope
- `agent_id` &mdash; default agent scope
- `similarity_threshold` &mdash; reconciliation similarity threshold (default 0.7)
- `embedding_dimensions` &mdash; vector dimensions (default 1536)
- `vector_property` &mdash; property name for embeddings (default `"embedding"`)
- `text_property` &mdash; property name for text content (default `"text"`)

### `EmbeddingClient` (Protocol)

- `.embed(texts: list[str]) -> list[list[float]]` &mdash; generate embeddings for a batch of texts
- `.dimensions -> int` &mdash; return the embedding vector dimensionality

### Types

- `MemoryAction` &mdash; enum: `ADD`, `UPDATE`, `DELETE`, `NONE`
- `MemoryEvent` &mdash; action, memory_id, text, old_text
- `SearchResult` &mdash; memory_id, text, score, user_id, metadata, relations

## Ecosystem

grafeo-memory is part of the GrafeoDB ecosystem:

- **[grafeo](https://github.com/GrafeoDB/grafeo)** &mdash; Core graph database engine (Rust)
- **[grafeo-langchain](https://github.com/GrafeoDB/grafeo-langchain)** &mdash; LangChain integration
- **[grafeo-llamaindex](https://github.com/GrafeoDB/grafeo-llamaindex)** &mdash; LlamaIndex integration
- **[grafeo-mcp](https://github.com/GrafeoDB/grafeo-mcp)** &mdash; MCP server for AI agents

All packages share the same `.db` file. Build memories with grafeo-memory, query them with grafeo-langchain, expose them via grafeo-mcp.

## Requirements

- Python 3.12+

## License

Apache-2.0
