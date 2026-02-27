"""Prompt templates for LLM-driven extraction and reconciliation."""

from __future__ import annotations

FACT_EXTRACTION_SYSTEM = """\
You are a memory extraction assistant. Your job is to extract discrete, \
self-contained facts from conversation text.

Rules:
- Each fact should be a short, clear statement (one sentence).
- Replace pronouns with the actual names or identifiers when possible.
- Prefer third-person statements ("alice likes hiking" not "I like hiking").
- Extract preferences, biographical details, relationships, events, and opinions.
- Do NOT extract greetings, pleasantries, or filler.
- If the text contains no memorable facts, return an empty list."""

FACT_EXTRACTION_USER = """\
Extract facts from the following text. The speaker's user_id is "{user_id}".

Text:
{text}"""


# --- Procedural memory extraction ---

PROCEDURAL_EXTRACTION_SYSTEM = """\
You are a procedural memory extraction assistant. Your job is to extract instructions, \
preferences, behavioral rules, and step-by-step procedures from conversation text.

Rules:
- Each extracted item should be a clear, actionable instruction or preference.
- Focus on: behavioral preferences ("always X", "prefer Y"), workflow instructions \
("when X happens, do Y"), style guidelines ("use formal tone"), tool/method preferences \
("use pytest for testing"), and step-by-step procedures.
- Replace pronouns with actual names or identifiers when possible.
- Prefer imperative or third-person form ("always use type hints" or "user prefers type hints").
- Do NOT extract factual knowledge, biographical details, or events — those are semantic memories.
- Do NOT extract greetings, pleasantries, or filler.
- If the text contains no instructions or preferences, return an empty list."""


# --- Combined extraction (facts + entities in one call) ---

COMBINED_EXTRACTION_SYSTEM = """\
You are a memory extraction assistant. Your job is to extract facts, entities, \
and relationships from conversation text.

Extract two things:
1. FACTS: Discrete, self-contained factual statements.
   - Each fact should be a short, clear statement (one or two sentences).
   - Group closely related details into a single fact when they describe \
the same topic (e.g., "marcus plays guitar, is learning jazz, and focuses on \
Wes Montgomery's style" rather than three separate facts).
   - Replace pronouns with actual names or identifiers when possible.
   - Prefer third-person statements ("alice likes hiking" not "I like hiking").
   - Extract preferences, biographical details, relationships, events, and opinions.
   - Do NOT extract greetings, pleasantries, or filler.

2. ENTITIES and RELATIONSHIPS from the facts you extracted.
   - Identify key entities (people, organizations, locations, concepts).
   - Entity names should be lowercase with underscores for spaces (e.g., "acme_corp").
   - Identify relationships between entities.

If the text contains no memorable facts, return empty lists."""

COMBINED_EXTRACTION_USER = """\
Extract facts, entities, and relationships from the following text. \
The speaker's user_id is "{user_id}".

Text:
{text}"""

COMBINED_PROCEDURAL_EXTRACTION_SYSTEM = """\
You are a procedural memory extraction assistant. Extract instructions, preferences, \
behavioral rules, entities, and relationships from conversation text.

Extract two things:
1. FACTS: Clear, actionable instructions or preferences.
   - Focus on: behavioral preferences ("always X", "prefer Y"), workflow instructions \
("when X happens, do Y"), style guidelines, tool/method preferences, and procedures.
   - Group related preferences into single statements when they describe the same topic.
   - Replace pronouns with actual names or identifiers when possible.
   - Prefer imperative or third-person form.
   - Do NOT extract factual knowledge, biographical details, or events.

2. ENTITIES and RELATIONSHIPS from the extracted items.
   - Identify key entities (tools, technologies, people, concepts).
   - Entity names should be lowercase with underscores for spaces.
   - Identify relationships between entities.

If the text contains no instructions or preferences, return empty lists."""


# --- Entity / Relation extraction (standalone, used for search query extraction) ---

ENTITY_EXTRACTION_SYSTEM = """\
You are an entity and relationship extraction assistant. Given a list of facts, \
identify the key entities (people, organizations, locations, concepts, etc.) and \
the relationships between them.

Return all entities and relationships you find. Entity names should be lowercase and \
use underscores for spaces (e.g., "acme_corp", "new_york")."""

ENTITY_EXTRACTION_USER = """\
Extract entities and relationships from these facts (user_id: "{user_id}"):

{facts}"""


# --- Reconciliation ---

RECONCILIATION_SYSTEM = """\
You are a memory reconciliation assistant. You will receive NEW facts and \
EXISTING memories from a knowledge base. For each new fact, decide what to do:

- ADD: The fact is genuinely new information. Store it as a new memory.
- UPDATE: The fact modifies or replaces an existing memory. \
You MUST set target_memory_id to the ID of the memory being updated, and provide the new text.
- DELETE: The fact contradicts an existing memory and the old memory should be removed. \
You MUST set target_memory_id to the ID of the memory being deleted.
- NONE: The fact is already captured by existing memories. Skip it.

Rules:
- Prefer UPDATE over DELETE+ADD when information changes \
(e.g., job change, moved cities, relationship change).
- Only DELETE when information is explicitly contradicted and cannot be merged via UPDATE.
- Merge related facts into a single UPDATE when they refer to the same memory.
- If there are no existing memories, all facts should be ADD.
- For UPDATE and DELETE, always include target_memory_id — without it the action cannot be executed."""

RECONCILIATION_USER = """\
NEW FACTS:
{new_facts}

EXISTING MEMORIES:
{existing_memories}

Decide what to do with each new fact."""


# --- Relation reconciliation ---

RELATION_RECONCILE_SYSTEM = """\
You are a graph memory manager. You will receive EXISTING relationships from a \
knowledge graph and NEW relationships extracted from recent text. Identify existing \
relationships that should be DELETED because they are contradicted or outdated.

Deletion criteria:
- The new information directly contradicts an existing relationship \
(e.g., changed job, moved to a new city).
- The existing relationship is outdated and replaced by the new information.

DO NOT delete if:
- The same relationship type can have multiple valid destinations. \
Example: "alice -- likes -- pizza" and "alice -- likes -- sushi" can BOTH exist.
- The relationships are about different topics or entities.

Return an empty list if nothing should be deleted."""

RELATION_RECONCILE_USER = """\
EXISTING RELATIONSHIPS:
{existing_relations}

NEW RELATIONSHIPS:
{new_relations}

Identify which existing relationships should be deleted."""


# --- Summarization / consolidation ---

SUMMARIZE_SYSTEM = """\
You are a memory consolidation assistant. You will receive a batch of individual \
memory entries about a user. Your job is to consolidate them into fewer, more \
concise entries while preserving ALL factual information.

Rules:
- Group related memories by topic or theme (e.g., work, hobbies, personal details).
- Produce one consolidated memory per distinct topic.
- Each consolidated memory should be a clear, self-contained statement.
- Preserve ALL facts — do not drop any information, even if minor.
- Merge duplicate or overlapping information into single statements.
- Use third-person form (e.g., "User prefers python" not "I prefer python").
- If a memory cannot be grouped with others, keep it as-is.
- Do NOT invent or infer facts not present in the input."""

SUMMARIZE_USER = """\
Consolidate these {count} memories into fewer entries, grouped by topic:

{memories}"""


# --- Image description (vision) ---

IMAGE_DESCRIBE_SYSTEM = """\
You are an image analysis assistant. Describe images in factual, concise terms. \
Focus on people, objects, text, locations, activities, and any notable details. \
Do not speculate about context not visible in the image."""
