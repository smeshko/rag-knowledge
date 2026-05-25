# Epic 2 — Data Model & Migrations

**Status**: Blocked (depends on Epic 1)

## Overview

Implement all seven core tables as SQLAlchemy 2.0 ORM models with the relationships, JSON columns, pgvector columns, indexes, and uniqueness constraints defined in the architecture. Produce one initial Alembic migration that creates the entire schema. After this epic, the database matches the canonical data model and supports every downstream concern (file storage references, ingestion writes, retrieval reads).

## Architecture references

- [02 — Core Data Model](../../architecture/02-core-data-model.md) — full table definitions, statuses, denormalizations, hash semantics
- [05 — Storage and Indexing](../../architecture/05-storage-and-indexing.md) — table-by-table storage details, uniqueness constraints, timestamps
- [13 — Implementation Decisions, topic 2](../../architecture/13-implementation-decisions.md#2-database-client-and-migration-tooling) — SQLAlchemy 2.0 async + Alembic + pgvector-python

## Dependencies

- Epic 1 (scaffold, Alembic, async engine)

## Out of scope

- Repository pattern / query helpers (added per-epic as needed)
- Indexes that are pure query-optimization (those land alongside the queries that need them in later epics)
- Future projection tables like `ingredient_mentions` (deferred per doc 5)

---

## Phase 2.1 — Models: source_assets, documents, source_spans, extraction_runs

**Goal**: Implement the four append-only / parent-side tables using SQLAlchemy 2.0 typed style.

### What to build

In `src/rag_recipes/storage/models.py` (or split per table — pick one and be consistent):

- **`SourceAsset`** (table `source_assets`) — per [doc 2 § 1](../../architecture/02-core-data-model.md#1-sourceasset):
  - `id`, `source_type`, `original_filename`, `storage_provider`, `storage_key`, `content_hash`, `upload_status`, `created_at`, `updated_at`
  - Enum constraint on `upload_status`: `uploading | uploaded | upload_failed | deleted`
- **`Document`** (table `documents`) — per [doc 2 § 2](../../architecture/02-core-data-model.md#2-document):
  - `id`, `asset_id` (FK), `category`, `subcategory` (nullable), `title`, `author`, `source_type`, `language` (nullable), `active_source_version` (nullable), `status`, `created_at`, `updated_at`
  - Enum constraint on `status`: see full list in doc 2 (queued, extracting_text, ..., ready, needs_review, failed)
  - Intentional denormalization: `source_type` duplicates `source_assets.source_type` — add a validator at the ORM level
- **`SourceSpan`** (table `source_spans`) — per [doc 2 § 3](../../architecture/02-core-data-model.md#3-sourcespan):
  - `id`, `document_id` (FK), `source_version`, `source_type`, `locator` (JSONB), `locator_hash`, `text`, `text_hash`, `created_at`
  - Immutable (no `updated_at`)
- **`ExtractionRun`** (table `extraction_runs`) — per [doc 2 § 7](../../architecture/02-core-data-model.md#7-extractionrun):
  - `id`, `document_id` (FK), `source_version`, `provider`, `model`, `prompt_version`, `schema_version`, `input_source_span_ids` (JSONB array), `input_hash`, `status`, `output_json` (JSONB), `error_message`, `created_at`, `completed_at`
  - Enum constraint on `status`: `running | success | failed | rejected`

Relationship declarations using `Mapped[]` + `relationship(...)`:
- `Document.source_asset` ↔ `SourceAsset.document` (one-to-one)
- `Document.source_spans` (one-to-many)
- `Document.extraction_runs` (one-to-many)

### Acceptance criteria

- [ ] All four model classes defined with SQLAlchemy 2.0 typed style (`Mapped[...]`)
- [ ] Foreign keys, enum constraints, and nullable fields match doc 2 exactly
- [ ] `source_type` denormalization validated at ORM level (raises on mismatch)
- [ ] mypy passes against the models
- [ ] Unit tests round-trip each model through SQLAlchemy in-memory or via test DB (basic create/query)

### Validation

Run `make test-unit` — models construct and validate correctly without a real DB.

---

## Phase 2.2 — Models: knowledge_items, chunks, chunk_embeddings

**Goal**: Implement the three knowledge-layer tables, including pgvector `Vector` columns and JSON `structured_data`.

### What to build

- **`KnowledgeItem`** (table `knowledge_items`) — per [doc 2 § 4](../../architecture/02-core-data-model.md#4-knowledgeitem):
  - `id`, `document_id` (FK), `extraction_run_id` (FK), `source_version`, `item_type`, `title`, `normalized_title`, `summary`, `body_text`, `source_span_ids` (JSONB array of IDs), `structured_data` (JSONB), `confidence` (JSONB), `status`, `created_at`, `updated_at`
  - Enum constraint on `status`: `ready | needs_review | superseded`
- **`Chunk`** (table `chunks`) — per [doc 2 § 5](../../architecture/02-core-data-model.md#5-chunk):
  - `id`, `document_id` (FK; denormalized — validate matches parent item's document), `parent_type`, `parent_id`, `chunk_type`, `text`, `text_hash`, `source_span_ids` (JSONB), `metadata` (JSONB), `created_at`, `updated_at`
  - Enum constraint on `chunk_type`: `recipe_full | recipe_title | recipe_summary | recipe_ingredients | recipe_steps`
  - Enum constraint on `parent_type`: `knowledge_item` (only value for MVP)
- **`ChunkEmbedding`** (table `chunk_embeddings`) — per [doc 2 § 6](../../architecture/02-core-data-model.md#6-chunkembedding):
  - `id`, `chunk_id` (FK), `embedding_provider`, `embedding_model`, `embedding_dimensions`, `embedding_vector` (`Vector(1536)` from `pgvector.sqlalchemy`), `created_at`
  - No `updated_at` — replaceable rather than mutated

Relationship declarations:
- `KnowledgeItem.document`, `KnowledgeItem.extraction_run`, `KnowledgeItem.chunks`
- `Chunk.knowledge_item` (via `parent_id` when `parent_type = 'knowledge_item'`), `Chunk.embeddings`
- `ChunkEmbedding.chunk`

### Acceptance criteria

- [ ] All three model classes defined
- [ ] `Chunk.document_id` validator confirms it matches the parent `KnowledgeItem.document_id`
- [ ] `ChunkEmbedding.embedding_vector` uses the pgvector `Vector` type at dimension 1536
- [ ] JSONB columns are typed as `Mapped[dict[str, Any]]` (or a more specific TypedDict where helpful)
- [ ] mypy passes

### Validation

Models import without errors; unit tests construct instances; pgvector type imports cleanly.

---

## Phase 2.3 — Initial Alembic migration with indexes and uniqueness constraints

**Goal**: One Alembic migration creates the full schema, including pgvector extension, all uniqueness constraints, and the first batch of indexes.

### What to build

- A single revision `migrations/versions/<rev>_initial_schema.py` that:
  - `CREATE EXTENSION IF NOT EXISTS vector;`
  - Creates all seven tables with columns, FKs, enum checks, NOT NULL constraints
  - Adds **uniqueness constraints** per [doc 5 § Uniqueness Constraints Worth Stating Early](../../architecture/05-storage-and-indexing.md#uniqueness-constraints-worth-stating-early):
    - `source_assets.content_hash` unique
    - `documents.asset_id` unique
    - `source_spans (document_id, source_version, locator_hash)` unique
    - `chunk_embeddings (chunk_id, embedding_provider, embedding_model)` unique
  - Adds softer constraints in application logic for now (do *not* add DB constraint for `(document_id, item_type, normalized_title, source_version)` etc. — leave that to application layer until extraction/dedup are mature)
  - Adds **denormalization-enforcing composite FKs** to close the raw-FK-ID write gaps left by the Phase 2.2 `@validates` soft checks (which only fire on relationship assignment, not raw-ID construction):
    - `UNIQUE(knowledge_items.id, document_id)` + composite FK `chunks(parent_id, document_id) → knowledge_items(id, document_id)` — rejects any chunk whose `document_id` disagrees with its parent item's `document_id`, regardless of construction path
    - `UNIQUE(extraction_runs.id, document_id)` + composite FK `knowledge_items(extraction_run_id, document_id) → extraction_runs(id, document_id)` — rejects any knowledge item whose `document_id` disagrees with its extraction run's `document_id`
    - These keep the single-column FKs (`chunks.parent_id`, `knowledge_items.extraction_run_id`) as well; the composite FKs are additive integrity, and must be dropped/replaced when non-`knowledge_item` chunk parent types land (Phase 12+)
  - Adds initial indexes:
    - `chunks(document_id)`, `chunks(parent_type, parent_id)`
    - `knowledge_items(document_id)`, `knowledge_items(status)`
    - `source_spans(document_id, source_version)`
    - GIN/IVF/HNSW indexes for FTS and pgvector are deferred to Epic 10 (added when the queries that use them land)
- Migration is reversible (`downgrade()` drops everything cleanly)
- `make migrate` applies it; `alembic downgrade base && alembic upgrade head` round-trips

### Acceptance criteria

- [ ] `make migrate` on a fresh DB creates all tables with expected columns, FKs, constraints
- [ ] `pgvector` extension installed (verifiable via `\dx` in `psql`)
- [ ] All uniqueness constraints from doc 5 present
- [ ] Denormalization composite FKs present and enforced: an insert with `chunks.document_id` disagreeing with its parent item's `document_id`, or `knowledge_items.document_id` disagreeing with its extraction run's `document_id`, is rejected by the DB even via raw-ID writes
- [ ] `alembic downgrade base` cleanly removes everything
- [ ] `alembic upgrade head` after downgrade re-creates everything identically
- [ ] Integration test: insert sample data covering all FK relationships; FK violations rejected as expected

### Validation

Spin up a fresh DB: `make clean && make setup` — verify schema with `\d+` in psql for each table. Run a small integration test that exercises the FK chain SourceAsset → Document → SourceSpan → ExtractionRun → KnowledgeItem → Chunk → ChunkEmbedding.

---

## Epic-level acceptance criteria

- [ ] All seven tables created with correct columns, FKs, and constraints
- [ ] pgvector extension enabled
- [ ] Alembic migration applies cleanly on a fresh DB and round-trips through downgrade/upgrade
- [ ] Models pass mypy and have unit-test-level construction coverage
- [ ] Integration test exercises the full FK chain end-to-end
- [ ] Status in [`EPICS.md`](../EPICS.md) updated; Epics 3, 6, 7 unblocked
