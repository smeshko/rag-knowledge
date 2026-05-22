# Epic 10 — Chunking, Embedding, Indexing

**Status**: Blocked (depends on Epics 5, 9)

## Overview

Turn `ready` `KnowledgeItem` rows into searchable artifacts: create canonical recipe chunks (5 chunk types), generate embeddings for each chunk, and create the Postgres FTS + pgvector indexes. After this epic, the document lifecycle ends at `Document.status = "ready"` (or `needs_review` if no ready items were produced), and the database is queryable for retrieval (Epic 12).

## Architecture references

- [02 — Core Data Model § 5, 6](../../architecture/02-core-data-model.md#5-chunk) — chunk and embedding shapes, canonical chunk types
- [03 — PDF Ingestion Pipeline § 10, 11](../../architecture/03-pdf-ingestion-pipeline.md#10-create-chunks) — chunking and embedding stages
- [04 — LLM-Assisted Recipe Extraction § Chunk Creation](../../architecture/04-llm-assisted-recipe-extraction.md#chunk-creation) — how each chunk type maps to a retrieval pattern
- [05 — Storage and Indexing § Keyword Search / Vector Search](../../architecture/05-storage-and-indexing.md#keyword-search) — what indexes to create

## Dependencies

- Epic 5 (`OpenAIEmbeddingProvider`)
- Epic 9 (ready `KnowledgeItem` rows with `structured_data`)

## Out of scope

- Retrieval queries (Epic 12 — this epic just creates the indexes)
- Re-embedding on model change (handled via the EmbeddingProvider interface; ad-hoc until a re-embed admin path is needed)
- The future `ingredient_mentions` projection (deferred per doc 5)

---

## Phase 10.1 — Chunk creation from ready KnowledgeItems

**Goal**: For each `KnowledgeItem` with `status="ready"`, create the five canonical recipe chunks per [doc 2 § Canonical recipe chunk types](../../architecture/02-core-data-model.md#canonical-recipe-chunk-types).

### What to build

- **`src/rag_recipes/ingestion/pipeline/chunking.py`**:
  - `build_chunks(item: KnowledgeItem) -> list[Chunk]` returns five chunks:
    - `recipe_title` — text is `item.title` (or a slightly richer "title + author" line; designer's choice — document it)
    - `recipe_summary` — text is `item.summary`
    - `recipe_ingredients` — text is `structured_data.ingredients_text` (or constructed from the structured list if `ingredients_text` is absent)
    - `recipe_steps` — text is `structured_data.steps_text` (or constructed from steps)
    - `recipe_full` — text is `item.body_text` (or a concatenation of title + ingredients + steps if `body_text` is absent)
  - Each chunk: `document_id` (matches parent), `parent_type="knowledge_item"`, `parent_id=item.id`, `chunk_type`, `text`, `text_hash`, `source_span_ids` (inherits item's), `metadata` JSONB with `{category, item_type, title}`
- **Persistence**: insert chunks in a single transaction per item
- Handle missing source fields gracefully (skip chunks whose source text is empty rather than creating empty chunks)
- Unit tests:
  - Five chunks per ready item (when all source text is present)
  - Fewer chunks if some source text is empty
  - Skipping `needs_review` and `superseded` items
  - `text_hash` deterministic
  - Metadata block correctly populated

### Acceptance criteria

- [ ] Each `ready` item produces up to five chunks of the canonical types
- [ ] `needs_review` and `superseded` items produce no chunks
- [ ] `Chunk.document_id` matches parent item's document (FK + validator)
- [ ] `text_hash` deterministic for identical text
- [ ] Unit tests cover the chunk-building rules

### Validation

`make test-unit`; integration test runs full pipeline and inspects `chunks` table.

---

## Phase 10.2 — Embedding generation + chunk_embeddings storage

**Goal**: Every chunk gets a `ChunkEmbedding` row using the configured `EmbeddingProvider` (`OpenAIEmbeddingProvider` from Epic 5).

### What to build

- **`src/rag_recipes/ingestion/pipeline/embedding.py`**:
  - `embed_chunks(chunks: list[Chunk]) -> list[ChunkEmbedding]`:
    - Batches chunks (respect the embedding provider's batch size)
    - Calls `EmbeddingProvider.embed_batch([chunk.text for chunk in chunks])`
    - Builds `ChunkEmbedding` rows: `chunk_id`, `embedding_provider`, `embedding_model`, `embedding_dimensions`, `embedding_vector`
    - Inserts in batch
  - Respects the uniqueness constraint `(chunk_id, embedding_provider, embedding_model)` — on re-embed, upsert/replace
- **Re-embed helper** (not exposed via API yet, but used internally): `re_embed_for_model(provider, model)` that re-runs embedding generation when the configured model changes. Out of scope for MVP API, but the function should exist for ad-hoc use.
- Failure handling: a partial batch failure leaves the document in `embedding_chunks` status; either retry the whole batch or only the missing chunks (designer's choice — document the policy)

### Acceptance criteria

- [ ] Every chunk produces exactly one `ChunkEmbedding` for the configured provider/model
- [ ] Re-embedding the same chunk with the same provider/model upserts (does not duplicate)
- [ ] Re-embedding the same chunk with a *different* model inserts a new row
- [ ] Vector dimensions match `embedding_dimensions = 1536` for text-embedding-3-small
- [ ] Embedding generation respects batch sizes (verified via call log on `FakeEmbeddingProvider`)
- [ ] Langfuse traces capture batch embedding calls

### Validation

Integration test with `FakeEmbeddingProvider` covers single and multi-chunk paths; smoke test with real provider for end-to-end confidence.

---

## Phase 10.3 — PG FTS index, pgvector index, final status transition

**Goal**: Create the database indexes that retrieval queries depend on, and complete the ingestion lifecycle by transitioning `Document.status` to its terminal value.

### What to build

- **New Alembic migration** with:
  - Add a generated `tsvector` column on `chunks` (or a separate `chunks_fts` table — designer's choice; recommended: generated column on `chunks` keyed on `text`):
    - e.g., `ts_vector_column GENERATED ALWAYS AS (to_tsvector('english', text)) STORED`
  - GIN index on that column
  - HNSW index on `chunk_embeddings.embedding_vector` using `vector_cosine_ops` (pgvector syntax)
  - Index on `chunk_embeddings (embedding_provider, embedding_model)` so vector queries can filter
- **Pipeline orchestration**: after embeddings complete, transition `Document.status`:
  - If at least one chunk exists → `ready`
  - If zero ready items were produced (no chunks) → `needs_review` per [doc 2 § status](../../architecture/02-core-data-model.md#status)
  - Existing `needs_review` items stay; their lack of chunks is expected and consistent with doc 4
- Update `progress.stage` reporting through `creating_chunks`, `embedding_chunks`, `indexing`, `ready`

### Acceptance criteria

- [ ] Migration creates the GIN FTS index and the HNSW vector index
- [ ] `Document.status` lands on `ready` for documents with at least one chunked item
- [ ] `Document.status` lands on `needs_review` when ingestion produced zero ready items
- [ ] Status polling reflects the full sequence of stages
- [ ] Integration test: end-to-end ingestion of a multi-recipe fixture, ending at `ready`, with chunks + embeddings + indexes in place
- [ ] Simple SQL queries against the FTS column and the vector column return expected results (used in Epic 12)

### Validation

End-to-end integration test from `POST /documents` to `Document.status = "ready"`; `psql` queries confirm the indexes exist (`\d+ chunks`, `\d+ chunk_embeddings`) and that representative SELECTs use them (`EXPLAIN`).

---

## Epic-level acceptance criteria

- [ ] Chunks created from ready `KnowledgeItem` records (5 canonical types per item)
- [ ] Embeddings stored, uniqueness by `(chunk_id, provider, model)` respected
- [ ] FTS GIN index and pgvector HNSW index in place
- [ ] Full ingestion lifecycle ends at `Document.status = "ready"` or `needs_review`
- [ ] Status in [`EPICS.md`](../EPICS.md) updated; Epics 11 and 12 unblocked
