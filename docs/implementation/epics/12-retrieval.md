# Epic 12 — Retrieval Layer

**Status**: Blocked (depends on Epic 10)

## Overview

Pure-logic retrieval: turn a query string into a ranked list of `KnowledgeItem` candidates with matched chunks and citation data. No API surface yet — that lands in Epic 13. This epic implements query normalization, metadata filters, keyword (Postgres FTS) and vector (pgvector) search, RRF merging with chunk-type boosts, and grouping chunk candidates into knowledge-item-level results.

## Architecture references

- [07 — Retrieval Behavior](../../architecture/07-retrieval-behavior.md) — entire document; this epic implements it
- [02 — Core Data Model § 5, 6](../../architecture/02-core-data-model.md#5-chunk) — chunk and embedding shapes
- [05 — Storage and Indexing § Keyword/Vector/Hybrid Search](../../architecture/05-storage-and-indexing.md#keyword-search) — indexing setup the queries depend on
- [13 — Implementation Decisions, topic 5](../../architecture/13-implementation-decisions.md#5-embedding-provider-and-model) — embedding provider/model filtering rule

## Dependencies

- Epic 10 (chunks, embeddings, indexes exist)

## Out of scope

- The `/search` endpoint (Epic 13)
- Reranking (deferred per doc 5 § Reranking)
- Query expansion / multi-query / HyDE (deferred per doc 7 § What We Are Not Doing Yet)

---

## Phase 12.1 — Query normalization + filters + keyword search

**Goal**: A keyword-only search function: normalized query → metadata-filtered → PG FTS results.

### What to build

- **`src/rag_recipes/retrieval/normalize.py`**:
  - `normalize_query(raw: str) -> NormalizedQuery` — trim, collapse whitespace, lowercase for keyword side, preserve original for display per [doc 7 § 2](../../architecture/07-retrieval-behavior.md#2-query-normalization)
- **`src/rag_recipes/retrieval/filters.py`**:
  - `build_filters(request) -> FilterSet` returning the constraints from [doc 7 § 3](../../architecture/07-retrieval-behavior.md#3-metadata-filters):
    - `knowledge_items.status = 'ready'`
    - `knowledge_items.source_version = documents.active_source_version`
    - `chunks.parent_type = 'knowledge_item'`
    - `documents.category = request.category` (default `recipes`)
    - `knowledge_items.item_type = request.filters.item_type` (default `recipe`)
    - `documents.subcategory` exact match if provided
    - Optional `document_ids` allowlist
- **`src/rag_recipes/retrieval/keyword.py`**:
  - `keyword_search(query, filters, top_k) -> list[ChunkCandidate]` running PG FTS:
    - `SELECT chunks.id, chunks.parent_id, chunks.chunk_type, ts_rank_cd(chunk_tsv, plainto_tsquery('english', :q)) AS score FROM chunks JOIN knowledge_items ... JOIN documents ... WHERE filters AND chunks.chunk_tsv @@ plainto_tsquery('english', :q) ORDER BY score DESC LIMIT :top_k`
    - `ChunkCandidate` dataclass: `chunk_id`, `knowledge_item_id`, `chunk_type`, `retrieval_source="keyword"`, `rank`, `raw_score`
- Constants from `Settings` per [doc 7 § 6](../../architecture/07-retrieval-behavior.md#6-candidate-counts): `search_keyword_top_k` default `max(limit*5, 50)`

### Acceptance criteria

- [ ] Query normalization matches doc 7 examples
- [ ] Keyword search returns candidates with rank and raw_score
- [ ] `needs_review` and `superseded` items excluded by filters (verified by integration test)
- [ ] Category and item_type filters honored
- [ ] Subcategory exact-match works; null subcategory excluded when filter is non-null
- [ ] Active source version filter excludes superseded data

### Validation

Integration test: ingest two cookbooks, query for an exact ingredient, verify only `ready` items from the right category come back.

---

## Phase 12.2 — Vector search with provider/model filter

**Goal**: A vector-only search function: embed the query with the configured model, search pgvector, return chunk candidates filtered by `(embedding_provider, embedding_model)`.

### What to build

- **`src/rag_recipes/retrieval/vector.py`**:
  - `vector_search(query, filters, top_k) -> list[ChunkCandidate]`:
    - Calls `EmbeddingProvider.embed_text(query)` to get the query vector (provider/model from `Settings`)
    - Runs the pgvector query joining `chunk_embeddings → chunks → knowledge_items → documents` with the filter constraints from Phase 12.1 *plus* `chunk_embeddings.embedding_provider = :p AND chunk_embeddings.embedding_model = :m`
    - Orders by `embedding_vector <=> :query_vector` (cosine distance with pgvector)
    - Returns `ChunkCandidate` with `retrieval_source="vector"`, `rank`, `distance`, `similarity = 1 - distance` (so higher = better, mirroring keyword's score direction)
- `search_vector_top_k` from `Settings`, same defaults as keyword

### Acceptance criteria

- [ ] Vector search uses pgvector cosine distance and filters by provider+model per [doc 7 § Embedding model rule](../../architecture/07-retrieval-behavior.md#embedding-model-rule)
- [ ] Returns top_k chunk candidates with rank and similarity
- [ ] All the same `ready`/`active_source_version` filters from Phase 12.1 apply
- [ ] Integration test: query for a "vibe" phrase (e.g., "cozy weeknight dinner") returns semantically-related recipes even when keyword wouldn't match
- [ ] Different embedding models in the DB do not pollute results (verified by inserting embeddings from two distinct models in a test)

### Validation

Integration test described above; `EXPLAIN` confirms the HNSW index from Epic 10 is used.

---

## Phase 12.3 — RRF merge + chunk-type boosts + KnowledgeItem grouping

**Goal**: A full hybrid search function that combines keyword and vector candidates via rank fusion with chunk-type boosts, then groups by parent `KnowledgeItem` and scores at the item level.

### What to build

- **`src/rag_recipes/retrieval/merge.py`**:
  - `merge_candidates(keyword: list[ChunkCandidate], vector: list[ChunkCandidate], chunk_type_boosts) -> list[MergedChunk]` implementing the formula from [doc 7 § 8](../../architecture/07-retrieval-behavior.md#8-merging-keyword-and-vector-results):
    - `candidate_score = source_weight * chunk_type_boost * (1 / (rrf_k + rank))`
    - `rrf_k = 60` default (from `Settings`)
    - `keyword_source_weight = 1.0`, `vector_source_weight = 1.0` (configurable)
    - If a chunk appears in both keyword and vector results, sum the scores
  - **Chunk-type boosts** from `Settings` per [doc 7 § 7](../../architecture/07-retrieval-behavior.md#7-chunk-type-behavior):
    - Keyword boosts: `recipe_title=1.40`, `recipe_ingredients=1.20`, `recipe_steps=1.05`, `recipe_summary=1.00`, `recipe_full=0.95`
    - Vector boosts: `recipe_summary=1.20`, `recipe_full=1.10`, `recipe_steps=1.00`, `recipe_ingredients=0.95`, `recipe_title=0.90`
- **`src/rag_recipes/retrieval/group.py`**:
  - `group_by_item(merged: list[MergedChunk]) -> list[ItemResult]` per [doc 7 § 9](../../architecture/07-retrieval-behavior.md#9-grouping-by-knowledgeitem):
    - `item_score = best_chunk_score + supporting_chunk_bonus`
    - `supporting_chunk_bonus = 0.05 * (distinct matched chunk types - 1)`, capped at `0.15`
- **`src/rag_recipes/retrieval/search.py`** — facade:
  - `search(request: SearchRequest) -> SearchResult`:
    - Normalize query
    - Build filters
    - Branch on `mode`:
      - `"keyword"` → keyword_search only → group
      - `"vector"` → vector_search only → group
      - `"hybrid"` (default) → both + merge_candidates → group
    - Fetch `KnowledgeItem`, parent `Document`, matched `Chunk`s, and referenced `SourceSpan`s per [doc 7 § 10](../../architecture/07-retrieval-behavior.md#10-fetching-source-data)
    - Build citation labels from source-span locators (e.g., `pdf_page_range` → `"page 42"` or `"pages 42–43"`)
    - Return the result envelope used by the API in Epic 13
- Unit tests for merge formula, chunk-type boost application, grouping bonus

### Acceptance criteria

- [ ] Hybrid search returns merged + grouped item-level results
- [ ] Keyword-only and vector-only modes work as described
- [ ] Chunk-type boosts applied per side (keyword vs vector tables differ)
- [ ] Items with multiple matched chunk types get the supporting-chunk bonus (capped)
- [ ] Citations include readable labels (`"page 42"`, `"pages 42–43"`)
- [ ] Search excludes `needs_review` and `superseded` items by default
- [ ] Integration test: cozy-soup-with-beans query returns expected items with both keyword and vector signals contributing

### Validation

Integration test exercising hybrid mode against a multi-cookbook fixture; manual inspection of intermediate candidate lists and final item rankings to verify the formula's behavior.

---

## Epic-level acceptance criteria

- [ ] Full retrieval logic implemented end-to-end (no API surface yet)
- [ ] Hybrid, keyword, and vector modes all functional
- [ ] Chunk-type boosts applied correctly per side
- [ ] `KnowledgeItem` grouping with supporting-chunk bonus
- [ ] Source-span fetching and citation label construction
- [ ] Default filters enforce ready/active-version/non-superseded
- [ ] Unit tests for merge and grouping formulas
- [ ] Integration test against a real ingested fixture
- [ ] Status in [`EPICS.md`](../EPICS.md) updated; Epic 13 unblocked
