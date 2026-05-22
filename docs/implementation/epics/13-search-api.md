# Epic 13 — Search API & Debug Endpoints

**Status**: Blocked (depends on Epic 12)

## Overview

Expose the retrieval logic from Epic 12 as the public search API, add the knowledge-item detail endpoint, and add the dev-only debug endpoints. After this epic, the backend's API surface from [doc 6](../../architecture/06-backend-api-shape.md) is complete (minus the answer-layer endpoints, which are deferred).

## Architecture references

- [06 — Backend API Shape § 7, 8, 9](../../architecture/06-backend-api-shape.md#7-search) — search endpoint, knowledge-item endpoint, debug endpoints
- [07 — Retrieval Behavior § 11, 12, 13](../../architecture/07-retrieval-behavior.md#11-search-result-construction) — result envelope, debug payload, modes
- [02 — Core Data Model § 4](../../architecture/02-core-data-model.md#4-knowledgeitem) — KnowledgeItem full shape with `structured_data`

## Dependencies

- Epic 12 (retrieval logic)

## Out of scope

- Reranking (deferred)
- Answer-layer endpoints (deferred per [doc 8](../../architecture/08-query-time-answer-layer.md))
- Frontend client (separate repo)

---

## Phase 13.1 — `POST /api/v1/search` with hybrid/keyword/vector modes

**Goal**: Expose retrieval as a public endpoint with the full request/response shape from doc 6.

### What to build

- **`src/rag_recipes/api/routes/search.py`** — `POST /api/v1/search`:
  - Request body per [doc 6 § 7](../../architecture/06-backend-api-shape.md#7-search): `query` (required), `category` (default `recipes`), `subcategory`, `filters` (with `item_type`, `document_ids`, `exclude_needs_review`), `limit`, `include_debug`, `mode`
  - Validates `mode ∈ {hybrid, keyword, vector}` (default `hybrid`)
  - Calls `retrieval.search.search(request)` from Epic 12
  - Builds the response envelope:
    - `query` (echoed normalized form)
    - `results`: list of `KnowledgeItemResult` per [doc 7 § 11](../../architecture/07-retrieval-behavior.md#11-search-result-construction)
      - `type: "knowledge_item_result"`
      - `item`: `id`, `item_type`, `schema`, `title`, `summary`, `status`, `confidence`
      - `display`: `title`, `subtitle` (e.g., `"Simple Thai Food · page 42"`), `snippet`, `badges` (e.g., `"Serves 4"`, `"35 minutes"` derived from `structured_data`)
      - `structured_preview`: `recipe.preview.v1` shape — `schema`, `yield`, `top_ingredients` (top N from `structured_data.ingredients`)
      - `document`: id, title, author
      - `matched_chunks`: `chunk_id`, `chunk_type`, `score`
      - `source_citations`: `source_span_id`, `label`, `locator`
    - `debug`: included only when `include_debug=true` AND `debug_endpoints_enabled` setting is on (dev-only)
- **Pydantic API schemas** in `src/rag_recipes/api/schemas/search.py`
- Validation: empty query returns 400 with `code: "invalid_request"`; invalid mode returns 400
- Integration test:
  - Ingest fixture documents (use the synthetic recipes from `data/fixtures/`)
  - POST a hybrid search; assert the response shape
  - POST a keyword-only and vector-only search; assert different result ordering
  - POST with `include_debug=true` (in dev mode); assert debug payload present
  - POST with `include_debug=true` (in production mode); assert debug absent

### Acceptance criteria

- [ ] `POST /api/v1/search` returns the full response shape from doc 6
- [ ] All three modes work
- [ ] `structured_preview` populated correctly from `structured_data`
- [ ] Citation labels (`"page 42"`, `"pages 42–43"`) generated from PDF locators
- [ ] Debug payload gated by both `include_debug` and `debug_endpoints_enabled`
- [ ] Personal API token enforced
- [ ] Invalid mode / missing query / etc. produce the consistent error shape

### Validation

Integration test described; `curl -H "Authorization: Bearer $TOKEN" -d '{"query":"cozy soup with beans"}' localhost:8000/api/v1/search`.

---

## Phase 13.2 — `GET /api/v1/knowledge-items/{id}` with full structured_data

**Goal**: Return a single `KnowledgeItem` with its full `structured_data` payload (not the truncated `structured_preview` from search).

### What to build

- **`GET /api/v1/knowledge-items/{item_id}`** per [doc 6 § 8](../../architecture/06-backend-api-shape.md#8-get-knowledge-item):
  - Returns the full canonical `KnowledgeItem` including complete `structured_data` (all ingredients, all steps with `source_span_ids`)
  - Includes a small derived `display` block (same shape as in search results)
  - Includes `source_citations` for the item-level source spans
  - Returns 404 with `code: "knowledge_item_not_found"` (add this code to the error registry from Epic 6) if not found or if `status != "ready"` (returning superseded/needs_review items via this endpoint is debatable — recommendation: return them, since this is a direct lookup; superseded items are still useful for audit. Document the choice.)
- **Pydantic API schemas** in `src/rag_recipes/api/schemas/knowledge_items.py`
- Integration test covering happy path, 404, and (per the chosen behavior) returning a superseded/needs_review item

### Acceptance criteria

- [ ] Returns full `structured_data` (no truncation)
- [ ] `display` block populated
- [ ] `source_citations` present
- [ ] 404 with the new error code when not found
- [ ] Personal API token enforced

### Validation

After running an end-to-end ingestion, hit `GET /api/v1/knowledge-items/{id}` for an item that was returned by search and verify the full structured payload matches what was stored.

---

## Phase 13.3 — Debug endpoints (dev-only)

**Goal**: Implement the three debug endpoints per [doc 6 § 9](../../architecture/06-backend-api-shape.md#9-debug-endpoints), gated by the `debug_endpoints_enabled` setting.

### What to build

- **`src/rag_recipes/api/routes/debug.py`** with three endpoints:
  - `GET /api/v1/documents/{document_id}/extraction-runs` — list ExtractionRun rows for the document; query params `source_version`, `status`
  - `GET /api/v1/extraction-runs/{run_id}` — single ExtractionRun including `output_json` and `error_message`
  - `GET /api/v1/documents/{document_id}/source-spans` — list source spans; query params `source_version`, `page_start`, `page_end`. Returns full text (which may be copyrighted) — explicitly gated to dev mode per [doc 6 § 9 Important](../../architecture/06-backend-api-shape.md#9-debug-endpoints)
- **Gating**: a dependency that checks `Settings.debug_endpoints_enabled`; when disabled, all three endpoints return 404 (not 403 — we don't even acknowledge they exist in production)
- Pydantic schemas matching the data model
- Integration tests for each endpoint covering dev-mode-on and dev-mode-off

### Acceptance criteria

- [ ] All three endpoints functional in dev mode
- [ ] All three return 404 when `debug_endpoints_enabled=false`
- [ ] Source-spans endpoint warns in its OpenAPI description about copyright sensitivity
- [ ] Query-param filtering works (source_version, page range)
- [ ] Personal API token still enforced when dev mode is on

### Validation

Toggle `debug_endpoints_enabled` and exercise all three endpoints in both modes.

---

## Epic-level acceptance criteria

- [ ] `POST /api/v1/search` returns the full doc-6 response shape and supports all three modes
- [ ] `GET /api/v1/knowledge-items/{id}` returns full structured_data with citations
- [ ] Three debug endpoints functional in dev mode, hidden in production
- [ ] Search debug payload also dev-only
- [ ] Personal API token enforced on every endpoint
- [ ] Integration tests covering happy paths and error paths for all endpoints
- [ ] Backend API surface for ingestion + retrieval is feature-complete
- [ ] Status in [`EPICS.md`](../EPICS.md) updated; Epic 16 unblocked (also requires Epic 14)
