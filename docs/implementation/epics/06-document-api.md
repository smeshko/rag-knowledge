# Epic 6 — Document Management API

**Status**: In progress (Phase 6.1 done; 6.2/6.3 pending)

## Overview

Implement the document-lifecycle REST endpoints from [doc 6](../../architecture/06-backend-api-shape.md): upload (with duplicate detection), list, get, status, and reprocess. The reprocess endpoint *accepts* requests but doesn't yet kick off real ingestion — that arrives in Epic 8+. This epic also adds the personal API token auth and the consistent error shape.

## Architecture references

- [06 — Backend API Shape](../../architecture/06-backend-api-shape.md) — full endpoint shapes, request/response examples, error shape, auth note
- [02 — Core Data Model § 1, 2](../../architecture/02-core-data-model.md#1-sourceasset) — SourceAsset and Document creation rules
- [03 — PDF Ingestion Pipeline § 1, 2](../../architecture/03-pdf-ingestion-pipeline.md#1-upload-pdf) — what to do at upload time

## Dependencies

- Epic 2 (data model, migrations)
- Epic 4 (LocalFileStorage)

## Out of scope

- Actual ingestion (Epic 8+) — the `Document.status` stays at `queued` after upload
- Stuck-job recovery cron (Epic 7)
- Search/knowledge-item endpoints (Epic 13)
- Debug endpoints (Epic 13)

---

## Phase 6.1 — Upload + duplicate detection

**Goal**: `POST /api/v1/documents` accepts a multipart PDF upload, stores it via `LocalFileStorage`, creates `SourceAsset` + `Document(status=queued)`, and returns the response shape from doc 6.

### What to build

- **`src/rag_recipes/api/routes/documents.py`** — `POST /api/v1/documents`:
  - Multipart form: `file` (required), `category` (default `"recipes"`), `subcategory` (optional), `title` (optional), `author` (optional), `language` (optional)
  - Validates file is a PDF (content-type and magic bytes)
  - Computes `content_hash` (SHA-256 of bytes) before storing
  - **Duplicate detection**: if a `SourceAsset` with the same `content_hash` exists, return the existing `Document` instead of creating a new one (per doc 6 § Duplicate upload behavior)
  - Otherwise:
    - Generate `asset_id`
    - Store via `FileStorageProvider.put_object(key=f"source-assets/{asset_id}/original.pdf", ...)`
    - Insert `SourceAsset` row with `upload_status="uploaded"`
    - Insert `Document` row with `status="queued"`, `active_source_version=null`
  - Return the response shape from [doc 6 § 2](../../architecture/06-backend-api-shape.md#response)
- **Pydantic API schemas** in `src/rag_recipes/api/schemas/documents.py`:
  - `DocumentResponse`, `IngestionStatusResponse`, `UploadResponse`
- **Repository / service layer** in `src/rag_recipes/storage/repositories/documents.py` — query helpers needed here (find by content_hash, insert, etc.)

### Acceptance criteria

- [x] Upload succeeds end-to-end: file appears in `data/storage/`, rows in `source_assets` and `documents`, response matches doc 6 shape
- [x] Duplicate upload returns the existing document (no new rows, file not re-stored)
- [x] Non-PDF upload returns 415 with `code: "unsupported_file_type"`
- [x] Multipart fields validated; missing required fields return 400
- [x] Integration test against real Postgres + `LocalFileStorage`

### Validation

`curl -F "file=@sample.pdf" -F "category=recipes" -F "title=Test" localhost:8000/api/v1/documents` returns the expected JSON; second upload of the same file returns the same `document.id`.

---

## Phase 6.2 — List, Get, Status endpoints

**Goal**: `GET /documents`, `GET /documents/{id}`, `GET /documents/{id}/status` return data per doc 6.

### What to build

- **`GET /api/v1/documents`** with query filters: `category`, `status`, `source_type`. Returns list shape from [doc 6 § 3](../../architecture/06-backend-api-shape.md#3-list-documents). Add pagination scaffold (`limit`, `offset`) even if not exposed externally yet.
- **`GET /api/v1/documents/{document_id}`** returns metadata + `counts` block (source_spans, knowledge_items, ready_items, needs_review_items, chunks) per [doc 6 § 4](../../architecture/06-backend-api-shape.md#4-get-document). Counts come from aggregate queries.
- **`GET /api/v1/documents/{document_id}/status`** returns the polling shape from [doc 6 § 5](../../architecture/06-backend-api-shape.md#5-get-ingestion-status):
  - `document_id`, `status`, `active_source_version`, `current_source_version`
  - `progress` block: `stage`, `message`, `pages_total`, `pages_processed`
  - `terminal: bool` derived from status
  - For now, `pages_total` / `pages_processed` may be `null` — populated in Epic 8
- 404 with `code: "document_not_found"` when ID doesn't exist
- Integration tests for all three endpoints

### Acceptance criteria

- [x] All three endpoints return the documented shapes
- [x] Filters on list endpoint work correctly
- [x] Aggregate counts in `GET /{id}` match real row counts
- [x] 404 error shape matches the consistent error format from doc 6
- [x] Status endpoint correctly identifies terminal statuses (`ready`, `needs_review`, `failed`)

### Validation

`curl localhost:8000/api/v1/documents?category=recipes` and the three other variants — all return correctly-shaped JSON.

---

## Phase 6.3 — Reprocess scaffold + auth + error shape

**Goal**: `POST /documents/{id}/reprocess` accepts the request shape from doc 6 (no real ingestion behind it yet); personal API token auth gates all endpoints; consistent error shape across the API.

### What to build

- **`POST /api/v1/documents/{document_id}/reprocess`**:
  - Request body: `{ mode: "auto" | "reuse_source_spans" | "new_source_version", reason?: string }`
  - Validates `mode` and that the document exists
  - Validates the transition is legal (only from `ready`, `needs_review`, `failed`)
  - Sets `Document.status = "queued"` and records the requested mode somewhere (e.g., a column or a small `reprocess_requests` table — designer's choice; document it). Pick whichever requires fewer migrations now and revisit in Epic 11.
  - Returns the response shape from [doc 6 § 6](../../architecture/06-backend-api-shape.md#response-2): `previous_active_source_version`, `current_source_version`
  - **Does not actually start ingestion** — Epic 8 wires the actual job dispatch
- **Personal API token auth** per [doc 6 § Authentication Note](../../architecture/06-backend-api-shape.md#authentication-note):
  - Dependency that checks `Authorization: Bearer <token>` against `Settings.personal_api_token`
  - Returns 401 with `code: "unauthorized"` on missing/invalid token
  - In development mode (configurable), allow auth bypass for ergonomics — document the flag clearly
  - Applied to all routes via a router-level dependency
- **Consistent error shape** per [doc 6 § Error Shape](../../architecture/06-backend-api-shape.md#error-shape):
  - Centralized exception handlers translating common exceptions to `{ error: { code, message, details } }`
  - Initial error codes per doc 6: `invalid_request`, `unsupported_file_type`, `duplicate_source_asset`, `document_not_found`, `ingestion_already_running`, `internal_error`
  - Make this a module so later epics add codes consistently

### Acceptance criteria

- [ ] Reprocess endpoint accepts all three modes and sets `Document.status = "queued"`
- [ ] Reprocess from a non-terminal status returns 409 with `code: "ingestion_already_running"`
- [ ] All endpoints require a valid bearer token (except `/health`)
- [ ] Auth-bypass flag works in dev mode and is documented
- [ ] Errors across all endpoints use the consistent shape
- [ ] Integration tests cover happy path + 401 + 404 + 409

### Validation

`curl` exercises all auth paths; reprocess from `ready` document succeeds and flips `status` to `queued`; reprocess from a `queued` document returns 409 with the right code.

---

## Epic-level acceptance criteria

- [ ] Five endpoints functional: `POST /documents`, `GET /documents`, `GET /documents/{id}`, `GET /documents/{id}/status`, `POST /documents/{id}/reprocess`
- [ ] Duplicate-content uploads return the existing document
- [ ] Personal API token auth enforced on all non-health routes
- [ ] Consistent error shape across all endpoints
- [ ] Reprocess endpoint flips status to `queued` (real dispatch lives in Epic 8/11)
- [ ] Integration test suite covering happy paths and key error paths
- [ ] Status in [`EPICS.md`](../EPICS.md) updated; Epic 8 unblocked (in combination with Epics 4, 7)
