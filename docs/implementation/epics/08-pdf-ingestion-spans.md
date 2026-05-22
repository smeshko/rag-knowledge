# Epic 8 — PDF Ingestion: Text & SourceSpans

**Status**: Blocked (depends on Epics 4, 6, 7)

## Overview

Wire the first real ingestion job: when a Document is created via `POST /documents`, an arq job reads the PDF bytes via `FileStorageProvider`, extracts per-page text via `PdfTextExtractor`, and creates one `SourceSpan` per page with the proper `source_version`. Status transitions through `extracting_text` → `creating_source_spans` and lands at a state ready for Epic 9's LLM extraction step.

## Architecture references

- [03 — PDF Ingestion Pipeline § 1–4](../../architecture/03-pdf-ingestion-pipeline.md#1-upload-pdf) — upload, document creation, text extraction, span creation
- [02 — Core Data Model § 3](../../architecture/02-core-data-model.md#3-sourcespan) — SourceSpan immutability, source_version semantics, hash fields, retry rule
- [11 — Configuration and Providers § 2](../../architecture/11-configuration-and-providers.md#2-pdftextextractor) — extractor interface and the `pdf_min_text_chars_for_page` setting

## Dependencies

- Epic 4 (`LocalFileStorage`, `PyMuPdfExtractor`)
- Epic 6 (Document upload endpoint that creates the Document)
- Epic 7 (arq worker + status transition helpers)

## Out of scope

- LLM extraction of recipes (Epic 9)
- Chunk creation and embeddings (Epic 10)
- The reprocess "new source version" flow's text re-extraction (Epic 11 — uses the same machinery here)

---

## Phase 8.1 — PDF extraction job

**Goal**: An arq job takes a `document_id`, reads the file via storage, extracts per-page text via the PDF extractor, and writes `SourceSpan` rows.

### What to build

- **`src/rag_recipes/ingestion/pipeline/pdf_text.py`** — pure function `extract_and_persist_spans(document_id, source_version) -> int` (returns number of spans created):
  - Loads the `Document` and its parent `SourceAsset`
  - Reads PDF bytes via `FileStorageProvider.get_object(source_asset.storage_key)`
  - Calls `PdfTextExtractor.extract_pages(pdf_bytes)`
  - For each page, computes `locator = {"type": "pdf_page_range", "page_start": N, "page_end": N}`, `locator_hash` (stable hash of normalized JSON), `text_hash` (hash of text)
  - Inserts `SourceSpan` rows with the supplied `source_version`
  - Handles the uniqueness constraint `(document_id, source_version, locator_hash)` — duplicates indicate a bug or a retry that shouldn't recreate; raise a clear error
- **arq job wrapper** `process_document(ctx, document_id)` in `ingestion/jobs.py`:
  - For initial ingestion: `source_version = 1`
  - Transitions: `queued → extracting_text → creating_source_spans → [next stage placeholder]`
  - On exception, transitions to `failed` with the error captured
  - Sets a Langfuse session ID equal to `document_id` so all subsequent LLM/embedding calls in this run are grouped
- **`POST /documents`** in Epic 6 enqueues this job after the Document row is created (this completes the upload → background-job wiring)
- Edge cases:
  - Suspicious pages (low text count flagged by the extractor) still get a `SourceSpan`, but the suspicion flag should surface somewhere (e.g., on the span's `locator` JSON, or in logs)
  - Empty PDFs cause `failed` with a clear reason

### Acceptance criteria

- [ ] After `POST /documents`, the worker picks up the job, extracts text, and writes one `SourceSpan` per page
- [ ] `source_version = 1` on initial ingestion
- [ ] `locator_hash` and `text_hash` populated
- [ ] Uniqueness constraint enforced (no duplicate spans for the same page/version)
- [ ] Failure during extraction transitions Document to `failed` with a clear error
- [ ] Langfuse session ID set to `document_id`
- [ ] Integration test using a committed synthetic PDF fixture

### Validation

Upload a small synthetic PDF; poll `GET /documents/{id}/status` until status moves through `extracting_text` and `creating_source_spans`; query `source_spans` to verify one row per page with correct `locator` and hashes.

---

## Phase 8.2 — Status transitions, progress reporting, integration test

**Goal**: The status polling endpoint reflects accurate progress during ingestion, and an integration test exercises the full upload → spans created path.

### What to build

- Populate `progress.pages_total` and `progress.pages_processed` on `GET /documents/{id}/status` while ingestion runs:
  - Either by writing partial progress into a Redis key (keyed by `document_id`) the job updates as it processes, or by counting `source_spans` rows for the current version (simpler; designer's choice — document it)
- Stop the job from leaving the document in `creating_source_spans` forever — Epic 9 picks up here. For this epic, after spans are created, transition the document to `extracting_items` (the next stage per doc 2), and immediately to a placeholder "spans-ready" terminal-but-not-really state. Two options (pick one and document):
  - **Option A**: leave `Document.status = "creating_source_spans"` and let Epic 9 take over the same job
  - **Option B**: transition to `extracting_items` and let the stuck-job cron handle it if Epic 9 is not yet implemented (acceptable since the cron will mark it failed after timeout — but Epic 9 should land before this matters in practice)
- Integration test pathway:
  - Use `FakeLLMProvider` and `FakePdfExtractor` (or real PyMuPDF against a committed synthetic PDF)
  - Upload via the API
  - Poll status until `creating_source_spans` is reached
  - Verify span rows
  - Verify Langfuse trace exists (with tracing enabled)

### Acceptance criteria

- [ ] `progress.pages_total` and `progress.pages_processed` populated during/after extraction
- [ ] Status accurately reflects extraction stage during the run
- [ ] Integration test runs end-to-end from `POST /documents` to span rows in DB
- [ ] Failure mid-extraction (simulated via a Fake that raises) marks document `failed` with the error message captured
- [ ] No orphaned spans on failure (cleanup or leave-for-retry rule documented — for initial ingestion, leaving them is fine since `source_version=1` won't be reused for the same document if it failed; doc 2 source-span retry rule applies)

### Validation

`make test-integration` covers the new test; manually upload a multi-page PDF and verify status progresses correctly.

---

## Epic-level acceptance criteria

- [ ] Uploading a PDF triggers an arq job that extracts text and writes per-page SourceSpans
- [ ] `source_version = 1` on initial ingestion; uniqueness constraint respected
- [ ] Status accurately reflects progress through `extracting_text` and `creating_source_spans`
- [ ] Failures transition the document to `failed` with a clear error
- [ ] Langfuse session ID propagated for trace grouping (when enabled)
- [ ] Integration test exercises upload → spans end-to-end
- [ ] Status in [`EPICS.md`](../EPICS.md) updated; Epic 9 unblocked
