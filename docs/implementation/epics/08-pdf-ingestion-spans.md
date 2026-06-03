# Epic 8 — PDF Ingestion: Text & SourceSpans

**Status**: Done

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
  - **PyMuPDF concurrency guardrail** (deferred here from Phase 4.2's review): `PyMuPdfExtractor.extract_pages` dispatches to the default thread executor via `asyncio.to_thread` with no serialization, and PyMuPDF runs MuPDF in single-threaded mode (`reinit_singlethreaded()` at import). Concurrent extractions from multiple worker threads risk native crashes or silent extraction corruption. Before this provider runs under real job concurrency, serialize PyMuPDF access — a module-level lock or a dedicated single-worker executor in `PyMuPdfExtractor`, or a bounded process pool — and add a concurrent-extraction regression test (multiple `extract_pages` calls in flight against the synthetic fixture). See the Phase 4.2 archived `REVIEW.md` (round-1 #1 / round-2 #1).
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

- [x] After `POST /documents`, the worker picks up the job, extracts text, and writes one `SourceSpan` per page
- [x] `source_version = 1` on initial ingestion
- [x] `locator_hash` and `text_hash` populated
- [x] Uniqueness constraint enforced (no duplicate spans for the same page/version)
- [x] Failure during extraction transitions Document to `failed` with a clear error
- [x] Langfuse session ID set to `document_id`
- [x] PyMuPDF access is serialized (lock / single-worker executor / process pool) before running under job concurrency, with a concurrent-extraction regression test (deferred from Phase 4.2)
- [x] Integration test using a committed synthetic PDF fixture

### Validation

Upload a small synthetic PDF; poll `GET /documents/{id}/status` until status moves through `extracting_text` and `creating_source_spans`; query `source_spans` to verify one row per page with correct `locator` and hashes.

---

## Phase 8.2 — Status transitions, progress reporting, integration test

**Goal**: The status polling endpoint reflects accurate progress during ingestion, and an integration test exercises the full upload → spans created path.

> **Forward-note from Epic 6 Phase 6.2.** `GET /documents/{id}/status` currently mirrors `current_source_version = active_source_version` as a placeholder (both `null` at `queued`; the span/run tables are empty pre-Epic-8 so there is no in-progress version to point at). Once 8.2 writes versioned `source_span`s and real progress, replace the mirror with the actual in-progress version per [doc 6 §5](../../architecture/06-backend-api-shape.md#5-get-ingestion-status) — e.g. `active_source_version: null`, `current_source_version: 1` mid-ingestion. Update or replace the 6.2 mirror test (`tests/integration/test_documents_status.py::test_current_source_version_mirrors_active_when_set`) accordingly so the 6.2 placeholder isn't frozen as a permanent contract.

> **Forward-note from Epic 6 Phase 6.3.** `POST /documents/{id}/reprocess` returns `current_source_version` mirroring `previous_active_source_version` (the value of `active_source_version` at the time of the request) for the same reason — no new source version exists pre-Epic-8. Once 8.2 dispatches the actual job and writes a new versioned span set, return the real new version per [doc 6 §6](../../architecture/06-backend-api-shape.md#6-trigger-reprocessing) (e.g. `previous_active_source_version: 1`, `current_source_version: 2`). The reprocess happy-path tests in `tests/integration/test_documents_reprocess.py` currently assert the mirror; update them when the real version arrives.

### What to build

- Populate `progress.pages_total` and `progress.pages_processed` on `GET /documents/{id}/status` while ingestion runs:
  - Either by writing partial progress into a Redis key (keyed by `document_id`) the job updates as it processes, or by counting `source_spans` rows for the current version (simpler; designer's choice — document it)
- **Resolved (Phase 8.2): Option A.** The job leaves the document at `creating_source_spans` after extraction; Epic 9's `process_extraction_run` job transitions `creating_source_spans → extracting_items` when it lands. Because a successfully-extracted document is *not* stuck, Phase 8.1's review (`fix(ingestion): exempt creating_source_spans from stuck-job sweep`) added `creating_source_spans` to the stuck-job cron's `_SWEEP_EXEMPT_STATUSES`: the document rests at `creating_source_spans` indefinitely until Epic 9 owns it, rather than being marked `failed` after the timeout. (The exemption is to be removed once `process_extraction_run` exists, at which point a document wedged at `creating_source_spans` with item-extraction stalled is genuinely stuck.) Rationale: state fields should reflect what actually happened (Option A), not what is supposed to happen next (Option B). See [`.claude/plans/epic-08-phase-8-2-status-progress/DECISIONS.md` §1](../../.claude/plans/epic-08-phase-8-2-status-progress/DECISIONS.md) and `tests/integration/test_sweep_stuck_jobs.py::test_sweep_exempts_creating_source_spans_handoff_state`.
- Integration test pathway:
  - Use `FakeLLMProvider` and `FakePdfExtractor` (or real PyMuPDF against a committed synthetic PDF)
  - Upload via the API
  - Poll status until `creating_source_spans` is reached
  - Verify span rows
  - Verify Langfuse trace exists (with tracing enabled)

### Acceptance criteria

- [x] `progress.pages_total` and `progress.pages_processed` populated during/after extraction
- [x] Status accurately reflects extraction stage during the run
- [x] Integration test runs end-to-end from `POST /documents` to span rows in DB
- [x] Failure mid-extraction (simulated via a Fake that raises) marks document `failed` with the error message captured
- [x] No orphaned spans on failure (cleanup or leave-for-retry rule documented — for initial ingestion, leaving them is fine since `source_version=1` won't be reused for the same document if it failed; doc 2 source-span retry rule applies)

### Validation

`make test-integration` covers the new test; manually upload a multi-page PDF and verify status progresses correctly.

---

## Epic-level acceptance criteria

- [x] Uploading a PDF triggers an arq job that extracts text and writes per-page SourceSpans
- [x] `source_version = 1` on initial ingestion; uniqueness constraint respected
- [x] Status accurately reflects progress through `extracting_text` and `creating_source_spans`
- [x] Failures transition the document to `failed` with a clear error
- [x] Langfuse session ID propagated for trace grouping (when enabled)
- [x] Integration test exercises upload → spans end-to-end
- [x] Status in [`EPICS.md`](../EPICS.md) updated; Epic 9 unblocked
