# Epic 11 — Reprocessing Modes

**Status**: Blocked (depends on Epic 10)

## Overview

Make the `POST /documents/{id}/reprocess` endpoint from Epic 6 actually do something. Implement the two real reprocess flows — `reuse_source_spans` (downstream-only rerun, same `source_version`) and `new_source_version` (re-extract PDF text into a new version) — and the `auto` selector that picks between them. Ensure `Document.active_source_version` only flips after a re-extraction succeeds, so old `KnowledgeItem.source_span_ids` references never break.

## Architecture references

- [02 — Core Data Model § Source span stability, Superseding rule](../../architecture/02-core-data-model.md#source-span-stability) — source_version semantics, supersede
- [03 — PDF Ingestion Pipeline § Stuck-job recovery / retry](../../architecture/03-pdf-ingestion-pipeline.md#stuck-job-recovery) — retry rule
- [06 — Backend API Shape § 6](../../architecture/06-backend-api-shape.md#6-reprocess-document) — reprocess endpoint shape and mode meanings

## Dependencies

- Epic 10 (full ingestion pipeline through to `ready`)

## Out of scope

- Manual review UI for `needs_review` items
- Item-level superseding (MVP uses document-level only; see [doc 2 § Superseding rule](../../architecture/02-core-data-model.md#superseding-rule))
- Re-embedding workflows triggered by embedding model changes (covered ad-hoc by Epic 10's `re_embed_for_model` helper)

---

## Phase 11.1 — Reuse-source-spans mode

**Goal**: Re-run extraction → validation → chunking → embedding using the existing source spans, without re-extracting PDF text. Used after prompt/model changes or downstream failures.

### What to build

- **`POST /documents/{id}/reprocess`** with `mode="reuse_source_spans"`:
  - Validates document is in a terminal state (`ready`, `needs_review`, `failed`)
  - Sets `Document.status = "queued"`
  - Records the reprocess request (mode + reason) — extend the storage choice made in Epic 6 Phase 6.3
  - Enqueues `process_document` job with a `reuse_source_spans=True` flag and the existing `source_version` (same as `Document.active_source_version` if set, otherwise the highest existing source_version)
- **Pipeline behavior** when `reuse_source_spans=True`:
  - Skip PDF text extraction; load existing `SourceSpan` rows for the supplied version
  - Re-run all downstream stages (windowing, LLM extraction, validation, dedup, chunking, embedding, indexing)
  - **Important**: don't delete the old `KnowledgeItem` / `Chunk` / `ChunkEmbedding` rows. Old `KnowledgeItem`s get superseded at the end (per Epic 9 § 9.4 dedup phase); old chunks/embeddings remain referenced by superseded items (per [doc 2 § Superseded parent policy](../../architecture/02-core-data-model.md#superseded-parent-policy)) and are filtered out of retrieval rather than deleted.
  - `Document.active_source_version` does *not* change (same version)
- Integration test: ingest a document, change the prompt version in `Settings`, call reprocess with `reuse_source_spans`, verify new `KnowledgeItem`s are created with the *same* `source_version`, old ones marked `superseded`, retrieval returns the new ones only

### Acceptance criteria

- [ ] Reprocess endpoint with `reuse_source_spans` re-runs downstream pipeline on existing spans
- [ ] `source_version` unchanged
- [ ] Old `KnowledgeItem`s marked `superseded` after the new run reaches `ready`
- [ ] Old chunks/embeddings remain in DB but are excluded from retrieval (verified once Epic 12 lands; for now, verified via direct DB queries)
- [ ] `Document.active_source_version` unchanged
- [ ] Failed reprocess (mid-extraction) leaves the old `ready` items as the active set (active version unchanged)

### Validation

Integration test described above; manually trigger a reprocess against a `ready` document via `curl` and verify behavior in DB.

---

## Phase 11.2 — New-source-version mode + auto selector + active version handoff

**Goal**: Re-extract PDF text into a new `source_version` (used after PDF extractor changes or OCR additions); flip `Document.active_source_version` only after the new run reaches `ready`; the `auto` mode picks between reuse and new-version based on what changed.

### What to build

- **`POST /documents/{id}/reprocess`** with `mode="new_source_version"`:
  - Validates terminal state
  - Computes `new_version = max(existing source_versions) + 1`
  - Sets `Document.status = "queued"`
  - Enqueues `process_document` with the new `source_version` (not flagged as reuse)
  - **Critical**: `Document.active_source_version` remains pointed at the old version during the run
- **Pipeline behavior** for new source version:
  - Run PDF text extraction → create new `SourceSpan` rows with the new `source_version`
  - Run extraction → validation → dedup → chunking → embedding → indexing as normal
  - At the end, if status reaches `ready`:
    - Update `Document.active_source_version = new_version`
    - Mark all `KnowledgeItem`s from the prior `active_source_version` as `superseded`
  - If status ends as `needs_review` or `failed`: `active_source_version` stays unchanged (old version remains active), new-version artifacts (spans, items, chunks) remain in DB but are not active
- **`mode="auto"`** selector logic (lives in the reprocess endpoint, not the pipeline):
  - If the latest `ExtractionRun` for this document used the current `prompt_version`/`schema_version` and the failure was downstream → `reuse_source_spans`
  - If `Settings.pdf_text_extractor` differs from what produced the existing source spans (track this via a small marker on `SourceSpan` or via the source_version's extraction metadata — designer's choice; document it) → `new_source_version`
  - If unclear → default to `new_source_version` (the safer choice; over-extraction is fine, missing a needed re-extract isn't)
- **Response** populates `previous_active_source_version` and `current_source_version` per [doc 6](../../architecture/06-backend-api-shape.md#response-2)

### Acceptance criteria

- [ ] `new_source_version` mode creates new `SourceSpan` rows under a new version
- [ ] `Document.active_source_version` only flips on successful completion (`ready`)
- [ ] On failure mid-new-version-run, `active_source_version` stays on the old version
- [ ] `mode="auto"` correctly picks between reuse and new-version based on what changed
- [ ] Multiple `source_version`s coexist in `source_spans` table; queries filter correctly by version
- [ ] Integration tests cover: successful new-version → version flip; failed new-version → no flip; auto-mode dispatch decisions

### Validation

Integration test: ingest → wait for ready → reprocess with `new_source_version` → verify two versions of source spans, one flipped active, supersede applied. Then simulate failure during a new-version run and verify the rollback (active version stays on the old one).

---

## Epic-level acceptance criteria

- [ ] All three reprocess modes (`auto`, `reuse_source_spans`, `new_source_version`) work
- [ ] `Document.active_source_version` only flips on successful completion of a new-version run
- [ ] Old items get `superseded`; old chunks/embeddings remain for audit but excluded from retrieval
- [ ] `mode="auto"` makes the correct dispatch decision under common conditions
- [ ] Reprocess of a `failed` document re-runs cleanly
- [ ] Reprocess of a `ready` document with no changes still works (idempotent on the data side; new ExtractionRun rows recorded for audit)
- [ ] Status in [`EPICS.md`](../EPICS.md) updated
