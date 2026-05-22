# Epic 4 — Storage & PDF Extraction

**Status**: Blocked (depends on Epic 3)

## Overview

Ship the two simpler real provider implementations: `LocalFileStorage` (writes/reads bytes on the local filesystem under a configured root) and `PyMuPdfExtractor` (extracts per-page text from PDFs using PyMuPDF / `fitz`). Both implementations re-use the contract tests written in Epic 3 to confirm interface compliance.

## Architecture references

- [11 — Configuration and Providers § 1](../../architecture/11-configuration-and-providers.md#1-filestorageprovider) — file storage interface, provider/key model
- [11 — Configuration and Providers § 2](../../architecture/11-configuration-and-providers.md#2-pdftextextractor) — PDF extractor interface, future OCR hook
- [02 — Core Data Model § Storage fields](../../architecture/02-core-data-model.md#storage-fields) — `storage_provider` + `storage_key` semantics
- [13 — Implementation Decisions, topic 3](../../architecture/13-implementation-decisions.md#3-pdf-text-extraction-library) — PyMuPDF choice + AGPL caveat

## Dependencies

- Epic 3 (interfaces + contract tests)

## Out of scope

- S3-compatible file storage (deferred; provider interface keeps the door open)
- OCR fallback for scanned PDFs (deferred; `PdfTextExtractor` interface keeps the door open)
- Wiring providers into the API (Epic 6 wires file storage; Epic 8 wires PDF extraction)

---

## Phase 4.1 — LocalFileStorage implementation

**Goal**: A real `FileStorageProvider` that reads/writes to a configurable local directory and passes all contract tests from Epic 3.

### What to build

- **`src/rag_recipes/providers/file_storage/local.py`** — `LocalFileStorage`:
  - Constructor takes `root_path: Path` from config (`local_storage_root` setting)
  - `put_object(key, data, content_type)`:
    - Resolves to `root_path / key`
    - Creates parent directories as needed
    - Writes bytes atomically (temp file + rename)
    - Returns `StoredObject` with `provider="local"`, `key`, `size`, optional metadata
  - `get_object(key)` — reads bytes; raises `FileNotFoundError` if missing
  - `exists(key)` — checks file existence
  - `delete_object(key)` — removes file; no-op if missing (configurable)
  - **Path safety**: reject keys with `..` or absolute paths; all resolved paths must remain under `root_path`
- Wire `LocalFileStorage` as the registered implementation when `file_storage_provider == "local"` in config
- Contract tests from Epic 3 parameterized to run against `LocalFileStorage` using a `tmp_path` fixture

### Acceptance criteria

- [ ] `LocalFileStorage` implements every method on `FileStorageProvider`
- [ ] Path traversal (`../foo`, `/abs/path`) raises a clear error
- [ ] Concurrent `put_object` calls do not produce torn writes (atomic temp+rename verified)
- [ ] Contract tests from Epic 3 pass against `LocalFileStorage`
- [ ] mypy passes

### Validation

`make test-unit` and `make test-integration` — contract tests run against the Fake and against `LocalFileStorage` using a temporary directory.

---

## Phase 4.2 — PyMuPdfExtractor implementation

**Goal**: A real `PdfTextExtractor` using PyMuPDF that produces per-page `PdfPageText` records suitable for `SourceSpan` creation in Epic 8.

### What to build

- **`src/rag_recipes/providers/pdf_extractor/pymupdf.py`** — `PyMuPdfExtractor`:
  - `extract_pages(file: bytes) -> list[PdfPageText]`:
    - Opens the PDF via `fitz.open(stream=file, filetype="pdf")`
    - For each page, extracts text via `page.get_text("text")` (basic mode for selectable text per doc 3)
    - Sets `page_number` (1-indexed), `text`, `extraction_method="embedded_text"`, `confidence=None`
    - Closes the document cleanly even on error
  - Honors the `pdf_min_text_chars_for_page` setting from doc 11: if a page has fewer characters than the threshold, mark it as suspicious in `confidence` or via a separate flag (designer's choice — surface it so later OCR work can find these pages)
  - Returns pages in document order
- Contract tests from Epic 3 parameterized to run against `PyMuPdfExtractor` using a small committed synthetic PDF fixture in `data/fixtures/pdfs/` (e.g., a 3-page recipe-like PDF generated for testing)
- Create at least one synthetic test PDF as a committed fixture; document how it was generated in `data/fixtures/pdfs/README.md`

### Acceptance criteria

- [ ] `PyMuPdfExtractor` implements `extract_pages` and returns 1-indexed pages
- [ ] Output is deterministic for the same input bytes
- [ ] Pages with very little text are flagged (per `pdf_min_text_chars_for_page`)
- [ ] Contract tests from Epic 3 pass against `PyMuPdfExtractor` with a real synthetic PDF
- [ ] Document/resource handles are properly closed (no leaks; verified in a test that opens many PDFs)
- [ ] License note (AGPL-3.0) referenced in the module docstring per [doc 13 § 3](../../architecture/13-implementation-decisions.md#3-pdf-text-extraction-library)

### Validation

`make test` — all contract tests pass; manual smoke test via a small Python script that loads a real PDF (e.g., a public-domain recipe page) and prints the extracted text per page.

---

## Epic-level acceptance criteria

- [ ] `LocalFileStorage` and `PyMuPdfExtractor` exist and pass all Epic-3 contract tests
- [ ] Path safety enforced in `LocalFileStorage`
- [ ] At least one committed synthetic PDF fixture exists for tests
- [ ] AGPL-3.0 note present in PyMuPDF wrapper
- [ ] mypy passes in strict mode against `providers/file_storage/` and `providers/pdf_extractor/`
- [ ] Status in [`EPICS.md`](../EPICS.md) updated; Epic 6 and Epic 8 unblocked (for storage); Epic 8 fully unblocked once Epic 7 is also done
