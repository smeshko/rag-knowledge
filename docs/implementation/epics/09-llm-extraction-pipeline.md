# Epic 9 — LLM Extraction Pipeline

**Status**: In progress

## Overview

The heart of ingestion. Take per-page `SourceSpan` records, group them into overlapping page windows, call the LLM to extract recipe `KnowledgeItem` candidates with the `recipe.v1` schema, record every call as an `ExtractionRun`, validate hard and soft, deduplicate overlapping candidates, and persist `KnowledgeItem` rows with the right status (`ready` / `needs_review`) plus document-level superseding for re-extractions.

This is the largest epic by surface area. It has four phases that should land sequentially.

## Architecture references

- [04 — LLM-Assisted Recipe Extraction](../../architecture/04-llm-assisted-recipe-extraction.md) — entire document; this epic implements doc 4 end-to-end
- [03 — PDF Ingestion Pipeline § 5–9](../../architecture/03-pdf-ingestion-pipeline.md#5-create-page-windows) — page windows, extraction, dedup
- [02 — Core Data Model § 4, 7](../../architecture/02-core-data-model.md#4-knowledgeitem) — KnowledgeItem and ExtractionRun shapes, superseding rule
- [11 — Configuration and Providers § 3](../../architecture/11-configuration-and-providers.md#3-llmprovider) — caching by input_hash, failure categories

## Dependencies

- Epic 5 (`OpenAILLMProvider`)
- Epic 8 (SourceSpans created and ready)

## Out of scope

- Chunk creation (Epic 10)
- Embedding generation (Epic 10)
- Reprocess flows (Epic 11)
- The `judge_alignment` workflow (Epic 15)

---

## Phase 9.1 — Page windowing + window input formatting

**Goal**: Given a document's source spans for the current version, produce overlapping page windows formatted for LLM input per doc 4.

### What to build

- **`src/rag_recipes/ingestion/pipeline/windows.py`**:
  - `build_windows(spans: list[SourceSpan], window_size: int, overlap: int) -> list[Window]` where `Window` carries the contiguous list of spans
  - Defaults from `Settings`: `pdf_window_size_pages=3`, `pdf_overlap_pages=1` per [doc 3 § 5](../../architecture/03-pdf-ingestion-pipeline.md#5-create-page-windows)
  - `format_window_for_llm(window: Window) -> str` produces the `[SOURCE_SPAN span_xyz | PDF page N]` text format from [doc 4](../../architecture/04-llm-assisted-recipe-extraction.md#sourcespan-input-convention)
  - `compute_input_hash(window: Window, prompt_version: str, schema_version: str) -> str` — stable hash used for caching and ExtractionRun dedup
- Unit tests:
  - Window sizes match config
  - Overlap creates the expected sequence (1-3, 3-5, 5-7, ...)
  - End-of-document edge cases (last window may be shorter)
  - Format produces the expected text shape

### Acceptance criteria

- [x] Windows respect configured size and overlap
- [x] Format matches doc 4 exactly (so the prompt can rely on it)
- [x] `compute_input_hash` is deterministic across runs
- [x] Unit tests cover normal, edge, and overlap-0 cases

### Validation

`make test-unit` against `tests/unit/ingestion/test_windows.py`.

---

## Phase 9.2 — LLM call + ExtractionRun recording

**Goal**: For each window, make the LLM call and record a fully-populated `ExtractionRun` row with the correct status (`running` → `success` | `failed` | `rejected`).

### What to build

- **`src/rag_recipes/ingestion/pipeline/extraction.py`**:
  - `recipe.v1` Pydantic schema covering the full output shape in [doc 4 § Strict Output Shape](../../architecture/04-llm-assisted-recipe-extraction.md#strict-output-shape):
    - Top-level: `items: list[ExtractedRecipe]`
    - Each `ExtractedRecipe`: `item_type`, `title`, `summary`, `body_text`, `source_span_ids`, `structured_data: RecipeV1StructuredData`, `confidence: RecipeConfidence`, `warnings: list[str]`
    - `RecipeV1StructuredData`: `schema="recipe.v1"`, `yield`, times, `ingredients_text`, `ingredients: list[ExtractedIngredient]`, `steps_text`, `steps: list[ExtractedStep]`
    - Ingredient/step models with the field meanings from [doc 4](../../architecture/04-llm-assisted-recipe-extraction.md#ingredient-field-meaning)
    - Confidence models with the canonical five-field shape from [doc 2](../../architecture/02-core-data-model.md#ingredient-confidence)
  - **Prompt template** — versioned (e.g., `recipe-extraction-v1`), stored under `src/rag_recipes/ingestion/prompts/recipe_extraction_v1.md`. Content per doc 4 guidance: instruct the model to extract recipe candidates, cite `source_span_ids`, return confidence, and use the schema. Versioning is part of the API to the LLM provider.
  - `run_extraction(window, source_version)`:
    - Compute `input_hash`
    - Insert `ExtractionRun` row with status `running`
    - Call `LLMProvider.generate_structured_output(...)` with the `recipe.v1` JSON schema and prompt
    - On technical failure (raised `LLMTechnicalError`): status → `failed`, record error message
    - On successful response: parse into `recipe.v1` Pydantic model
    - On parse/validation failure (Pydantic raises): status → `rejected`, store raw output and error
    - On success: status → `success`, store parsed output
    - Populate `provider`, `model`, `prompt_version`, `schema_version`, `input_source_span_ids`, `input_hash`, `output_json`, `completed_at`
  - **Caching** by `(input_hash, provider, model, prompt_version, schema_version)`: before making a real LLM call, check for a prior successful `ExtractionRun` with matching key; if found, reuse it (still create a new ExtractionRun row pointing at the same parsed output, *or* document a clear policy — recommendation: skip and reuse for cost/latency; new row only if config flag forces).
  - Langfuse session ID set to `document_id`

### Acceptance criteria

- [ ] `recipe.v1` Pydantic model matches doc 4 exactly (field names, nullability, nesting)
- [ ] ExtractionRun written for every LLM attempt with full metadata
- [ ] `success`, `failed`, `rejected` statuses distinguished correctly (failed = transport/system; rejected = output doesn't validate)
- [ ] Cache hit on identical `input_hash` + provider/model/prompt_version/schema_version reuses prior output
- [ ] Langfuse session grouping works
- [ ] Integration test with `FakeLLMProvider` covers all three statuses (success, failed, rejected)

### Validation

Run an ingestion against a Fake LLM with canned responses for each status; assert ExtractionRun table contains the right rows.

---

## Phase 9.3 — Validation + KnowledgeItem storage

**Goal**: Hard and soft validation per doc 4 § Hard vs Soft Validation. Hard failures stay on the ExtractionRun (`rejected`). Soft failures store `KnowledgeItem` with `status="needs_review"`. Otherwise store as `ready`.

### What to build

- **`src/rag_recipes/ingestion/validation.py`**:
  - `validate_hard(extracted_recipe, window) -> list[HardValidationFailure]` per [doc 4 § Hard validation failures](../../architecture/04-llm-assisted-recipe-extraction.md#hard-validation-failures):
    - schema mismatch (caught by Pydantic, but verify the post-parse invariants too)
    - `item_type != "recipe"`
    - missing title
    - missing source_span_ids
    - source_span_ids not in the input window
    - confidence value outside `[0, 1]`
    - ingredient missing `raw_text`
  - `validate_soft(extracted_recipe) -> list[SoftValidationWarning]` per [doc 4 § Soft validation failures](../../architecture/04-llm-assisted-recipe-extraction.md#soft-validation-failures):
    - no ingredients
    - no steps
    - low overall confidence (threshold from `Settings`)
    - low boundary confidence
    - unusually short/long recipe (configurable bounds)
    - low normalization confidence
- **`src/rag_recipes/ingestion/pipeline/persist.py`**:
  - `normalize_title(title: str) -> str` — deterministic lowercasing + whitespace collapsing (used for `normalized_title`)
  - `persist_knowledge_item(extracted, extraction_run_id, window_spans) -> KnowledgeItem`:
    - If hard validation fails: do NOT persist; bubble up so the caller can flip the entire run to `rejected` if appropriate
    - If soft validation fails: persist with `status="needs_review"`, attach warnings to a column (e.g., a new JSONB column on `knowledge_items` named `validation_warnings`, or stored in `structured_data["warnings"]` — pick one and document)
    - Otherwise: persist with `status="ready"`
    - Populate `extraction_run_id`, `source_version`, `normalized_title`, copy `structured_data` and `confidence` from the extracted output
  - **JSONB mutation contract** (deferred here from Phase 2.2): the `knowledge_items` / `chunks` JSONB columns (`structured_data`, `confidence`, `source_span_ids`, `chunk_metadata`) are plain `JSONB` mappings, so SQLAlchemy does **not** flag in-place edits (e.g. `item.structured_data["x"] = 1`) as dirty — they silently won't persist on commit. This phase is the first write/normalization consumer, so it must pick and document one convention: either (a) reassign the whole object (`item.structured_data = {**item.structured_data, ...}`) on every edit, or (b) wrap the columns with `MutableDict`/`MutableList.as_mutable(JSONB)` in the model layer (note: `MutableDict` tracks only top-level keys, so nested `structured_data` edits still need whole-subtree reassignment). Add attribute-history/session tests proving representative edits are detected before commit.

### Acceptance criteria

- [ ] Hard validation failures keep the ExtractionRun status as `success` (the model returned valid JSON) but produce no `KnowledgeItem`. If *all* candidates from one ExtractionRun fail hard, document the behavior (recommendation: do not flip the whole run to `rejected` — `rejected` is for schema/Pydantic-level failures; hard validation failures on individual candidates are application-level)
- [ ] Soft validation failures produce `KnowledgeItem` with `status="needs_review"` and recorded warnings
- [ ] Healthy candidates produce `status="ready"` items
- [ ] `normalized_title` deterministic and stored alongside `title`
- [ ] Unit tests for each hard and soft validation rule
- [ ] Integration test ingests a fixture that produces all three outcomes (ready, needs_review, hard-fail) in one document

### Validation

Curate Fake LLM responses to trigger each validation path; ingest and inspect rows.

---

## Phase 9.4 — Deduplication + candidate scoring + supersede

**Goal**: When overlapping page windows return the same recipe, pick the best candidate via a backend `candidate_score`. When a document is re-extracted later, supersede the older accepted items at document level.

### What to build

- **`src/rag_recipes/ingestion/pipeline/dedup.py`**:
  - `compute_candidate_score(extracted) -> float` per [doc 4 § Deduplication](../../architecture/04-llm-assisted-recipe-extraction.md#deduplication):
    - Combines `confidence.overall`, `confidence.boundary`, ingredient presence, step presence, source-span coverage
    - Exact formula left to implementer; document it in code and in the eval reports later
  - `select_best(candidates: list[ExtractedRecipe]) -> ExtractedRecipe` — groups by `normalized_title` (plus optional source-span overlap heuristic) and keeps the highest-scoring candidate
  - Returns the `(chosen, discarded[])` pair so discarded candidates can be logged for observability (the ExtractionRun output still contains them, but the persistence layer needs to know which one won)
- **Document-level supersede** per [doc 2 § Superseding rule](../../architecture/02-core-data-model.md#superseding-rule):
  - On extraction completion for a document, when the new run reaches `Document.status = "ready"`, mark all older non-superseded `KnowledgeItem` rows for that document as `status = "superseded"` (this includes older `ready` and `needs_review` items)
  - This action runs once at the end of the ingestion pipeline, before final status transition — wire it into the orchestration code in `ingestion/jobs.py`
- **Pipeline orchestration** in `process_document` (extending Epic 8's wrapper):
  - After spans exist, loop over windows
  - For each window: `run_extraction → validate → persist`
  - Collect all newly-created KnowledgeItem candidates
  - Run `select_best` across all candidates (not just per-window — across the document's windows so the same recipe appearing twice resolves)
  - Discard losing candidates (delete the persisted rows, or mark with a separate status — recommend simply deleting since they're not yet referenced by anything; document the choice)
  - Apply document-level supersede over previous accepted items if this is a re-extraction (Epic 11 uses this hook)
  - Transition `Document.status` to `creating_chunks` to signal Epic 10's next stage

### Acceptance criteria

- [ ] Overlapping windows that produce the same recipe yield exactly one persisted `KnowledgeItem`
- [ ] The chosen candidate is the highest-scoring per the documented formula
- [ ] On re-extraction (initial Epic 11 stub), older accepted items get `status="superseded"`
- [ ] Pipeline transitions through `extracting_items` → `validating_items` → `creating_chunks`
- [ ] Integration test: ingest a 5-page synthetic doc where the same recipe spans pages 3–5 (overlapping in window 1 and window 2); assert only one final `KnowledgeItem` exists
- [ ] Failure mid-pipeline transitions document to `failed` with context

### Validation

Integration test described above; manual inspection of `knowledge_items` table after a multi-window ingestion.

---

## Epic-level acceptance criteria

- [ ] Full pipeline: windowed source spans → LLM extraction → ExtractionRun → validation → deduplicated KnowledgeItems
- [ ] Three KnowledgeItem outcomes (ready / needs_review / not persisted due to hard fail) all correctly handled
- [ ] ExtractionRun audit complete (success / failed / rejected) for every LLM call
- [ ] Caching by input_hash demonstrably skips duplicate LLM calls
- [ ] Document-level supersede on re-extraction working
- [ ] Status flow: queued → extracting_text → creating_source_spans → extracting_items → validating_items → creating_chunks (final status set in Epic 10)
- [ ] Status in [`EPICS.md`](../EPICS.md) updated; Epic 10 unblocked
