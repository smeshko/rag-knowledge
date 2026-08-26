# Implementation decisions — follow-up entries

`docs/architecture/13-implementation-decisions.md` (untracked since June 2026;
recoverable from git history at `a0e67ce~1`) records decisions #1–#13 and its
own rule: *if a choice is later replaced, leave the old entry in place and add a
follow-up entry rather than rewriting history.* This tracked file holds the
entries made since, in the same shape. Dates are when the code landed.

---

## 14. Hard delete of knowledge items and documents (2026-08-25)

**Decision**: `DELETE /knowledge-items/{id}` and `DELETE /documents/{id}` are
hard, cascading deletes (chunks, embeddings, favourites, the 22.1 edit
snapshot). Reprocessing still *supersedes*; only an explicit user delete
removes rows.

- **Options**: supersede-and-keep (doc 02 "old items are not deleted by default,
  because they are useful for audit/debugging"); soft delete via the existing
  `rejected` status; hard delete.
- **Why hard**: the doc 02 rule was written for *pipeline* outcomes, where a
  generation replaces another and the audit is the point. A user deleting a
  wrong or unwanted recipe is a product action; keeping the row would require
  filtering it out of every listing, search, favourites and count forever, and
  the review surface already has `rejected` for "the pipeline's candidate was
  wrong", which is a different statement. The per-table deleted-row counts are
  logged at INFO as the only trace. The handwritten shelf is exempt from
  whole-document delete (#20).

## 15. `menus/` as a second flow of the query-time answer layer (2026-08-25)

**Decision**: multi-course menu composition ships as `src/rag_recipes/menus/`,
a sibling package of `answers/`, despite doc 08 §10 deferring "multi-step meal
planning" and doc 13 §8 mapping top-level packages 1:1 to architectural layers.

- **Why now**: it is the first answer-shaped feature the retrieval layer could
  not serve with one embedding, and the grounding contract was reusable as-is.
- **Why a sibling, not `answers/menus/`**: it has its own plan → per-course
  retrieval → cross-course assignment stages and two prompts/schemas; nesting
  it under `answers/` would make that package the layer *and* one flow of it.
  The rule that keeps the two honest: anything both need lives in
  `answers/shared.py` (list coercion, chunk fetch, preview projection, the
  allowed-id footer) and the grounding contract (`build_context_pack`,
  `build_response_citations`, `GROUNDING_RULES`) is imported, never copied.
  Doc 08 §10's deferral is thereby closed for menus; meal *planning* over days
  remains deferred.

## 16. Assembly-recipe exemption and its trace (2026-08-25)

**Decision**: a step-less candidate with 3–12 ingredients and ≤400 body chars is
an *assembly* recipe (a cheese board, a composed plate): `no_steps` and
`recipe_too_short` are waived together, and the waiver is recorded as
`structured_data["validation_notes"] = ["assembly_recipe"]`.

- **Why the trace**: without it the waived row is indistinguishable from a
  fully structured recipe, and doc 12 §6/§9 (calibration buckets, regression
  diffs of validation-rule changes) cannot see the population the exemption
  created. Notes never affect status; they are audit data, the informational
  sibling of `warnings`. Bounds are Settings (doc 11 §5).

## 17. Thresholds snapshotted per item (2026-08-26)

**Decision**: the `SoftValidationThresholds` in force when an item's warnings
were derived are stored as `structured_data["validation_thresholds"]` (by
persist, the review edit path and the manual shelf). The review projections
read that back; rows that predate it fall back to the live Settings and say so
(`review_thresholds.source`).

- **Options**: a JSONB column on `ExtractionRun` (needs a migration; the manual
  shelf has one synthetic run for every item); on the item's `structured_data`
  next to `warnings`, which is already machine-owned provenance.
- **Why**: doc 11 §6 — thresholds are tunables; a reason's `threshold` shown
  from today's Settings can contradict the score it decorates.

## 18. Two version constants, two meanings (2026-08-26)

**Decision**: `PROMPT_VERSION` versions everything the model sees — template
*and* the model-facing JSON schema. `SCHEMA_VERSION` versions the stored payload
contract (`structured_data.schema`). The TOKEN BUDGET trim (prompt v2) shrank
the model-facing schema without changing the payload, so only `PROMPT_VERSION`
moved. A fingerprint test pins `build_recipe_v1_json_schema()` to the prompt
version; retired templates stay in `prompts/` for reproducibility.

## 19. `claude_cli` as a subscription-billed extraction provider (2026-08-24)

**Decision**: a fourth `LLMProvider` shells out to the local `claude -p`. Auth is
the CLI's own login; `ANTHROPIC_API_KEY`/`AUTH_TOKEN` are stripped from the
subprocess so a call can never silently bill the API. Exact model pins are
required (aliases drift and invalidate the cache key). Quota exhaustion fails
the document rather than retrying; `/documents/batch` is refused (Anthropic
batch API only). Per doc 11 §1 it sits behind the same interface and registry
as the API-billed providers; the CLI-specific billing details go on the trace
observation, not on `StructuredOutputResponse`.

## 20. The handwritten shelf (2026-08-26)

**Decision**: `POST /knowledge-items` writes a human-typed recipe onto a single
synthetic document (`SourceType.MANUAL`, one placeholder `SourceAsset`, one
`ExtractionRun` labelled `manual`) so that every read path — listings, search,
favourites, citations — keeps its "item → run → document → asset" chain intact.

- **Consequences, chosen**: reprocess and whole-document delete are refused on
  the shelf (400 `invalid_request`); recipes leave it one at a time. The shelf
  appears in `GET /documents` with `source_type: "manual"`. The item's row shape
  is produced by the *edit* layer (`apply_edit`, `compose_body_text`,
  `warnings_for_item`), so a typed recipe and an extracted one are validated
  and projected identically.
