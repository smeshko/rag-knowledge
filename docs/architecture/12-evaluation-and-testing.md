# 12 — Evaluation and Testing

## Goal

This document explains how we test the system up to retrieval.

Current focus:

- PDF ingestion
- source span creation
- LLM-assisted extraction
- structured recipe data
- confidence scores
- chunk creation
- embeddings
- keyword/vector/hybrid retrieval
- API search responses

Query-time answer generation is not the focus of this document.

---

## First Principle: RAG Quality Needs Evaluation, Not Just Unit Tests

Normal software tests answer questions like:

```text
Does this function return the expected value?
```

RAG evaluation also asks:

```text
Did we retrieve the right sources?
Did extraction preserve the right fields?
Did ranking get better or worse after a change?
Are citations still correct?
```

This project needs both:

```text
software tests + quality evaluations
```

These are different workflows. **Tests** assert binary correctness and gate merges in CI. **Evaluations** measure quality, produce metrics and reports, and run manually — often with real LLM/embedding calls and private fixtures.

Conflating them produces misleading signal: an evaluation with a terrible NDCG score would "pass" the test runner and pollute the green-build illusion. Keep them in separate workflows from the start.

Implementation specifics (directory layout, CLI shape, libraries) are recorded in [13 — Implementation Decisions](./13-implementation-decisions.md). This document stays at the architectural level: *what* we test and evaluate, and *why*.

---

## Testing Layers

The system uses several layers of correctness tests and quality evaluations.

Correctness tests (binary pass/fail, run in CI):

- Unit tests
- Provider contract tests
- Ingestion integration tests

Quality evaluations (produce metrics and reports, run locally):

- Extraction evaluation — objective fields plus LLM-as-judge for subjective fields
- Confidence calibration
- Retrieval evaluation
- Regression diffs against committed baselines

Each layer catches a different kind of problem. Tests answer "is this still correct?" Evaluations answer "is this still good?"

---

## 1. Unit Tests

Unit tests should be fast, deterministic, and not call external APIs.

Good unit-test targets:

- query normalization
- title normalization
- `locator_hash` generation
- `text_hash` generation
- source span locator formatting
- chunk creation from a `KnowledgeItem`
- chunk-type boost lookup
- rank-fusion scoring
- metadata filter construction
- citation label formatting
- hard vs soft validation rules

Example:

```text
"Tomato and White Bean Soup" → "tomato and white bean soup"
```

Example:

```text
PDF locator page_start=42,page_end=42 → label "page 42"
```

These tests should run in normal CI.

---

## 2. Provider Contract Tests

Provider contract tests verify that each provider implementation follows the expected interface.

### FileStorageProvider contract

Test:

- put object
- get object
- check existence
- delete object
- duplicate key behavior

### LLMProvider contract

With a fake provider:

- returns structured output
- reports provider/model
- reports usage if available
- surfaces technical failures

Do not require real LLM calls for normal tests.

### EmbeddingProvider contract

With a fake provider:

- returns correct dimensions
- records provider/model
- supports batch embedding
- handles empty or invalid text consistently

Real provider smoke tests can exist, but should be opt-in.

---

## 3. Ingestion Integration Tests

Integration tests verify that major pieces work together.

A minimal ingestion test should check:

```text
upload fixture PDF
  → SourceAsset created
  → Document created
  → SourceSpans created per page
  → ExtractionRun created
  → KnowledgeItems stored
  → Chunks created for ready items
  → Embeddings created
```

Use fake providers where possible.

For example:

- fake PDF extractor returns known page text
- fake LLM returns known recipe JSON
- fake embedding provider returns deterministic vectors

This lets ingestion behavior be tested without relying on real PDFs or paid APIs.

---

## 4. Golden Fixtures

A golden fixture is a small, known input with an expected output.

### Recipe extraction fixtures

For extraction, useful fixtures include:

```text
one-page recipe
multi-page recipe
two recipes on one page
non-recipe intro page
recipe with unusual ingredients
recipe with missing time/yield
```

Each fixture should define expected outputs:

- source spans
- recipe title
- ingredient lines
- normalized ingredients
- steps
- confidence expectations
- chunk types

### Retrieval fixtures

Retrieval evaluation uses a separate fixture shape:

- **queries**: one entry per query with a stable query ID and the query text
- **qrels** (query relationships): one entry per `(query_id, knowledge_item_id)` pair with a relevance score

For MVP, relevance scores are binary (`1 = relevant`, omitted = not). The format supports graded scores (`0/1/2/3`) later without rebuilding fixtures.

### Judge fixtures

The LLM-as-judge workflow (see [Section 5](#5-extraction-evaluation)) needs two more fixture kinds:

- **judge prompts**: versioned prompts the judge uses to rate a subjective field
- **judge alignment data**: per-fixture records of the human rating, the judge rating, and the agreement status — the data that proves whether the judge is trustworthy

### Copyright note

Do not commit full copyrighted cookbook PDFs or large source excerpts to the repository.

For tests, use:

- synthetic recipe text
- public-domain material
- tiny private local fixtures excluded from git

---

## 5. Extraction Evaluation

Extraction evaluation checks whether the ingestion-time LLM extracted the right structured data.

There are two kinds of extraction fields, and they need different evaluation strategies.

### Objective fields

Objective fields can be checked with exact or near-exact comparison:

- `title` (exact, and after `normalized_title` normalization)
- `yield` (string match)
- `prep_time`, `cook_time`, `total_time` (parsed minutes comparison)
- ingredient count, step count
- per-ingredient: `raw_text` preserved, `quantity_value` parsed, `unit_normalized`, `item_normalized`, `preparation` extracted
- `source_span_ids` (set membership)

Example expected ingredient:

```json
{
  "position": 1,
  "raw_text": "2 tbsp olive oil",
  "quantity_value": 2,
  "unit_normalized": "tablespoon",
  "item_normalized": "olive oil"
}
```

Metrics for objective fields:

```text
exact match
partial match
missing field count
```

These are cheap, deterministic, and trustworthy. Run them on every fixture, every eval run.

### Subjective fields

Subjective fields need judgment, not exact match:

- `summary` quality (captures the recipe's character?)
- recipe boundary correctness (one recipe vs merged vs split)
- step-text fidelity (paraphrased without losing instructions?)
- title accuracy when extracted vs ground-truth differ stylistically

For these, exact match is too brittle and surface-similarity metrics (BLEU, ROUGE, string similarity) are poor proxies for meaning. The right tool is an **LLM-as-judge**: another LLM call that rates a model output as pass/fail with a written critique, using a stable prompt.

### Human-aligned LLM-judge process

The critical rule:

> Do not trust an LLM judge until you have shown it agrees with a human.

The alignment loop:

1. Pick a small annotated fixture set (around 30–50 cases to start).
2. Run extraction over the set; collect model outputs.
3. **You** rate each output as pass/fail with a written critique.
4. The LLM judge rates each output with the same pass/fail + critique format using a versioned judge prompt.
5. Compute agreement: the percentage of cases where the human and judge ratings match.
6. Inspect disagreements. The pattern in them reveals what the judge prompt is missing.
7. Iterate the judge prompt. A useful technique: meta-prompt a strong model with the disagreements and ask it to refine the judge prompt.
8. Re-run and re-check agreement. Stop at the target bar (around 90%+ is defensible).
9. Use the aligned judge for ongoing extraction evaluations at scale.
10. Periodically re-check agreement because data drifts.

This is how teams build LLM judges that actually mean something. Skipping the alignment is how teams build dashboards full of numbers that don't.

### Tracking the judge over time

Judge alignment is itself a metric we track across runs. When the judge prompt is revised, agreement is recomputed and stored. A drop in agreement is a regression signal — the judge has either gotten worse or the data shape has shifted.

Both judge prompts and alignment data are first-class fixtures, not test scaffolding. They live alongside the other fixtures and are versioned in source control.

---

## 6. Confidence Calibration

LLM-reported confidence is not calibrated truth. But it should still be useful as a review signal.

Calibration evaluation answers:

- Are high-confidence items usually correct?
- Are low-confidence items usually problematic?
- Do `needs_review` items actually deserve review?
- Are ingredient normalization confidence scores helpful?

The MVP approach: group extracted items by confidence bucket, then overlay the actual quality scores from objective + judge evaluation.

Example buckets:

```text
0.90–1.00
0.75–0.89
0.50–0.74
below 0.50
```

If many items in the highest confidence bucket score poorly in objective + judge evaluation, the model's confidence signal is not yet useful. The mitigation is either to improve prompts or to build a system confidence score that combines LLM-reported confidence with deterministic validation checks, as noted in [04 — LLM-Assisted Recipe Extraction](./04-llm-assisted-recipe-extraction.md#confidence-scores).

---

## 7. Retrieval Evaluation

Retrieval evaluation checks whether search finds the right items.

It uses a small query set with expected relevant `KnowledgeItem` IDs per query, stored separately from query text so the same IDs can be reused as query phrasing evolves.

Example queries:

```text
white beans
cozy soup with beans
quick weeknight chicken
recipes with gochujang
roasted eggplant technique
```

Each query should have expected relevant `KnowledgeItem` IDs recorded in qrels.

### Metrics

We use one headline metric and three interpretable secondaries.

Headline:

```text
NDCG@10
```

**NDCG@10** (Normalized Discounted Cumulative Gain at 10) rewards correct results in higher positions and supports graded relevance (e.g., "highly relevant" vs "somewhat relevant"). It is the standard metric in modern information retrieval (BEIR, MS MARCO, TREC) and is the single defensible quality number for the retrieval system overall. Range 0 to 1; higher is better.

Secondaries:

```text
Recall@5
Recall@10
MRR
```

### Recall@K

Question:

> Did any expected result appear in the top K?

Example:

```text
Recall@5 = expected recipe appears in first 5 results
```

### MRR

Mean Reciprocal Rank rewards putting the right result near the top.

If the expected recipe is ranked first:

```text
1 / 1 = 1.0
```

If ranked fifth:

```text
1 / 5 = 0.2
```

### Why one headline plus secondaries

NDCG@10 is a single defensible quality number, but it is not the most interpretable metric on its own. Recall@K answers *"did we retrieve the expected item at all?"* and MRR answers *"did the top result help?"* Together they explain *where* NDCG goes wrong when it does — was the item ranked 8th instead of 1st (low MRR, decent Recall), or was it missing entirely (zero Recall, zero MRR)?

### Headline target

We do not pick a specific NDCG@10 target as a "good enough" bar until we have ingested enough cookbooks to know what the baseline looks like. The headline is tracked over time and compared against committed baselines; absolute targets come after some data exists.

---

## 8. Retrieval Debug Review

Metrics are useful, but manual review still matters.

For each test query, inspect:

- keyword candidates
- vector candidates
- merged candidates
- matched chunk types
- final `KnowledgeItem` results
- citations

This uses the debug output defined in [07 — Retrieval Behavior](./07-retrieval-behavior.md#12-debug-information).

Common failure patterns:

- exact ingredient not found by keyword
- vector search retrieves thematically similar but wrong recipes
- full recipe chunks dominate too much
- title chunks are over-boosted
- `needs_review` item accidentally appears in search
- superseded item appears in search
- citations point to wrong pages

---

## 9. Regression Testing

Regression testing answers:

> Did this change make quality worse?

Changes that need regression checks:

- prompt changes
- schema changes
- PDF extraction changes
- chunking changes
- embedding model changes
- ranking/boost changes
- validation rule changes

For each change, compare a new eval run against a committed baseline. The diff covers:

- number of extracted items, ready vs `needs_review` counts
- per-field extraction accuracy (objective fields)
- judge pass rate and judge-human agreement (subjective fields)
- confidence bucket vs actual quality (calibration)
- NDCG@10, Recall@5, Recall@10, MRR
- top results for key queries

The committed baseline is the last accepted eval report. Runs that improve the headline metric become the new baseline; runs that regress get inspected before merging.

---

## 10. Evaluation Reports

Reports come in two formats:

- a **Markdown summary** for humans (readable, narrative)
- a **JSON results** file for machine comparison (deterministic, diffable)
- optional **per-item Markdown breakdowns** for deep inspection

Reports are written per-run with a timestamped run identifier so runs can be compared without overwriting. Committed baselines for regression comparison are stored separately from per-run reports.

### Extraction report shape

```text
Document: Simple Thai Food
Source version: 1
Recipes extracted: 118
Ready: 115
Needs review: 3
Average confidence: 0.87
Objective field accuracy:
  title: 0.97
  yield: 0.92
  ingredients (count match): 0.89
  ingredients (per-field): 0.84
  steps (count match): 0.91
Judge pass rate (summary_quality, v3): 0.86
Judge-human agreement (summary_quality, v3): 0.91
Missing ingredients: 1
Missing steps: 2
```

### Retrieval report shape

```text
Query set: synthetic-v1 (24 queries)
Embedding model: text-embedding-3-small @ 1536
NDCG@10: 0.61
Recall@5: 0.79
Recall@10: 0.88
MRR: 0.54

Per-query examples:
  cozy soup with white beans
    Expected: item_123
    Top 5: item_123, item_456, item_789, ...
    Rank: 1
    Matched chunks: recipe_ingredients, recipe_summary
```

---

## 11. Test Data Strategy

Use three kinds of fixture data, shared between tests and evaluations to avoid duplication.

### Synthetic fixtures

Good for unit and integration tests, and for the first eval runs.

Pros:

- safe to commit
- deterministic
- easy to reason about

### Private local fixtures

Good for realistic cookbook testing.

Pros:

- realistic layouts and language

Cons:

- should not be committed if copyrighted
- may not run in CI

### Public-domain fixtures

Good for shared examples and demos.

Pros:

- safe to commit if license allows
- useful for documentation

### Fixture taxonomy

Within those data categories, fixtures play a few different roles:

- **input fixtures**: synthetic and public-domain recipe text or PDFs that the system processes
- **expected-output fixtures**: the golden `recipe.v1` JSON each input should produce
- **query fixtures**: query text + qrels for retrieval evaluation
- **judge prompt fixtures**: versioned prompts used by the LLM-as-judge
- **judge alignment fixtures**: per-case human and judge ratings used to track agreement over time

Private real cookbooks live alongside synthetic ones in a directory that stays out of source control. Both tests and evaluations import from the same fixture root to avoid duplication.

---

## 12. CI vs Local Evaluation

Not all checks should run in CI.

### CI tests

Should be:

- fast
- deterministic
- no paid API calls
- no private copyrighted fixtures

Run in CI:

```text
unit tests
provider fake tests
schema validation tests
small synthetic integration tests
```

### Local evaluation

Can use:

- real LLM provider
- real embedding provider
- private cookbook samples
- longer extraction and retrieval evaluation reports
- judge-alignment iterations

Run manually before major prompt/model/chunking changes are accepted.

LLM tracing and observability tooling (see [13 — Implementation Decisions](./13-implementation-decisions.md)) is local-only by default — traces stay on the developer machine alongside private fixtures.

---

## 13. MVP Acceptance Checks

Before calling the first recipe RAG flow usable, we should be able to verify:

### Ingestion

- upload one PDF
- create one `Document`
- create one `SourceSpan` per page
- create `ExtractionRun` records
- store ready recipe `KnowledgeItem`s
- store `needs_review` items without chunking them
- create canonical recipe chunks for ready items
- create embeddings for chunks

### Retrieval

- keyword search finds exact ingredients
- vector search finds semantic matches
- hybrid search returns better results than either alone for at least some queries
- search excludes `needs_review` and `superseded` items
- citations point to correct PDF pages
- debug output explains why results appeared

### Reprocessing

- failed ingestion can retry
- ready document can reprocess
- active source version stays stable until a new version is accepted
- old items are superseded, not deleted

### Evaluation workflow

- objective extraction fields scored automatically across the fixture set
- LLM-as-judge runs against subjective fields, with at least one judge prompt aligned to a target agreement bar
- NDCG@10, Recall@5, Recall@10, and MRR computed for the first golden query set
- per-run report (Markdown + JSON) produced and storable as a baseline
- regression diff between two runs is readable and points at specific items that changed

---

## 14. What We Are Not Evaluating Yet

Not yet:

- query-time generated answer quality
- meal planning quality
- shopping list correctness
- nutrition correctness
- multi-category routing
- OCR quality
- cross-device/frontend behavior

Those can be added later.

For now, the priority is:

```text
Can we ingest recipes correctly and retrieve the right ones?
```

---

## What This Teaches

This step teaches that RAG systems need feedback loops.

You cannot improve what you cannot inspect.

Good evaluation gives us confidence when changing:

- prompts
- schemas
- chunking
- embeddings
- retrieval scoring
- provider choices

And the methodology matters as much as the metrics. An LLM-as-judge that has never been compared against a human is just another opinionated black box. A NDCG score with no committed baseline is just a number, not a signal.

---

## Resolved Evaluation Choices

For the MVP:

1. Use both synthetic fixtures and one private real cookbook sample.
2. Synthetic fixtures are safe for CI and source control.
3. The real cookbook sample is for local evaluation only and should stay out of git if copyrighted.
4. **NDCG@10** is the headline retrieval metric; Recall@5, Recall@10, and MRR are interpretable secondaries.
5. Retrieval fixtures use a stable `query_id → knowledge_item_id` qrels structure that supports binary today and graded relevance later without rebuild.
6. **Objective extraction fields** use exact-match or numeric comparison.
7. **Subjective extraction fields** use a **human-aligned LLM-as-judge**.
8. The LLM judge is not trusted until human-judge agreement reaches a defensible bar; agreement is itself a tracked metric.
9. Confidence is evaluated as **calibration** — confidence buckets overlaid against actual quality from objective + judge evaluation.
10. Evaluation runs produce per-run reports (Markdown + JSON); regression diffs compare against committed baselines.
11. Tests and evaluations are kept in separate workflows so quality metrics never silently "pass" CI.

Implementation details for these choices — directory layout, CLI shape, metrics library, judge prompt format — are recorded in [13 — Implementation Decisions](./13-implementation-decisions.md).

## Open Questions Before Implementation

Before implementation, we still need to decide:

1. What small set of synthetic recipe fixtures should we create first?
2. Which real cookbook sample should be used for private local evaluation?
3. Which retrieval queries should become the first golden query set?
4. Which judge prompts do we write first (summary quality, boundary correctness, step-text fidelity)?
5. What initial agreement bar (e.g., 80%? 90%?) is "aligned enough" to start trusting the LLM judge for ongoing eval at scale?

These are best answered after the first end-to-end ingestion runs exist. Picking imaginary fixtures, queries, and judge prompts before any real data exists would be premature.
