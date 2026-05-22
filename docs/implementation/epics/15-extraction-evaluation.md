# Epic 15 — Extraction Evaluation

**Status**: Blocked (depends on Epics 9, 14)

## Overview

Build the actual extraction-evaluation logic: per-field objective scoring for measurable fields, an LLM-as-judge implementation for subjective fields, the human-aligned judge workflow that tracks agreement between human and judge ratings, and a confidence calibration report. After this epic, prompt and extraction changes can be evaluated systematically against fixtures and committed baselines.

## Architecture references

- [12 — Evaluation and Testing § 5, 6](../../architecture/12-evaluation-and-testing.md#5-extraction-evaluation) — extraction methodology, the human-aligned judge process, confidence calibration
- [12 § 11](../../architecture/12-evaluation-and-testing.md#11-test-data-strategy) — fixture taxonomy
- [13 — Implementation Decisions, topic 12](../../architecture/13-implementation-decisions.md#12-extraction-evaluation-methodology) — methodology resolution

## Dependencies

- Epic 9 (extraction pipeline produces KnowledgeItem rows we can evaluate against fixtures)
- Epic 14 (CLI, fixture loaders, report writer)

## Out of scope

- Specific synthetic fixture content (create alongside this work as needed; tracked in `data/fixtures/synthetic_recipes/`)
- Specific judge prompts (initial drafts can land with this epic; they evolve through iteration)
- Retrieval evaluation (Epic 16)

---

## Phase 15.1 — Objective field scoring

**Goal**: `rag-evals extraction` runs the extraction pipeline (or replays cached ExtractionRun output) over a fixture set and produces an objective-score report.

### What to build

- **`evals/scripts/extraction_eval.py`** (or `evals/extraction.py` imported by the CLI):
  - `run_extraction_eval(fixture_set: str, label: str) -> ReportRun`:
    - Loads recipe fixtures via Epic 14 loader
    - For each fixture: drive the extraction pipeline (the cleanest path is to call the pipeline's pure-function entry point directly with the fixture's `source.md` as input text — no need to go through PDF extraction. Add an alternative LLM call path if `source.md` represents a single recipe, calling the LLM with a synthetic page-window equivalent.)
    - Compare the produced `KnowledgeItem` against `expected.json`
    - Compute per-field scores
- **`evals/scoring/objective.py`** with scoring functions:
  - `score_title(actual, expected)` — exact match + normalized match
  - `score_yield(actual, expected)` — string match
  - `score_times(actual, expected)` — parse to minutes, compare numerically
  - `score_ingredient_count(actual, expected)` — exact count match
  - `score_step_count(actual, expected)` — exact count match
  - `score_ingredients_detail(actual, expected) -> IngredientScoreBreakdown` — per ingredient: raw_text preserved, quantity_value parsed, unit_normalized, item_normalized, preparation; aggregate to precision/recall/F1
  - `score_source_span_ids(actual, expected)` — set membership
- **Report structure** (in `results.json`):
  - Per-fixture: per-field scores
  - Aggregate across fixtures: per-field accuracy, missing-field counts, ready-vs-needs_review counts
- **`summary.md`**: human-readable summary in the shape from [doc 12 § 10](../../architecture/12-evaluation-and-testing.md#10-evaluation-reports) "Extraction report"

### Acceptance criteria

- [ ] `rag-evals extraction --fixtures synthetic --label <label>` produces a timestamped report
- [ ] Per-field accuracy aggregated across the fixture set
- [ ] Ingredient detail scoring works per-field (not just count)
- [ ] Missing-field counts surfaced
- [ ] Ready vs `needs_review` counts surfaced
- [ ] Report compares against the committed baseline (if one exists) and prints a diff
- [ ] Unit tests for each objective scorer

### Validation

Create a small fixture set (2–3 hand-crafted synthetic recipes) and run the eval; inspect the produced report for accuracy.

---

## Phase 15.2 — LLM-as-judge integration + versioned judge prompts

**Goal**: For subjective fields (summary quality, boundary correctness, step-text fidelity), call an LLM judge using versioned prompts and record pass/fail + critique alongside the objective scores.

### What to build

- **Initial judge prompts** in `data/fixtures/judge_prompts/`:
  - `summary_quality.md` — prompt the judge to rate whether the extracted summary captures the recipe's character
  - `boundary_correctness.md` — prompt the judge to rate whether the recipe was correctly delimited (one recipe vs merged vs split)
  - `step_text_quality.md` — prompt the judge to rate whether step text preserves cooking instructions without paraphrasing-induced loss
  - Each prompt has a version embedded in the filename or front matter (e.g., `# version: v1`)
  - Each prompt enforces a Pydantic-typed output: `{ rating: "pass" | "fail", critique: str }`
- **`evals/judges.py`**:
  - `Judge` class wrapping `LLMProvider`:
    - `Judge(name: str, version: str, llm_provider: LLMProvider)`
    - `judge(extracted: ExtractedRecipe, expected: ExpectedRecipe, source_text: str) -> JudgeRating`
    - Uses strict JSON schema mode to produce typed output
  - `JudgeRating` Pydantic model: `judge_name`, `judge_version`, `rating`, `critique`, `metadata` (model, prompt_version, etc.)
- **Integration with extraction eval**:
  - When `rag-evals extraction --judge <name>` is invoked, run the named judge on each fixture
  - Record judge ratings alongside objective scores
  - Aggregate judge pass rate per judge
- **Cost / latency awareness**: judges may invoke real LLM calls; cache results by `(fixture_id, judge_name, judge_version, model)` to avoid re-running on every eval

### Acceptance criteria

- [ ] At least three judge prompts committed (summary, boundary, step-text)
- [ ] Each prompt produces typed `pass/fail + critique` output
- [ ] `rag-evals extraction --judge summary_quality` runs the judge across the fixture set
- [ ] Judge results recorded in the report alongside objective scores
- [ ] Judge cache prevents redundant LLM calls when re-running the same `(fixture, judge_version)` pair
- [ ] mypy passes against `evals/judges.py`

### Validation

Run an extraction eval with `--judge summary_quality` against a small fixture; inspect the report's judge ratings; manually compare a few to your own assessment.

---

## Phase 15.3 — Judge alignment workflow + confidence calibration

**Goal**: Compute and track judge–human agreement over time; identify miscalibrated confidence buckets.

### What to build

- **`rag-evals judge-alignment --judge <name>`** subcommand:
  - Loads existing judge-alignment records from `data/fixtures/judge_alignment/`
  - For each fixture:
    - If a human rating exists for the current judge version: reuse it
    - If not: prompt the user (interactively in the CLI) to provide pass/fail + critique
    - Save the human rating
  - Run the judge on the fixture (or load the cached judge rating from Phase 15.2)
  - Compare ratings; compute `agreement_status: "agree" | "disagree"`
  - Save the full alignment record
- **Agreement aggregation**:
  - Compute overall agreement percentage across the fixture set
  - List disagreements with their critiques (both human and judge) so the user can read the patterns
  - The agreement score itself is written to the report and treated as a tracked metric (it goes into the baseline diff)
- **`rag-evals confidence-review`** subcommand:
  - Loads the latest extraction-eval results
  - Groups items by `confidence.overall` bucket: `0.90–1.00`, `0.75–0.89`, `0.50–0.74`, `<0.50`
  - For each bucket, overlays the actual quality scores from objective + judge evaluation
  - Surfaces miscalibration: e.g., "12 items have confidence ≥0.9 but failed the summary judge"
  - Writes the calibration view to the report's summary and JSON
- **Baseline-diff logic** in `evals/reports.py` (filling in the skeleton from Epic 14):
  - For extraction reports: diff per-field accuracy, judge pass rates, agreement scores, item counts
  - Print regressions in red (or marked with `[REGRESSION]`) so they're visible

### Acceptance criteria

- [ ] `rag-evals judge-alignment --judge <name>` walks through fixtures and collects human ratings
- [ ] Alignment records persist across runs (re-running doesn't lose prior human ratings)
- [ ] Agreement score computed and reported
- [ ] Disagreements listed with both critiques for inspection
- [ ] `rag-evals confidence-review` produces the bucket view
- [ ] Baseline diff highlights regressions in per-field accuracy, judge pass rate, and agreement
- [ ] `evals/README.md` updated with the full extraction-eval workflow

### Validation

Run the full sequence on a small fixture set:
1. `rag-evals extraction --fixtures synthetic --judge summary_quality --label initial`
2. `rag-evals judge-alignment --judge summary_quality` (interactively rate disagreements)
3. Iterate the judge prompt; re-run
4. Verify agreement improves
5. `rag-evals confidence-review` produces the calibration view

---

## Epic-level acceptance criteria

- [ ] Objective field scoring works for all measurable extraction fields
- [ ] LLM-as-judge integration runs versioned judge prompts and records typed ratings
- [ ] Judge-alignment workflow collects human ratings, computes agreement, surfaces disagreements
- [ ] Confidence calibration report groups items by bucket and overlays actual quality
- [ ] Baseline diff for extraction reports highlights regressions
- [ ] Initial judge prompts (summary, boundary, step-text) committed
- [ ] All eval reports include the metrics from [doc 12 § 10](../../architecture/12-evaluation-and-testing.md#10-evaluation-reports)
- [ ] Status in [`EPICS.md`](../EPICS.md) updated
