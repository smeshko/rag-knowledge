# evals/

Offline evaluation harness for retrieval and extraction quality.

> **Working directory.** This is a `backend/` + `frontend/` monorepo. Everything below — every `uv run rag-evals …` command and every relative path such as `data/fixtures/` (meaning `backend/data/fixtures/`) — is run from and relative to **`backend/`**.

Background reading: [`docs/architecture/12-evaluation-and-testing.md`](../docs/architecture/12-evaluation-and-testing.md) (§ 5 judge alignment, § 6 confidence calibration, § 9 regression testing, § 10 report shapes) and [`docs/architecture/13-implementation-decisions.md`](../docs/architecture/13-implementation-decisions.md) (topics 10–12: report/baseline storage, retrieval metrics, LLM-judge fixtures).

## Running the harness

The CLI is installed as the `rag-evals` console script (`uv sync` links it):

```bash
uv run rag-evals --help
uv run rag-evals extraction --fixtures <set> --label <label> [--judge <name>]  # Epic 15
uv run rag-evals judge-alignment --judge <name> --fixtures <set> [--report <dir>]  # Epic 15
uv run rag-evals confidence-review [--report <dir>]             # Epic 15
uv run rag-evals diff <baseline_path> <report_dir>              # Epic 15 (extraction)
uv run rag-evals retrieval --queries <set> [--k 10]             # stub — Epic 16
```

**Provider cost warning.** `extraction`, `judge-alignment` (on a judge-cache miss), and any `--judge` run construct a real LLM provider from `Settings` (`LLM_PROVIDER`, `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) and incur real API cost. They are a **local workflow, never CI** — CI only runs the offline unit tests, which drive everything with `FakeLLMProvider`. The judge cache (below) is what keeps repeat runs cheap. `confidence-review` and `diff` make no provider calls at all.

## Extraction evaluation workflow

The full loop (doc 12 § 5 / § 9), in command order — the order matters because the later commands write back into the run directory the first one created:

1. **`rag-evals extraction --fixtures <set> --label <label> [--judge <name>]`** — drives the real LLM extraction over each fixture's `source.md` (as one synthetic single-page window), scores every `recipe.v1` field against the fixture's golden `expected.json`, and writes a timestamped run dir under `evals/reports/` with `results.json` + `summary.md` (the doc-12 § 10 extraction-report shape: recipes extracted, ready vs needs_review, average confidence, per-field accuracy, missing fields). With `--judge` it also runs the named judge per fixture and records per-fixture ratings + an aggregate pass rate.
2. **`rag-evals judge-alignment --judge <name> --fixtures <set>`** — interactive: you rate each fixture's extraction pass/fail with a critique; the judge's rating is replayed from the cache (LLM only on a miss); agreement % is computed over fixtures both sides rated and disagreements are listed with **both** critiques. The agreement section is written back into the run's `results.json` (`--report <dir>`, default: the latest run).
3. **Iterate the judge prompt** on the disagreement patterns, bump its `# version:` line, and re-run steps 1–2 — the version bump invalidates the judge cache and re-prompts for human ratings; ~0.90 agreement is the defensible bar (doc 12 § 5).
4. **`rag-evals confidence-review`** — buckets the run's items by `confidence.overall` (`0.90–1.00`, `0.75–0.89`, `0.50–0.74`, `<0.50`; boundary values land in the higher bucket), overlays mean objective accuracy + judge pass rate per bucket, and surfaces miscalibration lines ("N items with confidence ≥0.90 failed the judge"). Written back into the same run's `results.json` + `summary.md`.
5. **`rag-evals diff evals/baselines/extraction.json <report_dir>`** — diffs the run against the committed baseline: per-field accuracy, judge pass rate, judge-human agreement (higher is better; a drop beyond 0.01 is flagged `[REGRESSION]`) and item counts (informational). Calibration is a review view, not a diffed metric. Promote an accepted run with `evals.reports.save_as_baseline(run_dir, "extraction")`.

**`results.json` mutation contract:** `extraction` creates the per-run report; `judge-alignment` and `confidence-review` merge their `agreement` / `calibration` sections into that **same** file (preserving `metadata` and each other's sections); `diff` reads that file. Agreement/calibration therefore appear in a diff only when those commands ran against the run being diffed.

## Judge prompts, cache, and alignment records

- **Judge prompts** live at `data/fixtures/judge_prompts/<name>.md`. Three ship with Epic 15: `summary_quality`, `boundary_correctness`, `step_text_quality`. Each carries a `# version: v1` header line — **required for judges** (the harness refuses a version-less judge prompt) and load-bearing: any observable prompt change MUST bump it, because the version keys the cache and appears in every report line.
- **Judge cache** — `evals/reports/.judge_cache/` (gitignored), one JSON file per `(fixture_name, judge_name, judge_version, model)`. A hit skips the LLM call entirely; a version bump misses by construction.
- **Alignment records** — `data/fixtures/judge_alignment/<fixture_name>__<judge_name>.json` (committed). The id is judge-scoped so two judges aligned on one fixture never overwrite each other. `human_rating`/`judge_rating` hold bare `"pass"`/`"fail"`; both critiques plus `{judge_name, judge_version, model, fixture_name, rated_at}` live under `run_metadata`. Human ratings are version-keyed: re-running reuses them while the judge version matches; a bump re-prompts.

## Fixture layout

Loaders (`evals/fixtures.py`) read from `data/fixtures/`:

```
data/fixtures/
├── synthetic_recipes/<set>/<name>/   # one dir per recipe fixture (two-level layout)
│   ├── source.md                     # raw source text fed to extraction
│   ├── expected.json                 # golden recipe.v1 extraction output
│   └── notes.md                      # optional: why this fixture is tricky
├── queries/<set>/
│   ├── queries.tsv                   # query_id <TAB> query_text
│   └── qrels.tsv                     # query_id <TAB> knowledge_item_id <TAB> relevance
├── judge_prompts/<name>.md           # LLM-judge prompt ("# version: vN" header — required for judges)
├── judge_alignment/<fixture>__<judge>.json  # human-vs-judge agreement records (judge-scoped)
└── private/                          # gitignored — personal/copyrighted material
```

The loaders degrade gracefully: absent recipe/query sets load as empty, an absent alignment record loads as `None`. Only `load_judge_prompt` raises (`FileNotFoundError`) — asking for a *named* prompt that does not exist is a caller error.

## Adding a fixture

- **Synthetic recipe**: create `data/fixtures/synthetic_recipes/<set>/<name>/` with `source.md` (raw text) and `expected.json` (the golden `recipe.v1` JSON). Add `notes.md` if the fixture targets a specific failure mode. In `expected.json`, reference the synthetic span id `span_eval_<name>` (`evals.extraction.synthetic_span_id`) in `source_span_ids` so span-provenance scoring lines up.
- **Query set**: create `data/fixtures/queries/<set>/` with `queries.tsv` and `qrels.tsv` (tab-separated; blank lines and `#`-prefixed comment lines are skipped; `relevance` is an int, `1` for binary relevance today). Commit **both** files — a set with only one of them is rejected, because half a set silently scores every query `0.0`.
- **Judge prompt**: add `data/fixtures/judge_prompts/<name>.md` with a `# version: v1` header and the three placeholders `{extracted_output}`, `{expected_output}`, `{source_text}`. A version declaration may also be a bare `version: v1`, `<!-- version: v1 -->`, or a `version:` key inside a `---` front-matter fence; the prompt body is never scanned.
- **Judge alignment record**: written by `rag-evals judge-alignment` via `evals.fixtures.save_judge_alignment(record)` — one `judge_alignment/<fixture>__<judge>.json` per (fixture, judge) with `{fixture_id, human_rating, judge_rating, agreement_status, run_metadata}`.

Fixture shapes are the Pydantic models in `evals/models.py`.

## Reports and baselines

`evals/reports.py` writes one directory per run under `evals/reports/`, named `<YYYY-MM-DDTHH-MM-SS>-<slug>` (with a `-2`, `-3`, … suffix if that name is already taken, so runs never overwrite each other):

```
evals/reports/2026-05-22T12-30-00-extraction-v3/
├── results.json             # {"metadata": {...}, "status": ..., "results": {...}}
├── summary.md
└── per_item_breakdowns.md   # optional
```

`metadata` is provenance captured automatically (timestamp, git commit, embedding/LLM provider + model, prompt/schema version, argv, run label) — only those named fields, never a `Settings` dump, because baselines are committed. `status` is `completed`, or `failed` (plus an `error` key) when the run raised; `save_as_baseline(report_path, name)` refuses to promote a failed run, and `name` must be a single path segment (letters, digits, `.`, `_`, `-`).

Inside `results` an extraction run carries: `per_fixture` (per-field scores, review status, `confidence_overall`, keyed by fixture name), `aggregate` (per-field accuracy, missing-field counts, ready/needs_review), `judge` (per-fixture ratings + pass rate, or `null`), `agreement` (filled by `judge-alignment`, else `null`), and `calibration` (filled by `confidence-review`, else `null`).

## What to commit vs gitignore

Per doc 12 § 5 / § 11 and the existing `backend/.gitignore` rules:

| Data | Policy |
|---|---|
| Synthetic + public-domain fixtures (`data/fixtures/…`) | **commit** |
| Personal/copyrighted material (`data/fixtures/private/`) | **gitignored** |
| Judge prompts (`data/fixtures/judge_prompts/`) | **commit** |
| Judge alignment data (`data/fixtures/judge_alignment/`) | **commit** |
| Per-run reports (`evals/reports/<run>/`) | **gitignored** (only `.gitkeep` tracked) |
| Judge cache (`evals/reports/.judge_cache/`) | **gitignored** (covered by the reports rule) |
| Baselines (`evals/baselines/<name>.json`) | **commit** |
