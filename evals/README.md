# evals/

Offline evaluation harness for retrieval and extraction quality.

> **Working directory.** This is a `backend/` + `frontend/` monorepo. Everything below — every `uv run rag-evals …` command and every relative path such as `data/fixtures/` (meaning `backend/data/fixtures/`) — is run from and relative to **`backend/`**.

Background reading: [`docs/architecture/12-evaluation-and-testing.md`](../docs/architecture/12-evaluation-and-testing.md) (§ 5 judge alignment, § 6 confidence calibration, § 9 regression testing, § 10 report shapes) and [`docs/architecture/13-implementation-decisions.md`](../docs/architecture/13-implementation-decisions.md) (topics 10–12: report/baseline storage, retrieval metrics, LLM-judge fixtures).

## Judging with a different provider than the model under test

By default the eval harness builds **one** LLM provider and uses it for both
extraction and judging. That is fine for a single-provider run and wrong for a
comparison: with one provider in both roles, every candidate grades itself, and
a "provider A scores higher than provider B" result says as much about the
judges as about the models.

Set the judge separately for any cross-provider run:

```bash
# Extract with DeepSeek, judge with Claude.
LLM_PROVIDER=deepseek JUDGE_LLM_PROVIDER=anthropic rag-evals extraction \
    --fixtures cookbooks --label deepseek-judged-by-claude --judge summary_quality
```

Rules worth knowing:

- **Unset means "same provider as extraction"** — today's behaviour, unchanged.
- **Hold the judge fixed across a comparison.** Running candidate A judged by A
  and candidate B judged by B is the original confound with extra steps. Both
  baselines in a comparison must record the same judge provider.
- `JUDGE_LLM_MODEL` alone (provider unset) judges with a different model on the
  same vendor. Setting only `JUDGE_LLM_PROVIDER` picks that provider's own
  model — never `LLM_MODEL`, which would send one vendor's model id to another.
- A judge provider needs its own API key, checked at `Settings` load rather than
  at the first judge call — which happens after extraction has already run.
- The run's `results.json` records `judge.provider` alongside `judge.model`, so a
  committed baseline states who graded it.

### This change invalidated the existing judge cache

The judge-rating cache key now includes the judge's provider. Without it, two
providers serving the same model name would share cached ratings — the exact
cross-vendor contamination the split exists to remove. The cache path is a hash
over the key, so **every previously cached rating is orphaned** and the first run
after this change re-issues every judge call. That is expected, not a bug. The
orphaned files are harmless and are left in place.


## Running the harness

The CLI is installed as the `rag-evals` console script (`uv sync` links it):

```bash
uv run rag-evals --help
uv run rag-evals extraction --fixtures <set> --label <label> [--judge <name>]  # Epic 15
uv run rag-evals judge-alignment --judge <name> --fixtures <set> [--report <dir>]  # Epic 15
uv run rag-evals confidence-review [--report <dir>]             # Epic 15
uv run rag-evals retrieval --queries <set> [--k 10] [--mode hybrid] [--label <label>]  # Epic 16
uv run rag-evals diff <baseline_path> <report_dir>              # both report types
uv run rag-evals save-baseline <report_path> --name <name>
```

**Provider cost warning.** `extraction`, `judge-alignment`, and any `--judge` run construct a real LLM provider from `Settings` (`LLM_PROVIDER`, `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) and incur real API cost. `judge-alignment` never drives extraction — it rates the artifacts persisted in an existing extraction run and makes **at most one judge call per fixture**, and zero when the judge cache is already warm for that run's artifacts (the normal case after `extraction --judge`). They are a **local workflow, never CI** — CI only runs the offline unit tests, which drive everything with `FakeLLMProvider`. The judge cache (below) is what keeps repeat runs cheap. **`rag-evals retrieval` performs live searches**: it drives `POST /api/v1/search` in-process through the real app, whose dependencies construct the real embedding (and, when `reranking_enabled`, reranker) providers — running it needs a reachable database and provider credentials. `confidence-review`, `diff`, and `save-baseline` make no provider calls at all (files only).

## Extraction evaluation workflow

The full loop (doc 12 § 5 / § 9), in command order — the order matters because the later commands write back into the run directory the first one created:

1. **`rag-evals extraction --fixtures <set> --label <label> [--judge <name>]`** — drives the real LLM extraction over each fixture's `source.md` (as one synthetic single-page window), scores every `recipe.v1` field against the fixture's golden `expected.json`, and writes a timestamped run dir under `evals/reports/` with `results.json` + `summary.md` (the doc-12 § 10 extraction-report shape: recipes extracted, ready vs needs_review, average confidence, per-field accuracy, missing fields). Every scored fixture persists its full extracted `recipes` payload plus the `fixture_content_hash` it was produced from and a `scored_recipe_index`, and the run records its `extraction_prompt_version` — the provenance later stages rate against. With `--judge` it also runs the named judge per fixture and records per-fixture ratings + an aggregate pass rate.
2. **`rag-evals judge-alignment --judge <name> --fixtures <set>`** — interactive: rates the run's **persisted** artifacts — it never re-extracts and requires a usable extraction run to exist (exit 2 otherwise: no run, missing/malformed `results.json`, a failed or retrieval run, a fixture-set mismatch, or a fixture edited since the run). You are shown the *exact* serialized artifact the judge rates, told which judge dimension you are rating, and asked pass/fail with a critique; the judge's rating is replayed from the cache (LLM only on a miss); agreement % is computed over fixtures both sides rated and disagreements are listed with **both** critiques. A fixture with no persisted artifact in the run is recorded `unrated` and you are not prompted for it. The agreement section is written back into the run's `results.json` (`--report <dir>`, default: the latest run).
3. **Iterate the judge prompt** on the disagreement patterns, bump its `# version:` line, and re-run steps 1–2 — the version bump invalidates the judge cache and re-prompts for human ratings. That is only half the invalidation story: a *changed artifact* (a re-extraction that produced different output) is a second, independent trigger for both — the cache keys on the artifact's hash and human-rating reuse binds to it. ~0.90 agreement is the defensible bar (doc 12 § 5).
4. **`rag-evals confidence-review`** — buckets the run's items by `confidence.overall` (`0.90–1.00`, `0.75–0.89`, `0.50–0.74`, `<0.50`; boundary values land in the higher bucket), overlays mean objective accuracy + judge pass rate per bucket, and surfaces miscalibration lines ("N items with confidence ≥0.90 failed the judge"). Written back into the same run's `results.json` + `summary.md`.
5. **`rag-evals diff evals/baselines/extraction.json <report_dir>`** — diffs the run against the committed baseline: per-field accuracy, judge pass rate, judge-human agreement (higher is better; a drop beyond 0.01 is flagged `[REGRESSION]`) and item counts (informational). Calibration is a review view, not a diffed metric. Promote an accepted run with `rag-evals save-baseline <report_dir> --name extraction`.

**`results.json` mutation contract:** `extraction` creates the per-run report; `judge-alignment` and `confidence-review` merge their `agreement` / `calibration` sections into that **same** file (preserving `metadata` and each other's sections); `diff` reads that file. Agreement/calibration therefore appear in a diff only when those commands ran against the run being diffed.

## Judge prompts, cache, and alignment records

- **Judge prompts** live at `data/fixtures/judge_prompts/<name>.md`. Three ship with Epic 15: `summary_quality`, `boundary_correctness`, `step_text_quality`. Each carries a `# version: v1` header line — **required for judges** (the harness refuses a version-less judge prompt) and load-bearing: any observable prompt change MUST bump it, because the version keys the cache and appears in every report line.
- **Judge cache** — `evals/reports/.judge_cache/` (gitignored), one JSON file per eight-part key `(fixture_set, fixture_name, fixture_content_hash, extraction_prompt_version, artifact_hash, judge_name, judge_version, model)`. A hit skips the LLM call entirely. Invalidation is no longer only judge-prompt version bumps: a fixture `source.md`/`expected.json` edit, an extraction prompt-version bump, and **any change to the judged artifact itself** each miss by construction. What that buys and costs: a cached rating is replayed only for the byte-identical artifact it was produced from — so a fresh live extraction re-pays its judge calls, while re-running `judge-alignment` against an existing run stays free — and the cache grows by one entry per distinct artifact instead of replacing entries (gitignored and disposable; delete it any time).
- **The judge rates the full item list.** The judged string is the run's persisted `recipes` payload serialized as `{"items": [...]}` — so `boundary_correctness` actually sees splits instead of only item 0. Note the three committed prompts are still worded for a *single* item; how a judge aggregates a multi-item payload into one verdict is an open Phase 20.3 question (every synthetic fixture holds one recipe by design, so the wrapper is a one-element list in the normal case).
- **Alignment records** — `data/fixtures/judge_alignment/<set>__<fixture>__<judge>__<extraction_model>.json` (committed). The id is scoped by fixture set, judge, **and the model that produced the artifact** (the run's `metadata.llm_model`, not the judge's) so two sets with a same-named fixture, two judges on one fixture, and two extraction providers over the same set all keep distinct records. `human_rating`/`judge_rating` hold bare `"pass"`/`"fail"`; both critiques plus `{judge_name, judge_version, model, fixture_name, fixture_set, artifact_hash, judge_dimension, extraction_provider, extraction_model, aligned_run, rated_at}` live under `run_metadata`. Human ratings are reused only while the judge version **and** the artifact hash both match; a version bump *or* a changed artifact re-prompts.

## Retrieval eval workflow: run → save baseline → change → re-run → diff

```bash
# 1. Run the eval and note the printed report directory.
uv run rag-evals retrieval --queries golden --k 10 --mode hybrid --label before

# 2. Promote that run to the committed baseline (refuses failed runs).
uv run rag-evals save-baseline evals/reports/<run-dir> --name retrieval

# 3. Change something (prompt, boost knob, reranker, embedding model, ...),
#    then re-run with a fresh label.
uv run rag-evals retrieval --queries golden --k 10 --mode hybrid --label after

# 4. Diff the new run against the committed baseline.
uv run rag-evals diff evals/baselines/retrieval.json evals/reports/<new-run-dir>
```

`diff` prints a per-metric headline (`NDCG@10: 0.61 → 0.57 [REGRESSION -0.04]`
past a small delta threshold), the per-query regression list (an expected item
that dropped out of the top-k, or whose rank worsened by ≥ 3), and the biggest
per-query NDCG@10 drops — and exits **1** when a regression was found (0
otherwise, **2** for any input error), so it can gate a local check without
being a CI gate. Exit 2 covers a missing path, malformed JSON, a run finalized
`failed`, a `report_type` mismatch between the two sides, a payload that
fails structural validation (missing/mistyped `run`, `aggregate`,
`per_query`, comparability fields, non-finite metrics, or invalid
`expected_item_ranks`), and two runs that are not comparable. Runs are only comparable when `query_set` / `mode` /
`reranking_enabled` / `embedding_model` / `k` / `limit` all match; when they
differ the diff prints a prominent warning and the full deltas but reports no
quality verdict, so a run-config mismatch can never masquerade as a
regression. Commit the
updated `evals/baselines/retrieval.json` when a new baseline is intended —
per-run report directories stay gitignored.

`diff` dispatches on the reports' `report_type`: the same command diffs an
extraction pair (step 5 of the extraction workflow above) with the same exit
contract — 1 on a flagged `[REGRESSION]`, 0 otherwise, 2 for any input error
— and refuses a `report_type` mismatch between the two sides with exit 2.

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
├── judge_alignment/<set>__<fixture>__<judge>__<extraction_model>.json  # human-vs-judge agreement records
└── private/                          # gitignored — personal/copyrighted material
```

The loaders degrade gracefully: absent recipe/query sets load as empty, an absent alignment record loads as `None`. Only `load_judge_prompt` raises (`FileNotFoundError`) — asking for a *named* prompt that does not exist is a caller error.

## Adding a fixture

- **Synthetic recipe**: create `data/fixtures/synthetic_recipes/<set>/<name>/` with `source.md` (raw text) and `expected.json` (the golden `recipe.v1` JSON). Add `notes.md` if the fixture targets a specific failure mode. In `expected.json`, reference the synthetic span id `span_eval_<name>` (`evals.extraction.synthetic_span_id`) in `source_span_ids` so span-provenance scoring lines up.
- **Query set**: create `data/fixtures/queries/<set>/` with `queries.tsv` and `qrels.tsv` (tab-separated; blank lines and `#`-prefixed comment lines are skipped; `relevance` is an int, `1` for binary relevance today). Commit **both** files — a set with only one of them is rejected, because half a set silently scores every query `0.0`.
- **Judge prompt**: add `data/fixtures/judge_prompts/<name>.md` with a `# version: v1` header and the three placeholders `{extracted_output}`, `{expected_output}`, `{source_text}`. A version declaration may also be a bare `version: v1`, `<!-- version: v1 -->`, or a `version:` key inside a `---` front-matter fence; the prompt body is never scanned. The `Rate exactly ONE subjective dimension: … ?` sentence is **load-bearing**: `judge-alignment` lifts it verbatim (through the first `?`) to tell the human which dimension they are rating; a prompt without it falls back to showing only the judge name.
- **Judge alignment record**: written by `rag-evals judge-alignment` via `evals.fixtures.save_judge_alignment(record)` — one `judge_alignment/<set>__<fixture>__<judge>__<extraction_model>.json` per (set, fixture, judge, extraction model) with `{fixture_id, human_rating, judge_rating, agreement_status, run_metadata}`; `run_metadata` carries `artifact_hash`, `fixture_set`, `judge_dimension`, `extraction_provider`, `extraction_model`, and the `aligned_run` directory alongside the Epic-15 keys.

Fixture shapes are the Pydantic models in `evals/models.py`.

## Cutting fixtures from real PDFs

`rag-evals fixtures cut` (Epic 23 Phase 23.1, `evals/fixture_cutter.py`) turns real cookbook page ranges into fixture *candidates* — `source.md` + a provenance `notes.md`, with **no** `expected.json`. Goldens are authored separately (Phase 23.2), so a freshly cut set cannot be scored by `rag-evals extraction` yet, and `load_recipe_fixtures("<set>")` raises `FileNotFoundError` on it until the goldens land. That is by design; the layout gate is `evals.fixture_cutter.validate_candidate_layout`.

Single range:

```bash
uv run rag-evals fixtures cut \
  --pdf ~/Downloads/books/cidermadesimple.pdf \
  --set cookbooks \
  --pages 42-43 \
  --rationale "one complete recipe with headnote; ingredient table"
```

Batch — repeat `--pages` and `--rationale`, **strictly interleaved**:

```bash
uv run rag-evals fixtures cut \
  --pdf ~/Downloads/books/eatdrinkpaleocookbook.pdf \
  --set cookbooks \
  --pages 44 --rationale "clean single-page recipe" \
  --pages 61-62 --rationale "dense prose method" \
  --pages 88 --rationale "hard case: ingredients as a table"
```

The two options are collected independently and paired **positionally**: the Nth `--rationale` is attached to the Nth `--pages`, wherever the flags sit on the command line. Only the *counts* are checked, so `--rationale A --pages 1 --rationale B --pages 2` is accepted with exit 0 and the rationales attached the way the parser saw them, not the way you meant them. Always write one `--pages` immediately followed by its `--rationale`, as above, and read back the `rationale:` field in each `notes.md` before authoring goldens.

Batch mode is not a convenience: `PdfTextExtractor.extract_pages` takes whole-file bytes, so one invocation per range re-parses the entire book. At 190 MB (`eatdrinkpaleocookbook.pdf`) that is one parse versus eight. Cut every window you want from a book in a single command.

Rules the cutter enforces:

- **Names are derived, never supplied**: `<book-stem>-p<first>-<last>`. The name becomes `span_eval_<name>` (`evals.extraction.synthetic_span_id`), the span id goldens cite — so a collision exits 2 unless `--overwrite`, and overwriting a fixture that already carries `expected.json` additionally needs `--invalidate-goldens` (the golden then describes text that no longer exists: re-author it).
- **Pages are inclusive and 1-based**; `--pages 42` is shorthand for `42-42`. An inverted, zero/negative, unparseable, or out-of-document range exits 2 with the problem named, as do a missing `--pdf`, an unsafe `--set`, and a `--rationale` count that does not match `--pages`. Nothing is written unless the whole batch validates.
- **Text is production's text.** The cutter runs the same `PyMuPdfExtractor` ingestion uses and slices one whole-document extraction; `source.md` is exactly the selected pages' text joined by `\n\n`. Pages below `PDF_MIN_TEXT_CHARS_FOR_PAGE` (image plates) are *kept with their text*, matching production, and reported on stderr as a warning so you can re-cut a bad window. A range where *every* page is sub-threshold would yield an empty `source.md` — and, in 23.2, an empty prompt — so it exits 2 instead of writing the fixture.

Curation rules for the curator:

- **Exactly one complete recipe per fixture** — title, ingredients, steps. The extraction driver scores `recipes[0]`, so a window holding two recipes punishes a model for correctly returning both.
- **Prefer single-page ranges.** The eval harness collapses a whole `source.md` into a *single* span stamped `page_start: 1`, whereas production emits one `[SOURCE_SPAN … | PDF page N]` block per page. A multi-page fixture therefore exercises a prompt shape production never emits, and `source_span_ids_f1` is degenerate (there is only one span to cite). Known and documented, not hidden.
- **Spread the picks.** Draw across books rather than mining one, and include deliberately awkward windows (dense prose, ingredient tables, long headnotes) — a curator who only picks clean pages inflates measured accuracy.
- Record why you chose the window in `--rationale`; it lands in `notes.md` as a parseable field alongside the source filename, page range, extractor identity, and any flagged pages. `notes.md`'s provenance block is line-oriented, so the rationale is collapsed to a single line — a multi-line rationale is kept whole, not truncated, but it will not read as multiple lines in the file.

## Reports and baselines

`evals/reports.py` writes one directory per run under `evals/reports/`, named `<YYYY-MM-DDTHH-MM-SS>-<slug>` (with a `-2`, `-3`, … suffix if that name is already taken, so runs never overwrite each other):

```
evals/reports/2026-05-22T12-30-00-extraction-v3/
├── results.json             # {"metadata": {...}, "status": ..., "results": {...}}
├── summary.md
└── per_item_breakdowns.md   # optional
```

`metadata` is provenance captured automatically (timestamp, git commit, embedding/LLM provider + model, prompt/schema version, argv, run label) — only those named fields, never a `Settings` dump, because baselines are committed. `status` is `completed`, or `failed` (plus an `error` key) when the run raised; `save_as_baseline(report_path, name)` refuses to promote a failed run, and `name` must be a single path segment (letters, digits, `.`, `_`, `-`).

Inside `results` an extraction run carries: `fixture_set` and the run-level `extraction_prompt_version`, `per_fixture` (per-field scores, review status, `confidence_overall`, keyed by fixture name — each *scored* entry also persists `recipes`, its `fixture_content_hash`, and `scored_recipe_index`), `aggregate` (per-field accuracy, missing-field counts, ready/needs_review), `judge` (per-fixture ratings + pass rate, or `null`), `agreement` (filled by `judge-alignment`, else `null`), and `calibration` (filled by `confidence-review`, else `null`). Be precise about the slices: `scores`/`confidence_overall` describe `recipes[scored_recipe_index]` (always item 0) only, while `recipes` holds **every** item the extractor returned, validated or not — the judge and alignment rate that full list.

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
