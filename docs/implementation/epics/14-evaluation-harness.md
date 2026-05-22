# Epic 14 — Evaluation Harness

**Status**: Blocked (depends on Epic 1)

## Overview

Build the scaffolding for the evaluation workflow: a top-level `evals/` directory, a `typer` CLI for invocation, fixture-loading helpers, and a report writer that produces both Markdown summaries and JSON results into a timestamped directory. This epic ships the *workflow*; Epics 15 and 16 ship the actual extraction and retrieval evaluation logic that uses it.

## Architecture references

- [12 — Evaluation and Testing](../../architecture/12-evaluation-and-testing.md) — full evaluation methodology
- [12 § Evaluations vs Tests](../../architecture/12-evaluation-and-testing.md#first-principle-rag-quality-needs-evaluation-not-just-unit-tests) — workflow separation principle
- [13 — Implementation Decisions, topics 10, 11, 12](../../architecture/13-implementation-decisions.md#10-evaluation-harness-layout) — directory layout, CLI, fixture taxonomy

## Dependencies

- Epic 1 (project scaffold, including `evals/` and `data/fixtures/` skeletons)

## Out of scope

- Extraction-evaluation scoring logic (Epic 15)
- Retrieval-evaluation metric computation (Epic 16)
- Specific synthetic fixtures / golden queries / judge prompts — those are *content*, created alongside the first real ingestion runs

---

## Phase 14.1 — `typer` CLI scaffold + fixture loading

**Goal**: A working `rag-evals` CLI exposed as a console script, with subcommand stubs and shared fixture-loading helpers.

### What to build

- **`evals/__init__.py`** + **`evals/cli.py`** — `typer` app with subcommands:
  - `rag-evals extraction --fixtures <name> [--judge <name>]` — placeholder for Epic 15
  - `rag-evals retrieval --queries <name> [--k 10]` — placeholder for Epic 16
  - `rag-evals judge-alignment --judge <name>` — placeholder for Epic 15
  - `rag-evals confidence-review` — placeholder for Epic 15
  - `rag-evals diff <baseline_path> <new_report_path>` — regression diff (skeleton; full logic comes with Epics 15/16)
  - Each subcommand is a stub that prints "not implemented yet" so the CLI shape exists from this epic onward
- Register the CLI in `pyproject.toml` as a console script entry point: `rag-evals = "evals.cli:app"`
- **`evals/fixtures.py`** — loaders for each fixture kind per [doc 12 § 4](../../architecture/12-evaluation-and-testing.md#4-golden-fixtures):
  - `load_recipe_fixtures(fixture_set: str) -> list[RecipeFixture]` — reads `data/fixtures/synthetic_recipes/<name>/` (each subdirectory has `source.md`, `expected.json`, optional `notes.md`)
  - `load_query_fixtures(fixture_set: str) -> QueryFixtureSet` — reads `data/fixtures/queries/<name>/queries.tsv` and `qrels.tsv` per [doc 12 § Retrieval fixtures](../../architecture/12-evaluation-and-testing.md#retrieval-fixtures)
  - `load_judge_prompt(name: str) -> JudgePrompt` — reads `data/fixtures/judge_prompts/<name>.md`
  - `load_judge_alignment(name: str, fixture_id: str) -> JudgeAlignmentRecord | None` — reads `data/fixtures/judge_alignment/<fixture_id>.json` if present
  - `save_judge_alignment(record: JudgeAlignmentRecord)` — writes updates back
- Pydantic models for each fixture kind in `evals/models.py`
- `evals/README.md` documents:
  - How to run each subcommand
  - The fixture directory layout
  - How to add a new fixture
  - When to commit vs gitignore (synthetic + public-domain → commit; private → gitignore; judge prompts → commit; judge alignment data → commit per [doc 12 § 5](../../architecture/12-evaluation-and-testing.md#tracking-the-judge-over-time))

### Acceptance criteria

- [ ] `rag-evals --help` lists all subcommands
- [ ] Each subcommand stub prints "not implemented yet"
- [ ] Fixture loaders work against an empty or single-fixture dataset without crashing
- [ ] Fixture Pydantic models match doc 12's described shapes
- [ ] `evals/README.md` is accurate and useful as a contributor's first read
- [ ] Unit tests for fixture loaders covering empty, single, and multiple fixtures

### Validation

`uv run rag-evals --help` and `uv run rag-evals extraction --fixtures synthetic` both run without import errors.

---

## Phase 14.2 — Report writer + baseline storage

**Goal**: A reusable report writer that produces Markdown summaries and JSON results into a timestamped run directory under `evals/reports/`, with helpers for committed baselines and regression diffs.

### What to build

- **`evals/reports.py`**:
  - `ReportRun` context manager / class:
    - Created with a run label (e.g., `extraction-gpt4-prompt-v3`)
    - Generates a timestamped directory: `evals/reports/2026-05-22T12-30-00-<label>/`
    - Writes `summary.md` (human-readable, populated by callers)
    - Writes `results.json` (machine-readable, populated by callers)
    - Optionally writes `per_item_breakdowns.md` for deep inspection
    - Captures run metadata: timestamp, git commit hash, embedding model, LLM model + prompt version + schema version, command-line args
  - `save_as_baseline(report_path: Path, baseline_name: str)`:
    - Copies `results.json` from a run directory into `evals/baselines/<baseline_name>.json`
    - Adds a small metadata header (`baseline_set_at`, `run_label`)
  - `diff_against_baseline(baseline_path: Path, current_report_path: Path) -> DiffResult` — skeleton signature; concrete diff logic lands in Epics 15/16. For now, return a placeholder structure.
- **CLI integration**:
  - `rag-evals diff <baseline_path> <new_report_path>` invokes `diff_against_baseline` and prints a placeholder summary
  - Eventually Epics 15 and 16 fill in the actual diff content
- **`evals/baselines/.gitkeep`** committed so the folder exists; `evals/reports/` gitignored except a `.gitkeep`
- Unit tests for `ReportRun` covering directory creation, metadata capture, and file writes
- Unit tests for `save_as_baseline`

### Acceptance criteria

- [ ] `ReportRun` creates timestamped directories with correct naming
- [ ] `summary.md` and `results.json` both written
- [ ] Run metadata (timestamp, git commit, model versions) captured automatically
- [ ] `save_as_baseline` copies a result JSON into the baselines folder with the metadata header
- [ ] `rag-evals diff` runs without crashing on real report + baseline pairs
- [ ] `.gitignore` rules verified (per-run reports gitignored, baselines committed)

### Validation

Manually invoke the CLI to create a test report run; verify the directory structure and file contents.

---

## Epic-level acceptance criteria

- [ ] `rag-evals` CLI exposed as a console script with all subcommand stubs
- [ ] Fixture loaders for recipes, queries+qrels, judge prompts, judge alignment
- [ ] Report writer with Markdown + JSON output and timestamped directories
- [ ] Baseline save + diff scaffolding ready for Epics 15/16 to fill in
- [ ] `evals/README.md` documents the workflow
- [ ] Gitignore rules correct (reports gitignored except baselines)
- [ ] Status in [`EPICS.md`](../EPICS.md) updated; Epic 15 unblocked once Epic 9 is done; Epic 16 unblocked once Epic 13 is done
