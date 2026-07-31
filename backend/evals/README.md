# evals/

Offline evaluation harness for retrieval and extraction quality.

> **Working directory.** This is a `backend/` + `frontend/` monorepo. Everything
> below — every `uv run rag-evals …` command and every relative path such as
> `data/fixtures/` (meaning `backend/data/fixtures/`) — is run from and relative
> to **`backend/`**.

Background reading: [`docs/architecture/12-evaluation-and-testing.md`](../docs/architecture/12-evaluation-and-testing.md)
and [`docs/architecture/13-implementation-decisions.md`](../docs/architecture/13-implementation-decisions.md)
(topics 10–12: report/baseline storage, retrieval metrics, LLM-judge fixtures).

## Running the harness

The CLI is installed as the `rag-evals` console script (`uv sync` links it):

```bash
uv run rag-evals --help
uv run rag-evals extraction --fixtures <set> [--judge <name>]   # stub — Epic 15
uv run rag-evals retrieval --queries <set> [--k 10]             # stub — Epic 16
uv run rag-evals judge-alignment --judge <name>                 # stub — Epic 15
uv run rag-evals confidence-review                              # stub — Epic 15
uv run rag-evals diff <baseline_path> <new_report_path>         # skeleton — Epics 15/16
```

**Every subcommand is currently a scaffold.** Epic 14 ships the workflow
skeleton (CLI shape, fixture loaders, report/baseline plumbing); the commands print
`not implemented yet` and exit 0. Extraction scoring and judge tooling land in
Epic 15, retrieval metrics (`pytrec_eval`) in Epic 16. Nothing here calls an
LLM or embedding API today.

## Fixture layout

Loaders (`evals/fixtures.py`) read from `data/fixtures/`:

```
data/fixtures/
├── synthetic_recipes/<set>/<name>/   # one dir per recipe fixture
│   ├── source.md                     # raw source text fed to extraction
│   ├── expected.json                 # golden recipe.v1 extraction output
│   └── notes.md                      # optional: why this fixture is tricky
├── queries/<set>/
│   ├── queries.tsv                   # query_id <TAB> query_text
│   └── qrels.tsv                     # query_id <TAB> knowledge_item_id <TAB> relevance
├── judge_prompts/<name>.md           # LLM-judge prompt (optional leading "version:" line)
├── judge_alignment/<fixture_id>.json # human-vs-judge agreement records
└── private/                          # gitignored — personal/copyrighted material
```

None of the four fixture subdirectories exist yet — fixture *content* is
authored alongside the first real ingestion runs. The loaders degrade
gracefully: absent recipe/query sets load as empty, an absent alignment record
loads as `None`. Only `load_judge_prompt` raises (`FileNotFoundError`) — asking
for a *named* prompt that does not exist is a caller error.

## Adding a fixture

- **Synthetic recipe**: create `data/fixtures/synthetic_recipes/<set>/<name>/`
  with `source.md` (raw text) and `expected.json` (the golden `recipe.v1`
  JSON). Add `notes.md` if the fixture targets a specific failure mode.
- **Query set**: create `data/fixtures/queries/<set>/` with `queries.tsv` and
  `qrels.tsv` (tab-separated; blank lines and `#`-prefixed comment lines are
  skipped; `relevance` is an int, `1` for binary relevance today).
- **Judge prompt**: add `data/fixtures/judge_prompts/<name>.md`. An optional
  leading `version:` line (bare or inside `---` front-matter) is parsed into
  `JudgePrompt.version`.
- **Judge alignment record**: written programmatically via
  `evals.fixtures.save_judge_alignment(record)` — one
  `judge_alignment/<fixture_id>.json` per fixture with
  `{fixture_id, human_rating, judge_rating, agreement_status, run_metadata}`.

Fixture shapes are the Pydantic models in `evals/models.py`.

## What to commit vs gitignore

Per doc 12 § 5 / § 11 and the existing `backend/.gitignore` rules:

| Data | Policy |
|---|---|
| Synthetic + public-domain fixtures (`data/fixtures/…`) | **commit** |
| Personal/copyrighted material (`data/fixtures/private/`) | **gitignored** |
| Judge prompts (`data/fixtures/judge_prompts/`) | **commit** |
| Judge alignment data (`data/fixtures/judge_alignment/`) | **commit** |
| Per-run reports (`evals/reports/<run>/`) | **gitignored** (only `.gitkeep` tracked) |
| Baselines (`evals/baselines/<name>.json`) | **commit** |
