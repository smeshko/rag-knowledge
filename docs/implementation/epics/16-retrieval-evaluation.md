# Epic 16 — Retrieval Evaluation

**Status**: Blocked (depends on Epics 13, 14)

## Overview

Implement retrieval evaluation: run a golden query set against the live search API, compute NDCG@10 (headline), Recall@5, Recall@10, and MRR (secondaries) via `pytrec_eval`, and produce per-query breakdown reports plus regression diffs against committed baselines.

## Architecture references

- [12 — Evaluation and Testing § 7, 10](../../architecture/12-evaluation-and-testing.md#7-retrieval-evaluation) — metric set, fixture format, report shape
- [13 — Implementation Decisions, topic 11](../../architecture/13-implementation-decisions.md#11-retrieval-evaluation-metrics) — NDCG@10 + Recall@K + MRR via `pytrec_eval`, BEIR-style fixtures
- [07 — Retrieval Behavior § Debug Information](../../architecture/07-retrieval-behavior.md#12-debug-information) — debug payload used to inspect per-query results

## Dependencies

- Epic 13 (search API to query against)
- Epic 14 (CLI, fixture loaders, report writer)

## Out of scope

- Specific golden query content (created alongside this work as ingested cookbooks accumulate)
- Reranking eval (reranking itself deferred)
- Cross-category routing (deferred)

---

## Phase 16.1 — `pytrec_eval` integration + metric computation

**Goal**: `rag-evals retrieval` runs a query set against the search API, collects ranked results, and produces NDCG@10 / Recall@5 / Recall@10 / MRR via `pytrec_eval`.

### What to build

- **`evals/scripts/retrieval_eval.py`** (or `evals/retrieval.py`):
  - `run_retrieval_eval(query_set: str, k: int, label: str, mode: str = "hybrid") -> ReportRun`:
    - Loads query fixtures via Epic 14's `load_query_fixtures`
    - For each query: call `POST /api/v1/search` (via in-process function call or HTTP) with the configured mode and limit (≥ k)
    - Collect the ranked `KnowledgeItem.id` list
    - Build a `run` dict for `pytrec_eval`: `{query_id: {knowledge_item_id: score, ...}}`
- **`evals/metrics/retrieval.py`**:
  - Wraps `pytrec_eval` to compute:
    - `ndcg_cut_10` (headline)
    - `recall_5`, `recall_10`
    - `recip_rank` (MRR)
  - Returns per-query and aggregate metrics
- **Report structure**:
  - `results.json`: per-query metrics + aggregates + run metadata (query_set name, mode, k, embedding model, etc.)
  - `summary.md`: aggregate headline + secondaries; sample of best/worst queries
- Caller can supply `mode` (`hybrid` / `keyword` / `vector`) to run the same eval across modes for comparison

### Acceptance criteria

- [ ] `rag-evals retrieval --queries synthetic --k 10 --mode hybrid --label initial` produces a timestamped report
- [ ] NDCG@10, Recall@5, Recall@10, MRR all computed correctly (cross-check against a tiny hand-computable example in tests)
- [ ] Three modes (hybrid/keyword/vector) each runnable; results clearly labeled
- [ ] Embedding model and other run metadata captured in the report
- [ ] Unit tests verify metric computation against known qrels/run pairs

### Validation

Construct a tiny query set with known expected items, ingest fixtures so those items exist, run the eval, and verify metric values manually for a couple of queries.

---

## Phase 16.2 — Per-query breakdown reports + regression diff against baseline

**Goal**: Each retrieval eval produces a detailed per-query breakdown (top results, matched chunks, citations) plus a regression diff highlighting queries whose ranking changed since the committed baseline.

### What to build

- **Per-query breakdown** written to the report directory (e.g., `per_query.md`):
  - For each query: query text, expected relevant items, retrieved top-10 with scores, matched chunk types, citations
  - Optionally fetches and includes the search debug payload (when dev mode is enabled) for deeper inspection per [doc 7 § 12](../../architecture/07-retrieval-behavior.md#12-debug-information)
- **Baseline-diff logic** in `evals/reports.py` (extending Epic 14's skeleton with retrieval-specific diff):
  - For retrieval reports: diff NDCG@10 (overall + per-query), Recall@K, MRR
  - Flag per-query regressions (e.g., expected item dropped out of top 10, rank dropped by ≥3)
  - Print a top-line headline diff: "NDCG@10: 0.61 → 0.57 [REGRESSION -0.04]" or similar
  - Highlight queries with the biggest negative deltas
- **`rag-evals diff <baseline_path> <new_report_path>`** now supports retrieval reports as well as extraction reports
- **Save-as-baseline workflow**:
  - When a new retrieval eval is satisfactory, run `rag-evals diff` to inspect the change
  - If satisfactory, run `rag-evals save-baseline <report_path> --name retrieval` (or directly via `evals/baselines/retrieval.json`)
- Tests:
  - Unit tests for the diff logic with synthetic before/after JSON
  - Integration test: run an eval against an ingested fixture; save as baseline; modify a chunk-type boost in `Settings`; re-run; assert the diff reflects the change

### Acceptance criteria

- [ ] Per-query breakdown report produced
- [ ] Debug payload optionally included when dev mode is enabled
- [ ] Regression diff produces a readable top-line + per-query delta list
- [ ] `rag-evals diff` works for both extraction and retrieval reports
- [ ] Baseline save workflow documented
- [ ] Test coverage for diff logic

### Validation

End-to-end: ingest two cookbooks, build a small golden query set, run the eval, save baseline, change a config knob (e.g., disable vector boost on `recipe_summary`), re-run, observe a regression in the diff that points at specific queries.

---

## Epic-level acceptance criteria

- [ ] `rag-evals retrieval` produces NDCG@10 + Recall@5 + Recall@10 + MRR across a query set
- [ ] Three retrieval modes (hybrid / keyword / vector) runnable for comparison
- [ ] Per-query breakdown report includes matched chunks and citations
- [ ] Regression diff against a committed baseline works for retrieval reports
- [ ] `rag-evals diff` handles both extraction and retrieval reports
- [ ] Initial small golden query set used in tests (real content comes later as cookbooks are ingested)
- [ ] Status in [`EPICS.md`](../EPICS.md) updated; backend MVP feature-complete
