"""Tests for the per-query breakdown report (Epic 16 Phase 16.2).

Render-level: a fake search caller returns canned envelopes — one query's
envelope carries a ``debug`` block (with the merged ``RetrievalDebugInfo``
keys), the other omits the key entirely, mirroring merged ``routes/search.py``
which pops ``debug`` when gating is off. No DB, no HTTP, no providers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from evals.reports import ReportRun
from evals.retrieval import run_retrieval_eval

pytestmark = pytest.mark.asyncio


class _SettingsStandIn:
    search_default_limit = 10
    embedding_provider = "fake"
    embedding_model = "fake-embedding"
    reranking_enabled = False
    rerank_provider = "openai"
    rerank_model = "gpt-4.1"
    rerank_top_n = 50
    llm_provider = "openai"
    llm_model = "gpt-4.1"
    anthropic_llm_model = "claude-sonnet-4-6"


def _result(
    item_id: str,
    *,
    chunks: list[tuple[str, str, float]],
    citations: list[str],
) -> dict[str, Any]:
    return {
        "item": {"id": item_id},
        "matched_chunks": [
            {"chunk_id": chunk_id, "chunk_type": chunk_type, "score": score}
            for chunk_id, chunk_type, score in chunks
        ],
        "source_citations": [
            {"source_span_id": f"span_{label}", "label": label, "locator": None}
            for label in citations
        ],
    }


_DEBUG = {
    "retrieval_mode": "hybrid",
    "normalized_query": "white bean soup",
    "embedding_model": "fake-embedding",
    "keyword_top_k": 40,
    "vector_top_k": 40,
    "keyword_candidates": 2,
    "vector_candidates": 2,
    "merged_candidates": 2,
    "grouped_items": 2,
    "rerank_applied": False,
    # Doc-prose keys that do NOT exist on merged RetrievalDebugInfo — the
    # renderer must never emit them even if a payload smuggles them in.
    "filters_applied": {"bogus": True},
    "chunk_type_boosts": {"bogus": 2.0},
}

_ENVELOPES: dict[str, dict[str, Any]] = {
    # q1: two results, with debug present.
    "white bean soup": {
        "results": [
            _result(
                "item_stew",
                chunks=[("ch1", "recipe_title", 0.91), ("ch2", "recipe_full", 0.55)],
                citations=["page 42"],
            ),
            _result(
                "item_soup",
                chunks=[("ch3", "recipe_full", 0.40)],
                citations=["page 7", "page 8"],
            ),
        ],
        "debug": _DEBUG,
    },
    # q2: no debug key at all (production / dev-off shape).
    "thai basil chicken": {
        "results": [
            _result("item_basil", chunks=[("ch4", "recipe_full", 0.62)], citations=["page 3"]),
        ],
    },
}


class _FakeSearch:
    async def __call__(self, query_text: str, *, mode: str, limit: int) -> dict[str, Any]:
        return _ENVELOPES[query_text]


async def _run(tmp_path: Path, *, k: int = 10) -> ReportRun:
    settings = _SettingsStandIn()
    return await run_retrieval_eval(
        "tests",
        k=k,
        label="per-query",
        search=_FakeSearch(),
        report_factory=lambda label: ReportRun(label, reports_root=tmp_path, settings=settings),
        settings=settings,
    )


async def test_per_query_md_written_under_exact_filename(tmp_path: Path) -> None:
    report = await _run(tmp_path)
    assert (report.path / "per_query.md").is_file()


async def test_renders_query_text_expected_items_and_topk_rows(tmp_path: Path) -> None:
    report = await _run(tmp_path)
    md = (report.path / "per_query.md").read_text(encoding="utf-8")
    assert "white bean soup" in md
    assert "thai basil chicken" in md
    # Expected (qrels) item ids for q1 (two rows folded under one query).
    assert "item_soup" in md
    assert "item_stew" in md
    # Ranked rows with score = max(matched_chunks[].score).
    assert "| 1 | item_stew | 0.9100 |" in md
    assert "| 2 | item_soup | 0.4000 |" in md
    # Matched chunk types and citation labels.
    assert "recipe_title, recipe_full" in md
    assert "page 42" in md
    assert "page 7; page 8" in md


async def test_header_states_score_is_max_matched_chunk_score(tmp_path: Path) -> None:
    report = await _run(tmp_path)
    md = (report.path / "per_query.md").read_text(encoding="utf-8")
    assert "max(matched_chunks[].score)" in md


async def test_debug_rendered_only_when_present_and_only_real_keys(tmp_path: Path) -> None:
    report = await _run(tmp_path)
    md = (report.path / "per_query.md").read_text(encoding="utf-8")
    q1_section, q2_section = md.split("thai basil chicken", 1)
    # q1 carries the debug block with merged RetrievalDebugInfo keys...
    assert "rerank_applied" in q1_section
    assert "normalized_query" in q1_section
    assert "merged_candidates" in q1_section
    # ...but never the doc-prose keys that don't exist on the merged schema.
    assert "filters_applied" not in md
    assert "chunk_type_boosts" not in md
    # q2's envelope has no debug key: no debug section, no KeyError, no
    # empty stub.
    assert "Debug" not in q2_section


async def test_topk_rows_are_capped_at_k(tmp_path: Path) -> None:
    report = await _run(tmp_path, k=1)
    md = (report.path / "per_query.md").read_text(encoding="utf-8")
    assert "| 1 | item_stew |" in md
    assert "| 2 | item_soup |" not in md  # beyond top-k


async def test_results_json_still_carries_report_type_and_ranks(tmp_path: Path) -> None:
    # 16.1 wrote these; the per-query collection refactor must not lose them.
    report = await _run(tmp_path)
    payload = json.loads((report.path / "results.json").read_text(encoding="utf-8"))["results"]
    assert payload["report_type"] == "retrieval"
    assert payload["per_query"]["q1"]["expected_item_ranks"] == {"item_stew": 1, "item_soup": 2}
    assert payload["per_query"]["q2"]["expected_item_ranks"] == {"item_basil": 1}
    assert payload["per_query"]["q1"]["metrics"]["recip_rank"] == pytest.approx(1.0)
