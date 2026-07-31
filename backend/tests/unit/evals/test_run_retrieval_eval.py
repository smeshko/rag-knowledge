"""Tests for ``run_retrieval_eval`` (Epic 16 Phase 16.1).

Hermetic by construction: the search caller is an injected fake returning
canned envelopes, the ``report_factory`` binds ``ReportRun`` to ``tmp_path``
with a settings stand-in, and the runner's settings are a stand-in too — no
live API, no DB, no provider construction, no ambient ``get_settings()``.

Uses the committed ``data/fixtures/queries/tests`` set: ``q1`` ("white bean
soup", qrels ``item_soup`` + ``item_stew``) and ``q2`` ("thai basil chicken",
qrels ``item_basil``).
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
    """Every attribute the runner and ``ReportRun`` metadata read; nothing more."""

    def __init__(
        self,
        *,
        search_default_limit: int = 10,
        reranking_enabled: bool = False,
    ) -> None:
        self.search_default_limit = search_default_limit
        self.embedding_provider = "fake"
        self.embedding_model = "fake-embedding"
        self.reranking_enabled = reranking_enabled
        self.rerank_provider = "openai"
        self.rerank_model = "gpt-4.1"
        self.rerank_top_n = 50
        # SettingsLike attributes for ReportRun's metadata capture.
        self.llm_provider = "openai"
        self.llm_model = "gpt-4.1"
        self.anthropic_llm_model = "claude-sonnet-4-6"


class _FakeSearch:
    """Canned envelopes keyed by query text; records every call's arguments."""

    def __init__(self, envelopes: dict[str, dict[str, Any]]) -> None:
        self.envelopes = envelopes
        self.calls: list[tuple[str, str, int]] = []

    async def __call__(self, query_text: str, *, mode: str, limit: int) -> dict[str, Any]:
        self.calls.append((query_text, mode, limit))
        return self.envelopes[query_text]


def _envelope(*item_ids: str) -> dict[str, Any]:
    return {"results": [{"item": {"id": item_id}} for item_id in item_ids]}


_ENVELOPES = {
    # q1: item_stew ranked 1st, item_soup 2nd (both expected).
    "white bean soup": _envelope("item_stew", "item_soup"),
    # q2: the expected item_basil ranked 2nd behind noise.
    "thai basil chicken": _envelope("item_noise", "item_basil"),
}


async def _run(
    tmp_path: Path,
    *,
    settings: _SettingsStandIn | None = None,
    search: _FakeSearch | None = None,
    k: int = 10,
    mode: str = "hybrid",
) -> tuple[ReportRun, _FakeSearch]:
    settings = settings or _SettingsStandIn()
    search = search or _FakeSearch(_ENVELOPES)
    report = await run_retrieval_eval(
        "tests",
        k=k,
        label="unit",
        mode=mode,
        search=search,
        report_factory=lambda label: ReportRun(label, reports_root=tmp_path, settings=settings),
        settings=settings,
    )
    return report, search


async def test_search_called_once_per_query_with_mode_and_limit(tmp_path: Path) -> None:
    _, search = await _run(tmp_path, mode="keyword", k=3)
    # limit = max(k, search_default_limit) = max(3, 10) = 10.
    assert sorted(search.calls) == [
        ("thai basil chicken", "keyword", 10),
        ("white bean soup", "keyword", 10),
    ]


async def test_limit_is_k_when_k_exceeds_default(tmp_path: Path) -> None:
    settings = _SettingsStandIn(search_default_limit=5)
    _, search = await _run(tmp_path, settings=settings, k=8)
    assert {call[2] for call in search.calls} == {8}


async def test_results_json_is_nested_epic_14_doc_with_run_block(tmp_path: Path) -> None:
    report, _ = await _run(tmp_path)
    doc = json.loads((report.path / "results.json").read_text(encoding="utf-8"))
    assert set(doc) >= {"metadata", "results"}
    assert doc["metadata"]["run_label"] == "unit"
    payload = doc["results"]
    assert payload["report_type"] == "retrieval"
    run_block = payload["run"]
    assert run_block["query_set"] == "tests"
    assert run_block["mode"] == "hybrid"
    assert run_block["k"] == 10
    assert run_block["limit"] == 10
    assert run_block["embedding_provider"] == "fake"
    assert run_block["embedding_model"] == "fake-embedding"
    assert run_block["reranking_enabled"] is False
    assert "rerank_provider" not in run_block  # only recorded when reranking is on


async def test_rerank_config_recorded_when_reranking_enabled(tmp_path: Path) -> None:
    settings = _SettingsStandIn(reranking_enabled=True)
    report, _ = await _run(tmp_path, settings=settings)
    doc = json.loads((report.path / "results.json").read_text(encoding="utf-8"))
    run_block = doc["results"]["run"]
    assert run_block["reranking_enabled"] is True
    assert run_block["rerank_provider"] == "openai"
    assert run_block["rerank_model"] == "gpt-4.1"
    assert run_block["rerank_top_n"] == 50


async def test_per_query_metrics_match_canned_rankings(tmp_path: Path) -> None:
    report, _ = await _run(tmp_path)
    payload = json.loads((report.path / "results.json").read_text(encoding="utf-8"))["results"]
    per_query = payload["per_query"]
    # q1: both expected items retrieved at ranks 1-2 → perfect scores.
    assert per_query["q1"]["metrics"]["recall_10"] == pytest.approx(1.0)
    assert per_query["q1"]["metrics"]["recip_rank"] == pytest.approx(1.0)
    assert per_query["q1"]["metrics"]["ndcg_cut_10"] == pytest.approx(1.0)
    # q2: item_basil at rank 2 → MRR 1/2.
    assert per_query["q2"]["metrics"]["recip_rank"] == pytest.approx(0.5)
    assert payload["aggregate"]["recip_rank"] == pytest.approx((1.0 + 0.5) / 2)


async def test_two_qrels_rows_fold_under_one_query_id(tmp_path: Path) -> None:
    # The fixture's q1 has two qrels rows (item_soup, item_stew); both must
    # land under q1 in the folded qrels — visible via expected_item_ranks.
    report, _ = await _run(tmp_path)
    payload = json.loads((report.path / "results.json").read_text(encoding="utf-8"))["results"]
    ranks = payload["per_query"]["q1"]["expected_item_ranks"]
    assert ranks == {"item_stew": 1, "item_soup": 2}


async def test_expected_item_absent_from_results_recorded_as_null(tmp_path: Path) -> None:
    envelopes = dict(_ENVELOPES)
    envelopes["thai basil chicken"] = _envelope("item_noise")  # item_basil not retrieved
    report, _ = await _run(tmp_path, search=_FakeSearch(envelopes))
    payload = json.loads((report.path / "results.json").read_text(encoding="utf-8"))["results"]
    assert payload["per_query"]["q2"]["expected_item_ranks"] == {"item_basil": None}
    assert payload["per_query"]["q2"]["metrics"]["recall_10"] == pytest.approx(0.0)


async def test_retrieved_ids_preserve_result_order(tmp_path: Path) -> None:
    report, _ = await _run(tmp_path)
    payload = json.loads((report.path / "results.json").read_text(encoding="utf-8"))["results"]
    assert payload["per_query"]["q1"]["retrieved_ids"] == ["item_stew", "item_soup"]
    assert payload["per_query"]["q2"]["retrieved_ids"] == ["item_noise", "item_basil"]


async def test_summary_md_written_with_headline_mode_and_rerank_state(tmp_path: Path) -> None:
    report, _ = await _run(tmp_path)
    summary = (report.path / "summary.md").read_text(encoding="utf-8")
    assert "NDCG@10" in summary
    assert "Recall@5" in summary
    assert "Recall@10" in summary
    assert "MRR" in summary
    assert "hybrid" in summary
    assert "reranking: off" in summary
    # Best/worst sample by NDCG@10.
    assert "q1" in summary
    assert "q2" in summary


@pytest.mark.parametrize("mode", ["hybrid", "keyword", "vector"])
async def test_all_three_modes_accepted_and_recorded(tmp_path: Path, mode: str) -> None:
    report, _ = await _run(tmp_path, mode=mode)
    doc = json.loads((report.path / "results.json").read_text(encoding="utf-8"))
    assert doc["results"]["run"]["mode"] == mode


async def test_invalid_mode_raises_before_any_search_call(tmp_path: Path) -> None:
    class _Boom:
        async def __call__(self, query_text: str, *, mode: str, limit: int) -> dict[str, Any]:
            raise AssertionError("search must not be called for an invalid mode")

    settings = _SettingsStandIn()
    with pytest.raises(ValueError, match="mode"):
        await run_retrieval_eval(
            "tests",
            k=10,
            label="unit",
            mode="bogus",
            search=_Boom(),
            report_factory=lambda label: ReportRun(
                label, reports_root=tmp_path, settings=settings
            ),
            settings=settings,
        )
    assert list(tmp_path.iterdir()) == []  # no report dir was created either
