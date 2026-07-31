"""Tests for the retrieval baseline diff (Epic 16 Phase 16.2).

Synthetic before/after payloads only — ``diff_retrieval`` is pure
(dict-in / RetrievalDiff-out), and ``diff_against_baseline`` is exercised over
real-shaped files under ``tmp_path``. No DB, no HTTP, no providers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from evals.reports import (
    DiffResult,
    _unwrap,
    diff_against_baseline,
    diff_retrieval,
)

_METRICS = ("ndcg_cut_10", "recall_5", "recall_10", "recip_rank")


def _run_block(**overrides: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "query_set": "golden",
        "mode": "hybrid",
        "k": 10,
        "limit": 10,
        "embedding_provider": "fake",
        "embedding_model": "fake-embedding",
        "reranking_enabled": False,
    }
    block.update(overrides)
    return block


def _query(
    ndcg: float, ranks: dict[str, int | None], retrieved: list[str] | None = None
) -> dict[str, Any]:
    return {
        "metrics": {
            "ndcg_cut_10": ndcg,
            "recall_5": 1.0,
            "recall_10": 1.0,
            "recip_rank": 1.0,
        },
        "expected_item_ranks": ranks,
        "retrieved_ids": retrieved or [],
    }


def _payload(
    per_query: dict[str, dict[str, Any]],
    *,
    aggregate: dict[str, float] | None = None,
    run: dict[str, Any] | None = None,
    report_type: str = "retrieval",
) -> dict[str, Any]:
    if aggregate is None:
        count = len(per_query) or 1
        aggregate = {
            measure: sum(q["metrics"][measure] for q in per_query.values()) / count
            for measure in _METRICS
        }
    return {
        "report_type": report_type,
        "run": run or _run_block(),
        "aggregate": aggregate,
        "per_query": per_query,
    }


# --- overall deltas + headline ----------------------------------------------


def test_overall_deltas_are_current_minus_baseline() -> None:
    baseline = _payload({"q1": _query(0.61, {"a": 1})})
    current = _payload({"q1": _query(0.57, {"a": 1})})
    diff = diff_retrieval(baseline, current)
    assert diff.overall["ndcg_cut_10"]["baseline"] == pytest.approx(0.61)
    assert diff.overall["ndcg_cut_10"]["current"] == pytest.approx(0.57)
    assert diff.overall["ndcg_cut_10"]["delta"] == pytest.approx(-0.04)


def test_headline_renders_regression_tag() -> None:
    baseline = _payload({"q1": _query(0.61, {"a": 1})})
    current = _payload({"q1": _query(0.57, {"a": 1})})
    diff = diff_retrieval(baseline, current)
    assert "NDCG@10: 0.61 → 0.57 [REGRESSION -0.04]" in diff.headline


def test_headline_renders_improvement_tag() -> None:
    baseline = _payload({"q1": _query(0.50, {"a": 1})})
    current = _payload({"q1": _query(0.60, {"a": 1})})
    diff = diff_retrieval(baseline, current)
    assert "NDCG@10: 0.50 → 0.60 [IMPROVEMENT +0.10]" in diff.headline


def test_headline_no_change_within_threshold() -> None:
    baseline = _payload({"q1": _query(0.600, {"a": 1})})
    current = _payload({"q1": _query(0.601, {"a": 1})})
    diff = diff_retrieval(baseline, current)
    assert "NDCG@10: 0.60 → 0.60 [no change]" in diff.headline
    assert diff.status == "no_change"


def test_headline_threshold_boundary_is_no_change() -> None:
    # |delta| == threshold does not tag; only strictly past it does.
    baseline = _payload({"q1": _query(0.600, {"a": 1})})
    current = _payload({"q1": _query(0.595, {"a": 1})})
    diff = diff_retrieval(baseline, current, headline_threshold=0.005)
    assert "[no change]" in diff.headline.split("\n")[0]


def test_headline_threshold_is_configurable() -> None:
    baseline = _payload({"q1": _query(0.600, {"a": 1})})
    current = _payload({"q1": _query(0.595, {"a": 1})})
    diff = diff_retrieval(baseline, current, headline_threshold=0.001)
    assert "[REGRESSION" in diff.headline


# --- per-query regressions ---------------------------------------------------


def test_rank_drop_of_exactly_three_flags_a_regression() -> None:
    baseline = _payload({"q1": _query(0.9, {"a": 2})})
    current = _payload({"q1": _query(0.9, {"a": 5})})
    diff = diff_retrieval(baseline, current)
    assert len(diff.regressions) == 1
    entry = diff.regressions[0]
    assert entry["query_id"] == "q1"
    assert entry["item_id"] == "a"
    assert entry["baseline_rank"] == 2
    assert entry["current_rank"] == 5
    assert entry["reason"] == "rank_drop"
    assert diff.status == "regression"


def test_rank_drop_of_exactly_two_does_not_flag() -> None:
    baseline = _payload({"q1": _query(0.9, {"a": 2})})
    current = _payload({"q1": _query(0.9, {"a": 4})})
    diff = diff_retrieval(baseline, current)
    assert diff.regressions == []


def test_expected_item_dropped_from_top_k_flags_a_regression() -> None:
    # Baseline rank 9 (inside k=10); current absent entirely.
    baseline = _payload({"q1": _query(0.9, {"a": 9})})
    current = _payload({"q1": _query(0.9, {"a": None})})
    diff = diff_retrieval(baseline, current)
    assert len(diff.regressions) == 1
    assert diff.regressions[0]["reason"] == "dropped_from_top_k"
    assert diff.regressions[0]["current_rank"] is None


def test_item_absent_in_both_runs_does_not_flag() -> None:
    baseline = _payload({"q1": _query(0.0, {"a": None})})
    current = _payload({"q1": _query(0.0, {"a": None})})
    diff = diff_retrieval(baseline, current)
    assert diff.regressions == []


# --- per-query deltas, intersection, ordering --------------------------------


def test_per_query_deltas_use_intersection_and_report_added_removed() -> None:
    baseline = _payload({"q1": _query(0.9, {"a": 1}), "q_old": _query(0.5, {"b": 1})})
    current = _payload({"q1": _query(0.8, {"a": 1}), "q_new": _query(0.7, {"c": 1})})
    diff = diff_retrieval(baseline, current)
    assert set(diff.per_query_ndcg_delta) == {"q1"}
    assert diff.per_query_ndcg_delta["q1"] == pytest.approx(-0.1)
    assert diff.added_query_ids == ["q_new"]
    assert diff.removed_query_ids == ["q_old"]


def test_worst_queries_sorted_most_negative_first_and_capped() -> None:
    baseline = _payload(
        {f"q{i}": _query(0.9, {"a": 1}) for i in range(7)} | {"q_up": _query(0.1, {"a": 1})}
    )
    current = _payload(
        {f"q{i}": _query(0.9 - (i + 1) * 0.05, {"a": 1}) for i in range(7)}
        | {"q_up": _query(0.9, {"a": 1})}
    )
    diff = diff_retrieval(baseline, current)
    assert [entry["query_id"] for entry in diff.worst_queries] == [
        "q6",
        "q5",
        "q4",
        "q3",
        "q2",
    ]  # default top 5, most negative first; the improved q_up is excluded
    top2 = diff_retrieval(baseline, current, top_n=2)
    assert [entry["query_id"] for entry in top2.worst_queries] == ["q6", "q5"]


# --- run-config comparability ------------------------------------------------


def test_warns_when_rerank_state_differs() -> None:
    baseline = _payload({"q1": _query(0.9, {"a": 1})}, run=_run_block(reranking_enabled=False))
    current = _payload({"q1": _query(0.9, {"a": 1})}, run=_run_block(reranking_enabled=True))
    diff = diff_retrieval(baseline, current)
    assert any("reranking_enabled" in warning for warning in diff.warnings)
    assert "not comparable" in diff.headline


def test_incomparable_runs_report_no_quality_verdict() -> None:
    # A rank drop *and* a metric drop are present, but the run blocks differ:
    # the verdict must be "incomparable", never the fake regression the
    # warning exists to prevent. The deltas are still computed and printed.
    baseline = _payload({"q1": _query(0.9, {"a": 1})})
    current = _payload({"q1": _query(0.5, {"a": 8})}, run=_run_block(mode="vector"))
    diff = diff_retrieval(baseline, current)
    assert diff.status == "incomparable"
    assert diff.regressions  # still surfaced as evidence
    assert diff.per_query_ndcg_delta["q1"] == pytest.approx(-0.4)


@pytest.mark.parametrize(("field", "value"), [("k", 5), ("limit", 50)])
def test_k_and_limit_are_comparability_fields(field: str, value: int) -> None:
    # A --k 5 run against a --k 10 baseline manufactures dropped_from_top_k
    # entries for ranks 6-10; a shallower limit depresses Recall@10 outright.
    baseline = _payload({"q1": _query(0.9, {"a": 1})})
    current = _payload({"q1": _query(0.9, {"a": 1})}, run=_run_block(**{field: value}))
    diff = diff_retrieval(baseline, current)
    assert any(field in warning for warning in diff.warnings)
    assert diff.status == "incomparable"


def test_no_warning_when_run_blocks_match() -> None:
    baseline = _payload({"q1": _query(0.9, {"a": 1})})
    current = _payload({"q1": _query(0.9, {"a": 1})})
    diff = diff_retrieval(baseline, current)
    assert diff.warnings == []


@pytest.mark.parametrize("field", ["query_set", "mode", "embedding_model"])
def test_warns_on_each_comparability_field(field: str) -> None:
    baseline = _payload({"q1": _query(0.9, {"a": 1})})
    current = _payload({"q1": _query(0.9, {"a": 1})}, run=_run_block(**{field: "other"}))
    diff = diff_retrieval(baseline, current)
    assert any(field in warning for warning in diff.warnings)


# --- envelope unwrapping and dispatch ----------------------------------------


def test_unwrap_baseline_envelope() -> None:
    payload = _payload({"q1": _query(0.9, {"a": 1})})
    doc = {
        "baseline_set_at": "2026-01-01T00:00:00+00:00",
        "run_label": "x",
        "source": {"metadata": {}, "status": "completed", "results": payload},
    }
    assert _unwrap(doc) == payload


def test_unwrap_report_results_doc() -> None:
    payload = _payload({"q1": _query(0.9, {"a": 1})})
    assert _unwrap({"metadata": {}, "status": "completed", "results": payload}) == payload


def test_unwrap_bare_payload_passes_through() -> None:
    payload = _payload({"q1": _query(0.9, {"a": 1})})
    assert _unwrap(payload) == payload


def _write_pair(
    tmp_path: Path, baseline_payload: dict[str, Any], current_payload: dict[str, Any]
) -> tuple[Path, Path]:
    baseline_file = tmp_path / "retrieval.json"
    baseline_file.write_text(
        json.dumps(
            {
                "baseline_set_at": "2026-01-01T00:00:00+00:00",
                "run_label": "before",
                "source": {
                    "metadata": {"run_label": "before"},
                    "status": "completed",
                    "results": baseline_payload,
                },
            }
        ),
        encoding="utf-8",
    )
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "results.json").write_text(
        json.dumps(
            {
                "metadata": {"run_label": "after"},
                "status": "completed",
                "results": current_payload,
            }
        ),
        encoding="utf-8",
    )
    return baseline_file, report_dir


def test_diff_against_baseline_dispatches_retrieval_and_projects_diffresult(
    tmp_path: Path,
) -> None:
    baseline_file, report_dir = _write_pair(
        tmp_path,
        _payload({"q1": _query(0.61, {"a": 1})}),
        _payload({"q1": _query(0.57, {"a": 5})}),
    )
    result = diff_against_baseline(baseline_file, report_dir)
    assert isinstance(result, DiffResult)
    assert result.status == "regression"
    assert "[REGRESSION" in result.summary
    assert result.changes  # per-query regression entries as dicts
    assert result.changes[0]["query_id"] == "q1"
    assert result.baseline_path == str(baseline_file)
    assert result.current_path == str(report_dir)


def test_diff_against_baseline_no_change_pair(tmp_path: Path) -> None:
    payload = _payload({"q1": _query(0.9, {"a": 1})})
    baseline_file, report_dir = _write_pair(tmp_path, payload, payload)
    result = diff_against_baseline(baseline_file, report_dir)
    assert result.status == "no_change"
    assert result.changes == []


def test_extraction_typed_payload_is_not_routed_to_retrieval_diff(tmp_path: Path) -> None:
    extraction = {"report_type": "extraction", "results_by_fixture": {}}
    baseline_file, report_dir = _write_pair(tmp_path, extraction, extraction)
    result = diff_against_baseline(baseline_file, report_dir)
    assert result.status == "not_implemented"
    assert "NDCG" not in result.summary


def test_missing_inputs_still_raise_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        diff_against_baseline(tmp_path / "missing.json", tmp_path)


def test_report_directory_without_results_json_raises(tmp_path: Path) -> None:
    baseline_file, report_dir = _write_pair(
        tmp_path, _payload({"q1": _query(0.9, {"a": 1})}), _payload({})
    )
    (report_dir / "results.json").unlink()
    with pytest.raises(FileNotFoundError, match="report results not found"):
        diff_against_baseline(baseline_file, report_dir)


def test_pointing_at_results_json_instead_of_its_directory_raises(tmp_path: Path) -> None:
    # Previously died with an unhandled NotADirectoryError, which the CLI let
    # through as exit 1 — indistinguishable from "a regression was found".
    baseline_file, report_dir = _write_pair(
        tmp_path,
        _payload({"q1": _query(0.9, {"a": 1})}),
        _payload({"q1": _query(0.9, {"a": 1})}),
    )
    with pytest.raises(FileNotFoundError, match="report directory not found"):
        diff_against_baseline(baseline_file, report_dir / "results.json")


def test_failed_run_is_refused_rather_than_diffed_clean(tmp_path: Path) -> None:
    # A failed run keeps whatever partial payload it wrote; diffing it would
    # report a confident "no change" over results that were never produced.
    payload = _payload({"q1": _query(0.9, {"a": 1})})
    baseline_file, report_dir = _write_pair(tmp_path, payload, payload)
    doc = json.loads((report_dir / "results.json").read_text(encoding="utf-8"))
    doc["status"] = "failed"
    (report_dir / "results.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="failed run"):
        diff_against_baseline(baseline_file, report_dir)


def test_report_type_mismatch_is_refused(tmp_path: Path) -> None:
    # Retrieval baseline vs extraction report previously fell through to the
    # placeholder and exited 0 — a silently green-passed invalid comparison.
    baseline_file, report_dir = _write_pair(
        tmp_path,
        _payload({"q1": _query(0.9, {"a": 1})}),
        {"report_type": "extraction", "results_by_fixture": {}},
    )
    with pytest.raises(ValueError, match="report type mismatch"):
        diff_against_baseline(baseline_file, report_dir)


def _diff_with_current(tmp_path: Path, current: dict[str, Any]) -> DiffResult:
    baseline_file, report_dir = _write_pair(
        tmp_path, _payload({"q1": _query(0.9, {"a": 1})}), current
    )
    return diff_against_baseline(baseline_file, report_dir)


def test_missing_aggregate_is_refused_not_read_as_zeroes(tmp_path: Path) -> None:
    # `.get("aggregate", {})` defaulted every measure to 0.0, so a baseline or
    # report without it reported a fabricated improvement and exited 0.
    current = _payload({"q1": _query(0.9, {"a": 1})})
    del current["aggregate"]
    with pytest.raises(ValueError, match="'aggregate' is missing or not an object"):
        _diff_with_current(tmp_path, current)


def test_missing_per_query_is_refused_not_read_as_no_change(tmp_path: Path) -> None:
    current = _payload({"q1": _query(0.9, {"a": 1})})
    del current["per_query"]
    with pytest.raises(ValueError, match="'per_query' is missing or not an object"):
        _diff_with_current(tmp_path, current)


def test_wrongly_typed_run_block_is_refused_not_an_attributeerror(tmp_path: Path) -> None:
    # `run: []` used to leak AttributeError, which the CLI surfaced as exit 1 —
    # indistinguishable from a real regression.
    current = _payload({"q1": _query(0.9, {"a": 1})})
    current["run"] = []
    with pytest.raises(ValueError, match="'run' is missing or not an object"):
        _diff_with_current(tmp_path, current)


def test_nan_metric_is_refused_not_reported_as_no_change(tmp_path: Path) -> None:
    # json.loads parses the bare `NaN` token, and every NaN comparison is
    # False, so `_headline_tag` returned "[no change]" with full confidence.
    current = _payload({"q1": _query(0.9, {"a": 1})})
    current["aggregate"]["ndcg_cut_10"] = float("nan")
    with pytest.raises(ValueError, match="ndcg_cut_10 is not finite"):
        _diff_with_current(tmp_path, current)


def test_infinite_per_query_metric_is_refused(tmp_path: Path) -> None:
    current = _payload({"q1": _query(0.9, {"a": 1})})
    current["per_query"]["q1"]["metrics"]["ndcg_cut_10"] = float("inf")
    with pytest.raises(ValueError, match="ndcg_cut_10 is not finite"):
        _diff_with_current(tmp_path, current)


def test_non_numeric_metric_is_refused(tmp_path: Path) -> None:
    current = _payload({"q1": _query(0.9, {"a": 1})})
    current["aggregate"]["recall_5"] = "0.9"
    with pytest.raises(ValueError, match="recall_5 is not a number"):
        _diff_with_current(tmp_path, current)


def test_non_integer_expected_item_rank_is_refused(tmp_path: Path) -> None:
    current = _payload({"q1": _query(0.9, {"a": 1})})
    current["per_query"]["q1"]["expected_item_ranks"]["a"] = "1"
    with pytest.raises(ValueError, match="neither an integer rank nor null"):
        _diff_with_current(tmp_path, current)


def test_null_expected_item_rank_stays_valid(tmp_path: Path) -> None:
    # None is the legitimate "not retrieved" encoding — it must not be rejected.
    result = _diff_with_current(tmp_path, _payload({"q1": _query(0.9, {"a": None})}))
    assert result.status == "regression"  # 'a' dropped out of the top-k


def test_malformed_json_raises_a_caller_facing_error(tmp_path: Path) -> None:
    baseline_file, report_dir = _write_pair(
        tmp_path,
        _payload({"q1": _query(0.9, {"a": 1})}),
        _payload({"q1": _query(0.9, {"a": 1})}),
    )
    baseline_file.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        diff_against_baseline(baseline_file, report_dir)
