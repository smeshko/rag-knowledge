"""Tests for the extraction baseline diff (Epic 15 Phase 15.3).

Both inputs are built in their real wrapped shapes — a run dir whose
``results.json`` is the ``{"metadata", "results"}`` envelope, and a baseline
file with the whole document nested under ``source`` — so the unwrapping in
``diff_against_baseline`` is actually exercised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from evals.reports import FLAG_REGRESSION, diff_against_baseline


def _results(
    *,
    field_accuracy: dict[str, float | None] | None = None,
    counts: dict[str, int] | None = None,
    judge_pass_rate: float | None = None,
    with_judge: bool = False,
    agreement_rate: float | None = None,
    with_agreement: bool = False,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "field_accuracy": field_accuracy
        if field_accuracy is not None
        else {"title_normalized": 0.97, "yield": 0.92},
    }
    aggregate.update(counts if counts is not None else {"ready": 115, "needs_review": 3})
    results: dict[str, Any] = {
        "fixture_set": "smoke",
        "per_fixture": [],
        "aggregate": aggregate,
        "judge": None,
        "agreement": None,
        "calibration": None,
    }
    if with_judge:
        results["judge"] = {
            "name": "summary_quality",
            "version": "v1",
            "pass_rate": judge_pass_rate,
        }
    if with_agreement:
        results["agreement"] = {"judge_name": "summary_quality", "agreement_rate": agreement_rate}
    return results


def _write_run(tmp_path: Path, results: dict[str, Any]) -> Path:
    run_dir = tmp_path / "report"
    run_dir.mkdir()
    doc = {"metadata": {"run_label": "current"}, "status": "completed", "results": results}
    (run_dir / "results.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return run_dir


def _write_baseline(tmp_path: Path, results: dict[str, Any]) -> Path:
    path = tmp_path / "extraction.json"
    doc = {
        "baseline_set_at": "2026-01-01T00:00:00+00:00",
        "run_label": "baseline",
        "source": {
            "metadata": {"run_label": "baseline"},
            "status": "completed",
            "results": results,
        },
    }
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _change(result: Any, metric: str) -> dict[str, Any]:
    return next(change for change in result.changes if change["metric"] == metric)


def test_accuracy_drop_is_flagged_regression(tmp_path: Path) -> None:
    baseline = _write_baseline(
        tmp_path, _results(field_accuracy={"title_normalized": 0.97, "yield": 0.92})
    )
    run_dir = _write_run(
        tmp_path, _results(field_accuracy={"title_normalized": 0.90, "yield": 0.92})
    )
    result = diff_against_baseline(baseline, run_dir)
    change = _change(result, "field_accuracy.title_normalized")
    assert change["flag"] == FLAG_REGRESSION
    assert change["delta"] == pytest.approx(-0.07)
    assert result.status == "regressions_detected"
    assert "[REGRESSION] field_accuracy.title_normalized: 0.97 -> 0.90" in result.summary
    assert "1 regression(s) detected." in result.summary


def test_judge_pass_rate_and_agreement_drops_are_regressions(tmp_path: Path) -> None:
    baseline = _write_baseline(
        tmp_path,
        _results(
            with_judge=True, judge_pass_rate=0.9, with_agreement=True, agreement_rate=0.91
        ),
    )
    run_dir = _write_run(
        tmp_path,
        _results(
            with_judge=True, judge_pass_rate=0.7, with_agreement=True, agreement_rate=0.80
        ),
    )
    result = diff_against_baseline(baseline, run_dir)
    assert _change(result, "judge.pass_rate")["flag"] == FLAG_REGRESSION
    assert _change(result, "agreement.rate")["flag"] == FLAG_REGRESSION
    assert result.status == "regressions_detected"


def test_improvement_and_sub_tolerance_moves_are_not_flagged(tmp_path: Path) -> None:
    baseline = _write_baseline(
        tmp_path, _results(field_accuracy={"title_normalized": 0.90, "yield": 0.92})
    )
    run_dir = _write_run(
        tmp_path, _results(field_accuracy={"title_normalized": 0.97, "yield": 0.915})
    )
    result = diff_against_baseline(baseline, run_dir)
    assert _change(result, "field_accuracy.title_normalized")["flag"] == "improved"
    assert _change(result, "field_accuracy.yield")["flag"] == "unchanged"  # |Δ| ≤ 0.01
    assert result.status == "ok"
    assert "No regressions detected." in result.summary


@pytest.mark.parametrize(
    ("baseline_value", "current_value"),
    [(0.92, 0.91), (0.91, 0.92), (1.0, 0.99), (0.3, 0.29)],
)
def test_moves_of_exactly_the_tolerance_are_unchanged(
    tmp_path: Path, baseline_value: float, current_value: float
) -> None:
    # 0.91 - 0.92 == -0.010000000000000009 in binary floating point, so a naive
    # `delta < -0.01` flags a nominally on-tolerance move as a regression.
    baseline = _write_baseline(tmp_path, _results(field_accuracy={"yield": baseline_value}))
    run_dir = _write_run(tmp_path, _results(field_accuracy={"yield": current_value}))
    result = diff_against_baseline(baseline, run_dir)
    assert _change(result, "field_accuracy.yield")["flag"] == "unchanged"
    assert result.status == "ok"


def test_a_drop_just_beyond_the_tolerance_is_still_a_regression(tmp_path: Path) -> None:
    baseline = _write_baseline(tmp_path, _results(field_accuracy={"yield": 0.92}))
    run_dir = _write_run(tmp_path, _results(field_accuracy={"yield": 0.9}))
    result = diff_against_baseline(baseline, run_dir)
    assert _change(result, "field_accuracy.yield")["flag"] == FLAG_REGRESSION


def test_counts_are_diffed_but_informational(tmp_path: Path) -> None:
    baseline = _write_baseline(tmp_path, _results(counts={"ready": 115, "needs_review": 3}))
    run_dir = _write_run(tmp_path, _results(counts={"ready": 100, "needs_review": 18}))
    result = diff_against_baseline(baseline, run_dir)
    ready = _change(result, "count.ready")
    assert ready["flag"] == "info"
    assert ready["direction"] == "informational"
    assert ready["delta"] == -15
    assert result.status == "ok"
    assert "count.ready: 115 -> 100" in result.summary


def test_metric_missing_from_baseline_is_new_not_regression(tmp_path: Path) -> None:
    baseline = _write_baseline(tmp_path, _results())  # no judge section
    run_dir = _write_run(tmp_path, _results(with_judge=True, judge_pass_rate=0.86))
    result = diff_against_baseline(baseline, run_dir)
    change = _change(result, "judge.pass_rate")
    assert change["flag"] == "new"
    assert change["baseline"] is None
    assert result.status == "ok"
    assert "judge.pass_rate: n/a -> 0.86 (new)" in result.summary


def test_accuracy_the_baseline_measured_and_this_run_cannot_is_a_regression(
    tmp_path: Path,
) -> None:
    # A run whose every extraction was rejected/truncated stays `completed` with
    # all-None accuracy. Treating that as merely "missing" printed
    # "No regressions detected" for a total extraction collapse.
    baseline = _write_baseline(
        tmp_path, _results(field_accuracy={"title_normalized": 0.97, "yield": 0.92})
    )
    run_dir = _write_run(
        tmp_path,
        _results(
            field_accuracy={"title_normalized": None, "yield": None},
            counts={"fixtures": 2, "recipes_extracted": 0, "extraction_failures": 2},
        ),
    )
    result = diff_against_baseline(baseline, run_dir)
    assert _change(result, "field_accuracy.title_normalized")["flag"] == FLAG_REGRESSION
    assert result.status == "regressions_detected"
    assert "no longer measured" in result.summary
    assert "2 regression(s) detected." in result.summary


def test_metric_missing_from_current_is_missing_not_regression(tmp_path: Path) -> None:
    # e.g. the current run had no judge-alignment pass, so agreement is absent.
    baseline = _write_baseline(tmp_path, _results(with_agreement=True, agreement_rate=0.91))
    run_dir = _write_run(tmp_path, _results())
    result = diff_against_baseline(baseline, run_dir)
    change = _change(result, "agreement.rate")
    assert change["flag"] == "missing"
    assert change["current"] is None
    assert result.status == "ok"
    assert "agreement.rate: 0.91 -> n/a (missing)" in result.summary


def test_changes_shape_and_status_are_the_epic14_diffresult(tmp_path: Path) -> None:
    baseline = _write_baseline(tmp_path, _results())
    run_dir = _write_run(tmp_path, _results())
    result = diff_against_baseline(baseline, run_dir)
    assert result.status != "not_implemented"
    assert set(result.model_dump()) == {
        "baseline_path",
        "current_path",
        "status",
        "summary",
        "changes",
    }
    for change in result.changes:
        assert set(change) == {"metric", "baseline", "current", "delta", "direction", "flag"}


def test_failed_current_run_is_refused_not_reported_clean(tmp_path: Path) -> None:
    # A crashed run unwraps to empty metrics, which read as "missing" — the
    # diff must refuse rather than return "No regressions detected".
    baseline = _write_baseline(tmp_path, _results())
    run_dir = tmp_path / "report"
    run_dir.mkdir()
    doc = {
        "metadata": {"run_label": "current"},
        "status": "failed",
        "error": "LLMTechnicalError",
        "results": {},
    }
    (run_dir / "results.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failed report"):
        diff_against_baseline(baseline, run_dir)


def test_failed_baseline_source_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "extraction.json"
    path.write_text(
        json.dumps(
            {
                "baseline_set_at": "2026-01-01T00:00:00+00:00",
                "run_label": "baseline",
                "source": {"metadata": {}, "status": "failed", "error": "boom", "results": {}},
            }
        ),
        encoding="utf-8",
    )
    run_dir = _write_run(tmp_path, _results())
    with pytest.raises(ValueError, match="failed baseline"):
        diff_against_baseline(path, run_dir)


def test_run_dir_without_results_json_raises(tmp_path: Path) -> None:
    baseline = _write_baseline(tmp_path, _results())
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="results"):
        diff_against_baseline(baseline, empty_dir)
