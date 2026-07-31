"""Tests for the confidence-calibration review (Epic 15 Phase 15.3).

No provider, no API key, no ``live`` marker — run dirs are crafted under
``tmp_path`` in the real ``{"metadata", "results"}`` envelope shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from evals.calibration import _bucket_of, run_confidence_review


@pytest.mark.parametrize(
    ("confidence", "bucket"),
    [
        (1.0, "0.90-1.00"),
        (0.95, "0.90-1.00"),
        (0.90, "0.90-1.00"),
        (0.89, "0.75-0.89"),
        (0.75, "0.75-0.89"),
        (0.74, "0.50-0.74"),
        (0.50, "0.50-0.74"),
        (0.49, "<0.50"),
        (0.0, "<0.50"),
    ],
)
def test_bucket_boundaries(confidence: float, bucket: str) -> None:
    assert _bucket_of(confidence) == bucket


def _scores(
    *,
    title_normalized: bool = True,
    yield_match: bool = True,
    ingredient_count: bool = True,
    step_count: bool = True,
    f1: float = 1.0,
) -> dict[str, Any]:
    return {
        "title": {"exact": title_normalized, "normalized": title_normalized},
        "yield": yield_match,
        "prep_time": {"match": True, "actual_minutes": 15.0, "expected_minutes": 15.0},
        "cook_time": {"match": False, "actual_minutes": None, "expected_minutes": None},
        "total_time": {"match": True, "actual_minutes": 45.0, "expected_minutes": 45.0},
        "ingredient_count": ingredient_count,
        "step_count": step_count,
        "ingredients_detail": {"f1": f1, "precision": f1, "recall": f1},
        "source_span_ids": {"f1": f1, "precision": f1, "recall": f1},
    }


def _item(name: str, confidence: float, scores: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": "scored",
        "review_status": "ready",
        "warnings": [],
        "confidence_overall": confidence,
        "missing_fields": [],
        "scores": _scores() if scores is None else scores,
    }


def _write_run(
    reports_root: Path,
    dir_name: str,
    per_fixture: list[dict[str, Any]],
    judge: dict[str, Any] | None = None,
) -> Path:
    run_dir = reports_root / dir_name
    run_dir.mkdir(parents=True)
    doc = {
        "metadata": {"run_label": dir_name, "timestamp": "2026-07-31T00:00:00+00:00"},
        "status": "completed",
        "results": {
            "fixture_set": "smoke",
            "per_fixture": per_fixture,
            "aggregate": {"fixtures": len(per_fixture)},
            "judge": judge,
            "agreement": None,
            "calibration": None,
        },
    }
    (run_dir / "results.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(f"# Extraction eval — {dir_name}\n", encoding="utf-8")
    return run_dir


def _judge_section(ratings: dict[str, str]) -> dict[str, Any]:
    return {
        "name": "summary_quality",
        "version": "v1",
        "model": "fake-model",
        "per_fixture": {
            name: {"status": "rated", "rating": rating, "critique": "…"}
            for name, rating in ratings.items()
        },
        "pass_rate": None,
        "rated": len(ratings),
        "unrated": 0,
    }


def test_overlay_counts_accuracy_and_pass_rate_per_bucket(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        "2026-07-31T10-00-00-eval",
        [
            _item("a", 0.95),
            _item("b", 0.92, _scores(step_count=False)),  # 7/8 = 0.875
            _item("c", 0.80),
            _item("d", 0.40, _scores(title_normalized=False, yield_match=False, f1=0.0)),
        ],
        judge=_judge_section({"a": "pass", "b": "fail", "c": "pass"}),
    )
    report = run_confidence_review(run_dir)
    by_bucket = {bucket.bucket: bucket for bucket in report.buckets}
    top = by_bucket["0.90-1.00"]
    assert top.count == 2
    assert top.fixtures == ["a", "b"]
    assert top.mean_objective_accuracy == pytest.approx((1.0 + 7 / 8) / 2)
    assert top.judge_pass_rate == pytest.approx(0.5)
    assert by_bucket["0.75-0.89"].count == 1
    assert by_bucket["0.75-0.89"].judge_pass_rate == pytest.approx(1.0)
    assert by_bucket["0.50-0.74"].count == 0
    assert by_bucket["0.50-0.74"].mean_objective_accuracy is None
    assert by_bucket["<0.50"].count == 1
    assert by_bucket["<0.50"].judge_pass_rate is None  # judge never rated "d"


def test_miscalibration_line_for_high_confidence_judge_failure(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        "2026-07-31T10-00-00-eval",
        [_item("a", 0.95), _item("b", 0.92)],
        judge=_judge_section({"a": "pass", "b": "fail"}),
    )
    report = run_confidence_review(run_dir)
    assert any(
        "confidence ≥0.90" in line and "failed the judge" in line and "b" in line
        for line in report.miscalibration
    )


def test_miscalibration_line_for_high_confidence_low_objective(tmp_path: Path) -> None:
    bad = _scores(title_normalized=False, yield_match=False, step_count=False, f1=0.2)
    run_dir = _write_run(
        tmp_path, "2026-07-31T10-00-00-eval", [_item("a", 0.95, bad)]
    )
    report = run_confidence_review(run_dir)
    assert any("scored below 0.80" in line and "a" in line for line in report.miscalibration)


def test_no_judge_section_degrades_to_objective_only(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "2026-07-31T10-00-00-eval", [_item("a", 0.95)], judge=None)
    report = run_confidence_review(run_dir)
    top = next(bucket for bucket in report.buckets if bucket.bucket == "0.90-1.00")
    assert top.count == 1
    assert top.judge_pass_rate is None
    assert top.mean_objective_accuracy == pytest.approx(1.0)


def test_calibration_merges_into_run_preserving_other_sections(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        "2026-07-31T10-00-00-eval",
        [_item("a", 0.95)],
        judge=_judge_section({"a": "pass"}),
    )
    before = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    run_confidence_review(run_dir)
    after = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert after["metadata"] == before["metadata"]
    assert after["status"] == before["status"]
    assert after["results"]["per_fixture"] == before["results"]["per_fixture"]
    assert after["results"]["aggregate"] == before["results"]["aggregate"]
    assert after["results"]["judge"] == before["results"]["judge"]
    assert after["results"]["agreement"] == before["results"]["agreement"]
    calibration = after["results"]["calibration"]
    assert [bucket["bucket"] for bucket in calibration["buckets"]] == [
        "0.90-1.00",
        "0.75-0.89",
        "0.50-0.74",
        "<0.50",
    ]
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "Confidence calibration:" in summary
    assert "0.50-0.74: n/a (empty)" in summary


def test_defaults_to_latest_run_dir(tmp_path: Path) -> None:
    _write_run(tmp_path, "2026-07-30T10-00-00-old", [_item("old", 0.95)])
    newest = _write_run(tmp_path, "2026-07-31T10-00-00-new", [_item("new", 0.95)])
    report = run_confidence_review(None, reports_root=tmp_path)
    assert report.report_path == str(newest)


def test_missing_run_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_confidence_review(None, reports_root=tmp_path / "empty")


def test_cli_confidence_review_prints_bucket_view(tmp_path: Path) -> None:
    from evals.cli import app
    from typer.testing import CliRunner

    run_dir = _write_run(
        tmp_path,
        "2026-07-31T10-00-00-eval",
        [_item("a", 0.95)],
        judge=_judge_section({"a": "pass"}),
    )
    result = CliRunner().invoke(app, ["confidence-review", "--report", str(run_dir)])
    assert result.exit_code == 0
    assert "Confidence calibration:" in result.output
    assert "0.90-1.00: 1 item(s)" in result.output
