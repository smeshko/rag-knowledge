"""Confidence-calibration review (Epic 15 Phase 15.3, doc 12 § 6).

Buckets an extraction-eval run's items by ``confidence.overall`` into the four
doc-12 § 6 buckets and overlays *actual* quality — mean objective accuracy
(15.1 scores) and judge pass rate (15.2 ratings) — so miscalibration ("high
confidence, low quality") is visible at a glance.

Bucket boundaries (DECISIONS #4, half-open at the lower edges, total over
``[0, 1]``): ``0.90–1.00`` = ``[0.90, 1.00]``, ``0.75–0.89`` = ``[0.75,
0.90)``, ``0.50–0.74`` = ``[0.50, 0.75)``, ``<0.50`` = ``[0.00, 0.50)`` —
boundary values land in the higher bucket.

This module makes **no** provider calls: it reads the ratings 15.1/15.2
already recorded. The calibration section is merged back into the *same* run's
``results.json``/``summary.md`` via the read-modify-write helper (DECISIONS
#7) — never a new ``ReportRun``. Calibration is a review view, not a
regression-diffed metric (the baseline diff deliberately scopes to the scalar
metrics — see ``diff_against_baseline``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evals.reports import _append_run_summary, _update_run_results, latest_run_dir

__all__ = ["BucketOverlay", "CalibrationReport", "run_confidence_review"]

BUCKET_LABELS = ("0.90-1.00", "0.75-0.89", "0.50-0.74", "<0.50")

_TIME_FIELDS = ("prep_time", "cook_time", "total_time")
_HIGH_CONFIDENCE_BUCKETS = ("0.90-1.00", "0.75-0.89")
_LOW_OBJECTIVE_THRESHOLD = 0.8


class BucketOverlay(BaseModel):
    """One confidence bucket with its actual-quality overlay."""

    bucket: str
    count: int
    fixtures: list[str]
    mean_objective_accuracy: float | None
    judge_pass_rate: float | None


class CalibrationReport(BaseModel):
    """The four-bucket calibration view for one extraction-eval run."""

    report_path: str
    buckets: list[BucketOverlay]
    miscalibration: list[str]

    def calibration_payload(self) -> dict[str, Any]:
        """The ``results.calibration`` section written into the run's report."""
        return {
            "buckets": [bucket.model_dump() for bucket in self.buckets],
            "miscalibration": list(self.miscalibration),
        }


def _bucket_of(confidence: float) -> str:
    """Map a ``confidence.overall`` to its doc-12 § 6 bucket (total over [0, 1])."""
    if confidence >= 0.90:
        return "0.90-1.00"
    if confidence >= 0.75:
        return "0.75-0.89"
    if confidence >= 0.50:
        return "0.50-0.74"
    return "<0.50"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _objective_accuracy(scores: dict[str, Any]) -> float | None:
    """One fixture's mean objective accuracy over its comparable score slots.

    Time fields only count when the golden side carried a parseable value
    (mirroring the 15.1 aggregate's eligibility rule).
    """
    values: list[float] = []
    title = scores.get("title") or {}
    if "normalized" in title:
        values.append(float(title["normalized"]))
    if isinstance(scores.get("yield"), bool):
        values.append(float(scores["yield"]))
    for field in _TIME_FIELDS:
        time_score = scores.get(field) or {}
        if time_score.get("expected_minutes") is not None:
            values.append(float(time_score.get("match", False)))
    for field in ("ingredient_count", "step_count"):
        if isinstance(scores.get(field), bool):
            values.append(float(scores[field]))
    for field in ("ingredients_detail", "source_span_ids"):
        breakdown = scores.get(field) or {}
        if "f1" in breakdown:
            values.append(float(breakdown["f1"]))
    return _mean(values)


def _overlay(
    items: list[dict[str, Any]], judge_ratings: dict[str, dict[str, Any]]
) -> tuple[list[BucketOverlay], list[str]]:
    """Group scored items into buckets and derive the miscalibration lines."""
    grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in BUCKET_LABELS}
    for item in items:
        grouped[_bucket_of(float(item["confidence_overall"]))].append(item)

    overlays: list[BucketOverlay] = []
    miscalibration: list[str] = []
    for label in BUCKET_LABELS:
        bucket_items = grouped[label]
        names = [item["name"] for item in bucket_items]
        accuracies = [
            accuracy
            for item in bucket_items
            if (accuracy := _objective_accuracy(item.get("scores") or {})) is not None
        ]
        rated = [
            judge_ratings[name]
            for name in names
            if judge_ratings.get(name, {}).get("status") == "rated"
        ]
        passes = [entry for entry in rated if entry.get("rating") == "pass"]
        overlays.append(
            BucketOverlay(
                bucket=label,
                count=len(bucket_items),
                fixtures=names,
                mean_objective_accuracy=_mean(accuracies),
                judge_pass_rate=len(passes) / len(rated) if rated else None,
            )
        )
        if label in _HIGH_CONFIDENCE_BUCKETS and bucket_items:
            bound = "≥0.90" if label == "0.90-1.00" else "0.75–0.89"
            failed = [
                name
                for name in names
                if judge_ratings.get(name, {}).get("status") == "rated"
                and judge_ratings[name].get("rating") == "fail"
            ]
            if failed:
                miscalibration.append(
                    f"{len(failed)} item(s) with confidence {bound} failed the judge: "
                    f"{', '.join(sorted(failed))}"
                )
            low = [
                item["name"]
                for item in bucket_items
                if (accuracy := _objective_accuracy(item.get("scores") or {})) is not None
                and accuracy < _LOW_OBJECTIVE_THRESHOLD
            ]
            if low:
                miscalibration.append(
                    f"{len(low)} item(s) with confidence {bound} scored below "
                    f"{_LOW_OBJECTIVE_THRESHOLD:.2f} objective accuracy: {', '.join(sorted(low))}"
                )
    return overlays, miscalibration


def _render_block(report: CalibrationReport) -> str:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    lines = ["Confidence calibration:"]
    for bucket in report.buckets:
        if bucket.count == 0:
            lines.append(f"  {bucket.bucket}: n/a (empty)")
            continue
        lines.append(
            f"  {bucket.bucket}: {bucket.count} item(s), "
            f"objective {fmt(bucket.mean_objective_accuracy)}, "
            f"judge pass rate {fmt(bucket.judge_pass_rate)}"
        )
    for line in report.miscalibration:
        lines.append(f"  MISCALIBRATION: {line}")
    return "\n".join(lines)


def run_confidence_review(
    report_path: Path | None = None, *, reports_root: Path | None = None
) -> CalibrationReport:
    """Build the calibration view for a run and merge it into that run's report.

    ``report_path`` is the extraction run directory; ``None`` resolves to the
    latest run under ``reports_root`` (default ``evals/reports/``). Degrades
    gracefully: no judge section → objective-only overlay; empty buckets render
    ``n/a``. Raises ``FileNotFoundError`` when no run exists to review.
    """
    run_dir = report_path if report_path is not None else latest_run_dir(reports_root)
    if run_dir is None or not (run_dir / "results.json").is_file():
        raise FileNotFoundError(
            f"no extraction-eval run to review: {run_dir or 'no run dirs found'}"
        )
    doc = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    results = doc.get("results") or {}
    items = [
        entry
        for entry in results.get("per_fixture") or []
        if entry.get("status") == "scored" and entry.get("confidence_overall") is not None
    ]
    judge_section = results.get("judge") or {}
    judge_ratings: dict[str, dict[str, Any]] = judge_section.get("per_fixture") or {}

    overlays, miscalibration = _overlay(items, judge_ratings)
    report = CalibrationReport(
        report_path=str(run_dir), buckets=overlays, miscalibration=miscalibration
    )

    payload = report.calibration_payload()

    def _merge(section: dict[str, Any]) -> None:
        section["calibration"] = payload

    _update_run_results(run_dir, _merge)
    _append_run_summary(run_dir, _render_block(report))
    return report
