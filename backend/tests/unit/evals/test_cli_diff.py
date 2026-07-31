"""Tests for ``rag-evals diff`` retrieval dispatch and ``save-baseline`` (Epic 16 Phase 16.2).

The fixtures write a **real-shaped baseline file** (``{"baseline_set_at",
"run_label", "source": {...}}``) and a **real report directory** with a
``results.json`` (``{"metadata", ..., "results"}``) — the two inputs are
deliberately differently-shaped envelopes. No DB, no HTTP, no providers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evals.reports
import pytest
from evals.cli import app
from typer.testing import CliRunner

runner = CliRunner()

_METRICS = ("ndcg_cut_10", "recall_5", "recall_10", "recip_rank")


def _payload(ndcg: float, ranks: dict[str, int | None]) -> dict[str, Any]:
    return {
        "report_type": "retrieval",
        "run": {
            "query_set": "golden",
            "mode": "hybrid",
            "k": 10,
            "limit": 10,
            "embedding_provider": "fake",
            "embedding_model": "fake-embedding",
            "reranking_enabled": False,
        },
        "aggregate": {measure: ndcg for measure in _METRICS},
        "per_query": {
            "q1": {
                "metrics": {measure: ndcg for measure in _METRICS},
                "expected_item_ranks": ranks,
                "retrieved_ids": [],
            }
        },
    }


def _write_baseline(tmp_path: Path, payload: dict[str, Any]) -> Path:
    baseline = tmp_path / "retrieval.json"
    baseline.write_text(
        json.dumps(
            {
                "baseline_set_at": "2026-01-01T00:00:00+00:00",
                "run_label": "before",
                "source": {
                    "metadata": {"run_label": "before"},
                    "status": "completed",
                    "results": payload,
                },
            }
        ),
        encoding="utf-8",
    )
    return baseline


def _write_report(tmp_path: Path, payload: dict[str, Any], name: str = "report") -> Path:
    report_dir = tmp_path / name
    report_dir.mkdir()
    (report_dir / "results.json").write_text(
        json.dumps(
            {"metadata": {"run_label": "after"}, "status": "completed", "results": payload}
        ),
        encoding="utf-8",
    )
    return report_dir


def test_regression_pair_prints_headline_and_exits_nonzero(tmp_path: Path) -> None:
    baseline = _write_baseline(tmp_path, _payload(0.61, {"a": 1}))
    report = _write_report(tmp_path, _payload(0.57, {"a": 5}))
    result = runner.invoke(app, ["diff", str(baseline), str(report)])
    assert result.exit_code == 1
    assert "NDCG@10: 0.61 → 0.57 [REGRESSION -0.04]" in result.output
    assert "Per-query regressions:" in result.output
    assert "q1: a rank 1 → 5 (rank_drop)" in result.output
    assert "Biggest NDCG@10 drops:" in result.output


def test_clean_pair_exits_zero(tmp_path: Path) -> None:
    payload = _payload(0.9, {"a": 1})
    baseline = _write_baseline(tmp_path, payload)
    report = _write_report(tmp_path, payload)
    result = runner.invoke(app, ["diff", str(baseline), str(report)])
    assert result.exit_code == 0
    assert "[no change]" in result.output


def test_extraction_pair_keeps_placeholder_path(tmp_path: Path) -> None:
    extraction = {"report_type": "extraction", "results_by_fixture": {}}
    baseline = _write_baseline(tmp_path, extraction)
    report = _write_report(tmp_path, extraction)
    result = runner.invoke(app, ["diff", str(baseline), str(report)])
    assert result.exit_code == 0
    assert "not implemented" in result.output
    assert "NDCG" not in result.output


def test_missing_baseline_exits_two(tmp_path: Path) -> None:
    report = _write_report(tmp_path, _payload(0.9, {"a": 1}))
    result = runner.invoke(app, ["diff", str(tmp_path / "missing.json"), str(report)])
    assert result.exit_code == 2


def test_save_baseline_writes_named_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # save_as_baseline defaults to the repo's evals/baselines/ — point the
    # module root at tmp_path so the test never writes into the repo.
    monkeypatch.setattr(evals.reports, "BASELINES_ROOT", tmp_path / "baselines")
    report = _write_report(tmp_path, _payload(0.9, {"a": 1}))
    result = runner.invoke(app, ["save-baseline", str(report), "--name", "retrieval"])
    assert result.exit_code == 0, result.output
    baseline_path = tmp_path / "baselines" / "retrieval.json"
    assert baseline_path.is_file()
    assert str(baseline_path) in result.output
    doc = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert set(doc) == {"baseline_set_at", "run_label", "source"}
    assert doc["run_label"] == "after"
    assert doc["source"]["results"]["report_type"] == "retrieval"


def test_diff_of_incomparable_runs_exits_two_not_one(tmp_path: Path) -> None:
    baseline = _write_baseline(tmp_path, _payload(0.61, {"a": 1}))
    current = _payload(0.57, {"a": 5})
    current["run"]["mode"] = "vector"
    report = _write_report(tmp_path, current)
    result = runner.invoke(app, ["diff", str(baseline), str(report)])
    assert result.exit_code == 2
    assert "not comparable" in result.output
    assert "NDCG@10" in result.output  # the deltas are still printed


def test_diff_of_a_failed_run_exits_two_not_zero(tmp_path: Path) -> None:
    payload = _payload(0.9, {"a": 1})
    baseline = _write_baseline(tmp_path, payload)
    report = _write_report(tmp_path, payload)
    doc = json.loads((report / "results.json").read_text(encoding="utf-8"))
    doc["status"] = "failed"
    (report / "results.json").write_text(json.dumps(doc), encoding="utf-8")
    result = runner.invoke(app, ["diff", str(baseline), str(report)])
    assert result.exit_code == 2


def test_diff_of_mismatched_report_types_exits_two(tmp_path: Path) -> None:
    baseline = _write_baseline(tmp_path, _payload(0.9, {"a": 1}))
    report = _write_report(tmp_path, {"report_type": "extraction", "results_by_fixture": {}})
    result = runner.invoke(app, ["diff", str(baseline), str(report)])
    assert result.exit_code == 2


def test_diff_pointed_at_results_json_exits_two(tmp_path: Path) -> None:
    payload = _payload(0.9, {"a": 1})
    baseline = _write_baseline(tmp_path, payload)
    report = _write_report(tmp_path, payload)
    result = runner.invoke(app, ["diff", str(baseline), str(report / "results.json")])
    assert result.exit_code == 2


def test_save_baseline_pointed_at_results_json_exits_two(tmp_path: Path) -> None:
    report = _write_report(tmp_path, _payload(0.9, {"a": 1}))
    result = runner.invoke(
        app, ["save-baseline", str(report / "results.json"), "--name", "retrieval"]
    )
    assert result.exit_code == 2


def test_save_baseline_rejects_unsafe_name(tmp_path: Path) -> None:
    report = _write_report(tmp_path, _payload(0.9, {"a": 1}))
    result = runner.invoke(app, ["save-baseline", str(report), "--name", "../escape"])
    assert result.exit_code == 2


def test_save_baseline_missing_report_exits_two(tmp_path: Path) -> None:
    result = runner.invoke(app, ["save-baseline", str(tmp_path / "nope"), "--name", "retrieval"])
    assert result.exit_code == 2


def test_help_lists_save_baseline() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "save-baseline" in result.output
