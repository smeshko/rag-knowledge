"""Tests for the ``rag-evals`` CLI shape (Epic 14 Phase 14.1, Epics 15 & 16).

Every subcommand is real: extraction/judge behaviour landed in Epic 15
(covered offline in ``test_extraction_eval.py`` / ``test_alignment.py`` /
``test_calibration.py``), retrieval in Epic 16 (covered in
``test_cli_retrieval.py``), and diff/save-baseline span both (covered in
``test_reports_diff.py`` / ``test_cli_diff.py``). These tests pin the CLI
shape: the hyphenated subcommand names, flag options (``--fixtures``,
``--label``, ``--queries``, ``--k``, ``--judge``) vs the two positional
``diff`` arguments, and the flag defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import evals.cli
import evals.fixtures
import evals.reports
import pytest
from evals.cli import app
from typer.main import get_command
from typer.testing import CliRunner

from tests.unit.evals.eval_utils import STEW_EXPECTED, STEW_SOURCE, write_fixture

runner = CliRunner()

SUBCOMMANDS = (
    "extraction",
    "retrieval",
    "judge-alignment",
    "confidence-review",
    "diff",
    "save-baseline",
)


def test_help_lists_all_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in SUBCOMMANDS:
        assert name in result.output


def test_extraction_without_required_fixtures_flag_fails() -> None:
    result = runner.invoke(app, ["extraction", "--label", "smoke"])
    assert result.exit_code == 2


def test_extraction_without_required_label_flag_fails() -> None:
    # --label became a required option in Epic 15 Phase 15.1.
    result = runner.invoke(app, ["extraction", "--fixtures", "synthetic"])
    assert result.exit_code == 2


def test_retrieval_flag_defaults() -> None:
    # `retrieval` is real as of Epic 16 (behaviour covered in
    # test_cli_retrieval.py with run_retrieval_eval stubbed); this pins the
    # flag shape: --k/--mode/--label are *options* with these defaults.
    command = get_command(app)
    assert isinstance(command, click.Group)
    retrieval = command.commands["retrieval"]
    defaults = {param.name: param.default for param in retrieval.params}
    assert defaults["k"] == 10
    assert defaults["mode"] == "hybrid"
    assert defaults["label"] == "retrieval"


def test_judge_alignment_requires_judge_and_fixtures_flags() -> None:
    # Both became required options in Epic 15 Phase 15.3 (behaviour is covered
    # offline in test_alignment.py).
    result = runner.invoke(app, ["judge-alignment", "--judge", "extraction-judge"])
    assert result.exit_code == 2
    result = runner.invoke(app, ["judge-alignment", "--fixtures", "smoke"])
    assert result.exit_code == 2


# --- judge-alignment bad-run shapes (Epic 20 Phase 20.1) ---------------------


def _forbid(fired: list[str], name: str):  # noqa: ANN202
    def _raise(*args: object, **kwargs: object) -> object:
        fired.append(name)
        raise AssertionError(f"{name} must not be called on a bad-run path")

    return _raise


def _write_run(run_dir: Path, doc: dict[str, object]) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(json.dumps(doc), encoding="utf-8")
    return run_dir


_BAD_RUN_SHAPES = (
    "no_run",
    "missing_dir",
    "missing_results",
    "malformed_results",
    "malformed_per_fixture",
    "failed_run",
    "retrieval_run",
    "set_mismatch",
    "drifted_fixture",
)


@pytest.mark.parametrize("shape", _BAD_RUN_SHAPES)
def test_judge_alignment_bad_run_exits_2_without_provider_or_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    """Every unusable run exits 2 before ``get_settings``/provider construction.

    All three seams are monkeypatched to raise: ``get_settings()`` is where a
    missing API key would blow up, so proving only the provider half would leave
    the original failure mode uncovered. ``_build_judge_provider`` is listed too
    (Epic 23.3) — it is evaluated at the same call site, and relying on argument
    evaluation order to reach ``_build_llm_provider`` first would let a future
    reordering regress the DECISIONS #9 invariant silently.
    """
    fixtures_root = tmp_path / "fixtures"
    write_fixture(fixtures_root, "smoke", "bean-stew", STEW_SOURCE, STEW_EXPECTED)
    monkeypatch.setattr(evals.fixtures, "FIXTURES_ROOT", fixtures_root)
    monkeypatch.setattr(evals.reports, "REPORTS_ROOT", tmp_path / "reports")
    fired: list[str] = []
    monkeypatch.setattr(
        evals.cli, "_build_llm_provider", _forbid(fired, "_build_llm_provider")
    )
    monkeypatch.setattr(
        evals.cli, "_build_judge_provider", _forbid(fired, "_build_judge_provider")
    )
    monkeypatch.setattr("rag_recipes.config.get_settings", _forbid(fired, "get_settings"))

    args = ["judge-alignment", "--judge", "summary_quality", "--fixtures", "smoke"]
    if shape == "no_run":
        pass  # REPORTS_ROOT does not exist → latest_run_dir() → None
    elif shape == "missing_dir":
        args += ["--report", str(tmp_path / "nope")]
    elif shape == "missing_results":
        (tmp_path / "empty-run").mkdir()
        args += ["--report", str(tmp_path / "empty-run")]
    elif shape == "malformed_results":
        run = tmp_path / "bad-run"
        run.mkdir()
        (run / "results.json").write_text("{not json", encoding="utf-8")
        args += ["--report", str(run)]
    elif shape == "malformed_per_fixture":
        # Valid JSON, wrong shape: iterating a null per_fixture raises
        # TypeError, which this command does not catch — exit 1 + traceback.
        run = _write_run(
            tmp_path / "shape-run",
            {
                "metadata": {},
                "status": "completed",
                "results": {"fixture_set": "smoke", "per_fixture": None},
            },
        )
        args += ["--report", str(run)]
    elif shape == "failed_run":
        run = _write_run(
            tmp_path / "failed-run",
            {"metadata": {}, "status": "failed", "error": "ValueError", "results": {}},
        )
        args += ["--report", str(run)]
    elif shape == "retrieval_run":
        run = _write_run(
            tmp_path / "retrieval-run",
            {
                "metadata": {},
                "status": "completed",
                "results": {"report_type": "retrieval", "aggregate": {}},
            },
        )
        args += ["--report", str(run)]
    elif shape == "set_mismatch":
        run = _write_run(
            tmp_path / "other-run",
            {
                "metadata": {},
                "status": "completed",
                "results": {"fixture_set": "other", "per_fixture": []},
            },
        )
        args += ["--report", str(run)]
    elif shape == "drifted_fixture":
        run = _write_run(
            tmp_path / "drift-run",
            {
                "metadata": {},
                "status": "completed",
                "results": {
                    "fixture_set": "smoke",
                    "extraction_prompt_version": "recipe-extraction-v1",
                    "per_fixture": [
                        {
                            "name": "bean-stew",
                            "status": "scored",
                            "fixture_content_hash": "0" * 64,
                            "recipes": [{"title": "stale"}],
                        }
                    ],
                },
            },
        )
        args += ["--report", str(run)]

    result = runner.invoke(app, args)

    assert result.exit_code == 2
    assert "error" in result.output
    assert fired == []  # neither get_settings nor the provider seam ever ran


def test_confidence_review_with_missing_report_dir_exits_with_error(tmp_path: Path) -> None:
    # Real behaviour since Epic 15 Phase 15.3 (happy path is covered offline in
    # test_calibration.py); a nonexistent run dir is a caller error.
    result = runner.invoke(app, ["confidence-review", "--report", str(tmp_path / "missing")])
    assert result.exit_code == 2


def test_diff_accepts_two_positional_paths_and_prints_extraction_diff(tmp_path: Path) -> None:
    # Real diff since Epic 15 Phase 15.3 (deltas/flags are exercised in
    # test_reports_diff.py); both inputs are built in their wrapped shapes.
    import json

    results = {"aggregate": {"field_accuracy": {"yield": 0.9}, "ready": 2}}
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "results.json").write_text(
        json.dumps({"metadata": {}, "status": "completed", "results": results})
    )
    baseline = tmp_path / "b.json"
    baseline.write_text(
        json.dumps(
            {
                "baseline_set_at": "2026-01-01T00:00:00+00:00",
                "run_label": "base",
                "source": {"metadata": {}, "status": "completed", "results": results},
            }
        )
    )
    result = runner.invoke(app, ["diff", str(baseline), str(report_dir)])
    assert result.exit_code == 0
    assert "Extraction diff vs baseline:" in result.output
    assert "No regressions detected." in result.output


def test_diff_missing_path_exits_with_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["diff", str(tmp_path / "missing.json"), str(tmp_path)])
    assert result.exit_code == 2
    assert "error" in result.output
