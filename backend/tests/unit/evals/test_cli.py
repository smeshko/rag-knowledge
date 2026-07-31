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

from pathlib import Path

import click
from evals.cli import app
from typer.main import get_command
from typer.testing import CliRunner

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
