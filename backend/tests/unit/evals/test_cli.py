"""Tests for the ``rag-evals`` CLI shape (Epic 14 Phase 14.1).

Extraction/judge subcommands are scaffold stubs that print ``not implemented
yet`` and exit 0 until Epic 15; ``retrieval`` gained real behaviour in Epic 16
(covered in ``test_cli_retrieval.py``). These tests pin the CLI shape: the
five hyphenated subcommand names, flag options (``--fixtures``, ``--queries``,
``--k``, ``--judge``) vs the two positional ``diff`` arguments, and the flag
defaults.
"""

from __future__ import annotations

from pathlib import Path

import click
from evals.cli import app
from typer.main import get_command
from typer.testing import CliRunner

runner = CliRunner()

SUBCOMMANDS = ("extraction", "retrieval", "judge-alignment", "confidence-review", "diff")


def test_help_lists_all_five_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in SUBCOMMANDS:
        assert name in result.output


def test_extraction_stub_exits_zero_with_required_fixtures_flag() -> None:
    result = runner.invoke(app, ["extraction", "--fixtures", "synthetic"])
    assert result.exit_code == 0
    assert "not implemented yet" in result.output


def test_extraction_accepts_optional_judge_flag() -> None:
    result = runner.invoke(
        app, ["extraction", "--fixtures", "synthetic", "--judge", "extraction-judge"]
    )
    assert result.exit_code == 0
    assert "not implemented yet" in result.output


def test_extraction_without_required_fixtures_flag_fails() -> None:
    result = runner.invoke(app, ["extraction"])
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


def test_judge_alignment_stub_exits_zero_with_required_judge_flag() -> None:
    result = runner.invoke(app, ["judge-alignment", "--judge", "extraction-judge"])
    assert result.exit_code == 0
    assert "not implemented yet" in result.output


def test_confidence_review_stub_takes_no_args_and_exits_zero() -> None:
    result = runner.invoke(app, ["confidence-review"])
    assert result.exit_code == 0
    assert "not implemented yet" in result.output


def test_diff_accepts_two_positional_paths_and_prints_placeholder(tmp_path: Path) -> None:
    # Wired to the diff_against_baseline skeleton in Phase 14.2: both paths must
    # exist; the command prints the placeholder summary and exits 0.
    baseline = tmp_path / "b.json"
    baseline.write_text("{}")
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    result = runner.invoke(app, ["diff", str(baseline), str(report_dir)])
    assert result.exit_code == 0
    assert "not implemented" in result.output


def test_diff_missing_path_exits_with_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["diff", str(tmp_path / "missing.json"), str(tmp_path)])
    assert result.exit_code == 2
    assert "not implemented" not in result.output
