"""Tests for the real ``rag-evals retrieval`` subcommand (Epic 16 Phase 16.1).

``run_retrieval_eval`` is monkeypatched to a stub throughout: the CLI path
constructs the **default** in-process search caller, which reaches real
providers, so no test here may let the command run for real. A live invocation
against an ingested corpus is DEFERRED — requires live provider run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from evals.cli import app
from typer.testing import CliRunner

runner = CliRunner()


class _StubReport:
    def __init__(self, path: Path) -> None:
        self.path = path


def _install_stub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    async def stub(query_set: str, k: int, label: str, mode: str = "hybrid") -> _StubReport:
        calls.update(query_set=query_set, k=k, label=label, mode=mode)
        return _StubReport(tmp_path / "2026-01-01T00-00-00-stub")

    monkeypatch.setattr("evals.cli.run_retrieval_eval", stub)
    return calls


def test_parses_all_options_and_prints_report_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_stub(monkeypatch, tmp_path)
    result = runner.invoke(
        app,
        ["retrieval", "--queries", "tests", "--k", "10", "--mode", "hybrid", "--label", "initial"],
    )
    assert result.exit_code == 0, result.output
    assert calls == {"query_set": "tests", "k": 10, "label": "initial", "mode": "hybrid"}
    assert "2026-01-01T00-00-00-stub" in result.output


def test_defaults_k_10_mode_hybrid_label_retrieval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["retrieval", "--queries", "tests"])
    assert result.exit_code == 0, result.output
    assert calls == {"query_set": "tests", "k": 10, "label": "retrieval", "mode": "hybrid"}


@pytest.mark.parametrize("mode", ["keyword", "vector"])
def test_non_default_modes_parse_as_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    calls = _install_stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["retrieval", "--queries", "tests", "--mode", mode])
    assert result.exit_code == 0, result.output
    assert calls["mode"] == mode


def test_invalid_mode_exits_nonzero_before_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("run_retrieval_eval must not run for an invalid mode")

    monkeypatch.setattr("evals.cli.run_retrieval_eval", boom)
    result = runner.invoke(app, ["retrieval", "--queries", "tests", "--mode", "bogus"])
    assert result.exit_code == 2


@pytest.mark.parametrize("k", ["0", "-1"])
def test_non_positive_k_exits_two_before_running(
    monkeypatch: pytest.MonkeyPatch, k: str
) -> None:
    # k reaches a bare `results[:k]` slice and the diff's top-k cutoff, so a
    # negative k would silently trim each result list's tail.
    async def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("run_retrieval_eval must not run for a non-positive k")

    monkeypatch.setattr("evals.cli.run_retrieval_eval", boom)
    result = runner.invoke(app, ["retrieval", "--queries", "tests", "--k", k])
    assert result.exit_code == 2


def test_missing_required_queries_flag_fails() -> None:
    result = runner.invoke(app, ["retrieval"])
    assert result.exit_code == 2


def test_help_still_lists_all_five_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("extraction", "retrieval", "judge-alignment", "confidence-review", "diff"):
        assert name in result.output
