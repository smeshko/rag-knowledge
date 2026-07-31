"""Tests for the report writer (Epic 14 Phase 14.2).

Hermetic by construction: every test passes ``reports_root=tmp_path`` (never
the repo's ``evals/reports/``) and injects a ``_SettingsStandIn`` satisfying
the ``SettingsLike`` protocol — no test reads an ambient ``.env`` or requires
``get_settings()``'s mandatory env vars (``database_url``, ``redis_url``,
``openai_api_key``).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import evals.reports
import pytest
from evals.reports import (
    DiffResult,
    ReportRun,
    _create_run_dir,
    _git_commit,
    build_metadata,
    diff_against_baseline,
    save_as_baseline,
)

RUN_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-")


class _SettingsStandIn:
    """Lightweight ``SettingsLike`` stand-in (exactly the five read attributes)."""

    def __init__(self, llm_provider: str = "openai") -> None:
        self.embedding_provider = "openai"
        self.embedding_model = "text-embedding-3-small"
        self.llm_provider = llm_provider
        self.llm_model = "gpt-4.1"
        self.anthropic_llm_model = "claude-sonnet-4-6"


def _run(tmp_path: Path, label: str = "extraction-gpt4-prompt-v3") -> ReportRun:
    return ReportRun(label, reports_root=tmp_path, settings=_SettingsStandIn())


# --- run directory naming ---------------------------------------------------


def test_run_dir_is_timestamped_and_filesystem_safe(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert run.path.parent == tmp_path
    assert run.path.is_dir()
    assert RUN_DIR_PATTERN.match(run.path.name)
    assert run.path.name.endswith("-extraction-gpt4-prompt-v3")
    assert ":" not in run.path.name


def test_unsafe_label_is_slugified(tmp_path: Path) -> None:
    run = ReportRun(
        "Extraction (GPT-4.1)/v2", reports_root=tmp_path, settings=_SettingsStandIn()
    )
    label_segment = RUN_DIR_PATTERN.sub("", run.path.name)
    assert run.path.is_dir()
    for forbidden in (" ", ":", "/", "(", ")"):
        assert forbidden not in label_segment
    assert label_segment  # the label did not slugify away to nothing


def test_colliding_run_dir_name_is_suffixed(tmp_path: Path) -> None:
    # The dir name resolves only to the second, so same-label runs inside one
    # second collide; each must still get its own directory.
    first = _create_run_dir(tmp_path, "2026-05-22T12-30-00-extraction")
    second = _create_run_dir(tmp_path, "2026-05-22T12-30-00-extraction")
    third = _create_run_dir(tmp_path, "2026-05-22T12-30-00-extraction")
    assert [p.name for p in (first, second, third)] == [
        "2026-05-22T12-30-00-extraction",
        "2026-05-22T12-30-00-extraction-2",
        "2026-05-22T12-30-00-extraction-3",
    ]


def test_same_second_runs_do_not_overwrite_each_other(tmp_path: Path) -> None:
    first = _run(tmp_path)
    second = _run(tmp_path)
    assert first.path != second.path
    first.write_results({"run": 1})
    second.write_results({"run": 2})
    assert json.loads((first.path / "results.json").read_text())["results"] == {"run": 1}
    assert json.loads((second.path / "results.json").read_text())["results"] == {"run": 2}


# --- file writes ------------------------------------------------------------


def test_write_methods_produce_the_three_files(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run.write_summary("# Summary\n\nAll good.\n")
    run.write_results({"accuracy": 0.9})
    run.write_per_item_breakdowns("# Per item\n\n- fx-001: ok\n")

    assert (run.path / "summary.md").read_text() == "# Summary\n\nAll good.\n"
    assert (run.path / "per_item_breakdowns.md").read_text() == "# Per item\n\n- fx-001: ok\n"
    doc = json.loads((run.path / "results.json").read_text())
    assert doc["results"] == {"accuracy": 0.9}
    assert doc["metadata"]["run_label"] == "extraction-gpt4-prompt-v3"


def test_results_json_embeds_full_metadata(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run.write_results({"accuracy": 0.9})
    metadata = json.loads((run.path / "results.json").read_text())["metadata"]
    assert metadata["timestamp"] == run.metadata.timestamp
    assert metadata["git_commit"] == run.metadata.git_commit
    assert metadata["embedding_provider"] == "openai"
    assert metadata["embedding_model"] == "text-embedding-3-small"
    assert metadata["llm_provider"] == "openai"
    assert metadata["llm_model"] == "gpt-4.1"
    assert metadata["prompt_version"] == "recipe-extraction-v1"
    assert metadata["schema_version"] == "recipe.v1"
    assert metadata["command_args"] == list(sys.argv)
    assert metadata["run_label"] == "extraction-gpt4-prompt-v3"


def test_context_manager_exit_guarantees_results_json(tmp_path: Path) -> None:
    with _run(tmp_path) as run:
        pass  # caller never writes results
    doc = json.loads((run.path / "results.json").read_text())
    assert doc["metadata"]["run_label"] == "extraction-gpt4-prompt-v3"
    assert doc["results"] == {}


def test_context_manager_does_not_overwrite_written_results(tmp_path: Path) -> None:
    with _run(tmp_path) as run:
        run.write_results({"accuracy": 0.9})
    doc = json.loads((run.path / "results.json").read_text())
    assert doc["results"] == {"accuracy": 0.9}
    assert doc["status"] == "completed"


def test_crashed_run_is_finalized_as_failed(tmp_path: Path) -> None:
    run = _run(tmp_path)
    with pytest.raises(RuntimeError), run:  # the exception is never swallowed
        run.write_results({"accuracy": 0.9})
        raise RuntimeError("provider blew up")
    doc = json.loads((run.path / "results.json").read_text())
    assert doc["status"] == "failed"
    assert doc["error"] == "RuntimeError"
    assert doc["results"] == {"accuracy": 0.9}  # partial results are kept, not erased


def test_failed_run_cannot_be_promoted_to_a_baseline(tmp_path: Path) -> None:
    run = _run(tmp_path / "reports")
    with pytest.raises(RuntimeError), run:
        raise RuntimeError("provider blew up")
    with pytest.raises(ValueError, match="failed run"):
        save_as_baseline(run.path, "extraction", baselines_root=tmp_path / "baselines")


# --- metadata capture -------------------------------------------------------


def test_provider_resolution_openai_branch() -> None:
    metadata = build_metadata("run", settings=_SettingsStandIn(llm_provider="openai"))
    assert metadata.llm_provider == "openai"
    assert metadata.llm_model == "gpt-4.1"


def test_provider_resolution_anthropic_branch() -> None:
    metadata = build_metadata("run", settings=_SettingsStandIn(llm_provider="anthropic"))
    assert metadata.llm_provider == "anthropic"
    assert metadata.llm_model == "claude-sonnet-4-6"


def test_results_json_contains_no_secret_field_names(tmp_path: Path) -> None:
    # Baselines are committed to git — a Settings dump would leak credentials.
    run = _run(tmp_path)
    run.write_results({"accuracy": 0.9})
    text = (run.path / "results.json").read_text()
    for secret_field in ("openai_api_key", "anthropic_api_key", "database_url", "redis_url"):
        assert secret_field not in text


# --- git helper -------------------------------------------------------------


def test_git_commit_returns_head_sha_inside_repo() -> None:
    sha = _git_commit()
    assert sha is not None
    assert re.fullmatch(r"[0-9a-f]{40}", sha)


def test_git_commit_is_cwd_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The console script runs detached from the repo; the recorded sha must be
    # this checkout's HEAD, not that of whatever repo the cwd happens to be in.
    from_repo = _git_commit()
    monkeypatch.chdir(tmp_path)
    assert _git_commit() == from_repo


def test_git_commit_returns_none_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["git"], returncode=128, stdout="", stderr="fatal:")

    monkeypatch.setattr(evals.reports.subprocess, "run", _fail)
    assert _git_commit() is None
    metadata = build_metadata("run", settings=_SettingsStandIn())
    assert metadata.git_commit is None


def test_git_commit_returns_none_when_git_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git")

    monkeypatch.setattr(evals.reports.subprocess, "run", _raise)
    assert _git_commit() is None


# --- save_as_baseline -------------------------------------------------------


def test_save_as_baseline_writes_fixed_shape(tmp_path: Path) -> None:
    run = _run(tmp_path / "reports")
    run.write_results({"accuracy": 0.9})
    source_doc = json.loads((run.path / "results.json").read_text())

    baseline_path = save_as_baseline(run.path, "extraction", baselines_root=tmp_path / "baselines")

    assert baseline_path == tmp_path / "baselines" / "extraction.json"
    baseline = json.loads(baseline_path.read_text())
    # Header keys at top level; the verbatim source doc nested under "source".
    assert set(baseline) == {"baseline_set_at", "run_label", "source"}
    assert baseline["run_label"] == "extraction-gpt4-prompt-v3"
    assert baseline["baseline_set_at"]
    assert baseline["source"] == source_doc
    assert baseline["source"]["metadata"]["run_label"] == "extraction-gpt4-prompt-v3"
    assert baseline["source"]["results"] == {"accuracy": 0.9}


def test_save_as_baseline_creates_baselines_root(tmp_path: Path) -> None:
    run = _run(tmp_path / "reports")
    run.write_results({})
    baselines_root = tmp_path / "nested" / "baselines"
    path = save_as_baseline(run.path, "extraction", baselines_root=baselines_root)
    assert path.is_file()


# --- diff_against_baseline --------------------------------------------------


def test_diff_returns_not_implemented_placeholder(tmp_path: Path) -> None:
    run = _run(tmp_path / "reports")
    run.write_results({})
    baseline = save_as_baseline(run.path, "extraction", baselines_root=tmp_path / "baselines")

    result = diff_against_baseline(baseline, run.path)

    assert isinstance(result, DiffResult)
    assert result.status == "not_implemented"
    assert result.baseline_path == str(baseline)
    assert result.current_path == str(run.path)
    assert result.changes == []
    assert "not implemented" in result.summary


def test_diff_missing_path_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="baseline"):
        diff_against_baseline(tmp_path / "missing.json", tmp_path)


# --- module identity --------------------------------------------------------


def test_reports_module_resolves_to_the_py_file_not_the_output_dir() -> None:
    # evals/reports.py (module) shares a name with evals/reports/ (output dir);
    # CPython must keep resolving the import to the module file.
    assert evals.reports.__file__ is not None
    assert evals.reports.__file__.endswith("reports.py")
