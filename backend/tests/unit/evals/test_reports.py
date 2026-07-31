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
from evals.reports import ReportRun, _git_commit, build_metadata

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
    assert json.loads((run.path / "results.json").read_text())["results"] == {"accuracy": 0.9}


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


# --- module identity --------------------------------------------------------


def test_reports_module_resolves_to_the_py_file_not_the_output_dir() -> None:
    # evals/reports.py (module) shares a name with evals/reports/ (output dir);
    # CPython must keep resolving the import to the module file.
    assert evals.reports.__file__ is not None
    assert evals.reports.__file__.endswith("reports.py")
