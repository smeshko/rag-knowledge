"""Report writer for evaluation runs (doc 12 § 10, doc 13 topic 10).

A ``ReportRun`` creates a filesystem-safe timestamped run directory under
``evals/reports/``, captures run metadata once (timestamp, git commit,
embedding/LLM provider + model, prompt/schema version, argv), and writes
``summary.md`` / ``results.json`` / optional ``per_item_breakdowns.md``.

Metadata sources: an injectable ``SettingsLike`` object (defaulting to the real
``get_settings()``), the ``PROMPT_VERSION``/``SCHEMA_VERSION`` constants from
``rag_recipes.ingestion.pipeline.extraction`` (imported lazily so this module
stays light), ``sys.argv``, and ``git rev-parse HEAD``. Only the named model
fields are copied off settings — never a wholesale ``model_dump()``, because
baselines built from ``results.json`` are committed to git and a full dump
would leak credentials.

Naming note: this module (``evals/reports.py``) intentionally shares the dotted
name ``evals.reports`` with the ``evals/reports/`` *output* directory. CPython
resolves the import to the ``.py`` module because the extension-less directory
has no ``__init__.py`` — never add an ``evals/reports/__init__.py``, which
would silently shadow this module.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

from pydantic import BaseModel

__all__ = [
    "ReportRun",
    "RunMetadata",
    "SettingsLike",
    "build_metadata",
]

REPORTS_ROOT = Path(__file__).resolve().parents[1] / "evals" / "reports"


class SettingsLike(Protocol):
    """The five settings attributes the report metadata reads (and nothing more)."""

    @property
    def embedding_provider(self) -> str: ...

    @property
    def embedding_model(self) -> str: ...

    @property
    def llm_provider(self) -> str: ...

    @property
    def llm_model(self) -> str: ...

    @property
    def anthropic_llm_model(self) -> str: ...


class RunMetadata(BaseModel):
    """Provenance stamped into every ``results.json`` (and thus every baseline)."""

    timestamp: str
    git_commit: str | None
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    prompt_version: str
    schema_version: str
    command_args: list[str]
    run_label: str


def _git_commit() -> str | None:
    """Return the current ``HEAD`` sha, or ``None`` outside a repo / without git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _slugify(label: str) -> str:
    """Reduce a run label to a filesystem-safe directory-name segment."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-.").lower()


def _run_dir_name(timestamp: datetime, label: str) -> str:
    """``YYYY-MM-DDTHH-MM-SS-<slug>`` — colon-free ISO timestamp (DECISIONS #2)."""
    return f"{timestamp.strftime('%Y-%m-%dT%H-%M-%S')}-{_slugify(label)}"


def build_metadata(label: str, *, settings: SettingsLike | None = None) -> RunMetadata:
    """Capture run metadata; resolves the LLM model through the provider.

    ``settings`` is injectable so tests stay hermetic; ``None`` falls back to the
    real ``get_settings()``. Reading ``settings.llm_model`` unconditionally would
    stamp ``gpt-4.1`` on an Anthropic run, so the model is resolved through
    ``llm_provider`` (DECISIONS #4). Only the five named fields are read.
    """
    if settings is None:
        from rag_recipes.config import get_settings

        settings = get_settings()
    # Lazy import: keeps `import evals.reports` (and the rag-evals CLI) from
    # pulling in the whole extraction pipeline at startup.
    from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION

    llm_model = (
        settings.anthropic_llm_model
        if settings.llm_provider == "anthropic"
        else settings.llm_model
    )
    return RunMetadata(
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=_git_commit(),
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        llm_provider=settings.llm_provider,
        llm_model=llm_model,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        command_args=list(sys.argv),
        run_label=label,
    )


class ReportRun:
    """A single evaluation run's output directory, usable as a context manager.

    Creates ``<reports_root>/<ISO>-<slug(label)>/`` eagerly and captures
    ``metadata`` once at construction. On context exit the run directory is
    guaranteed to hold a ``results.json`` carrying at least the metadata, even
    if the caller never called :meth:`write_results`. Exceptions are never
    swallowed.
    """

    def __init__(
        self,
        label: str,
        *,
        reports_root: Path | None = None,
        settings: SettingsLike | None = None,
    ) -> None:
        root = REPORTS_ROOT if reports_root is None else reports_root
        self.metadata = build_metadata(label, settings=settings)
        started_at = datetime.fromisoformat(self.metadata.timestamp)
        self.path = root / _run_dir_name(started_at, label)
        self.path.mkdir(parents=True, exist_ok=True)
        self._results_written = False

    def write_summary(self, markdown: str) -> Path:
        """Write ``summary.md``; returns its path."""
        path = self.path / "summary.md"
        path.write_text(markdown)
        return path

    def write_results(self, results: dict[str, Any] | BaseModel) -> Path:
        """Write ``results.json`` with the run metadata embedded; returns its path.

        The document shape is ``{"metadata": {...}, "results": {...}}`` so every
        results file (and any baseline copied from it) is self-describing.
        """
        payload = results.model_dump() if isinstance(results, BaseModel) else results
        doc = {"metadata": self.metadata.model_dump(), "results": payload}
        path = self.path / "results.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        self._results_written = True
        return path

    def write_per_item_breakdowns(self, markdown: str) -> Path:
        """Write the optional ``per_item_breakdowns.md``; returns its path."""
        path = self.path / "per_item_breakdowns.md"
        path.write_text(markdown)
        return path

    def __enter__(self) -> ReportRun:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._results_written:
            self.write_results({})
