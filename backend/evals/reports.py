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
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

from pydantic import BaseModel

__all__ = [
    "RUN_STATUS_COMPLETED",
    "RUN_STATUS_FAILED",
    "DiffResult",
    "ReportRun",
    "RunMetadata",
    "SettingsLike",
    "build_metadata",
    "diff_against_baseline",
    "latest_run_dir",
    "save_as_baseline",
]

_SAFE_BASELINE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"

REPORTS_ROOT = Path(__file__).resolve().parents[1] / "evals" / "reports"
BASELINES_ROOT = Path(__file__).resolve().parents[1] / "evals" / "baselines"


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
    """Return the current ``HEAD`` sha, or ``None`` outside a repo / without git.

    Resolved against *this package's* directory rather than the process cwd: the
    ``rag-evals`` console script is cwd-independent by design (DECISIONS #1), so
    keying off cwd would record the sha of whatever unrelated repo the user
    happened to be standing in — silently wrong provenance in a committed
    baseline — or ``None`` when they stand outside a repo at all.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
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


def _create_run_dir(root: Path, name: str) -> Path:
    """Create a *fresh* run directory, disambiguating same-second collisions.

    The directory name only resolves to the second, so two runs sharing a label
    inside the same second (a retry, a loop over fixture sets, two concurrent
    invocations) would otherwise land in one directory and overwrite each
    other's ``results.json``. Claim the name with ``mkdir()`` — which is atomic
    — and fall back to ``<name>-2``, ``<name>-3``, … on ``FileExistsError``.
    """
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / name
    attempt = 1
    while True:
        try:
            candidate.mkdir()
        except FileExistsError:
            attempt += 1
            candidate = root / f"{name}-{attempt}"
        else:
            return candidate


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

    Creates ``<reports_root>/<ISO>-<slug(label)>/`` eagerly (with a ``-2``,
    ``-3``, … suffix if that name is already taken, so a run never writes into
    another run's directory) and captures ``metadata`` once at construction.
    On context exit the run directory is
    guaranteed to hold a ``results.json`` carrying at least the metadata, even
    if the caller never called :meth:`write_results`. Exceptions are never
    swallowed — but they *are* recorded: a run whose body raised is finalized
    with ``status="failed"``, so a crashed run cannot pass for an empty
    successful one (and :func:`save_as_baseline` refuses to promote it).
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
        self.path = _create_run_dir(root, _run_dir_name(started_at, label))
        self._results_written = False
        self._payload: dict[str, Any] = {}

    def write_summary(self, markdown: str) -> Path:
        """Write ``summary.md``; returns its path."""
        path = self.path / "summary.md"
        path.write_text(markdown, encoding="utf-8")
        return path

    def write_results(self, results: dict[str, Any] | BaseModel) -> Path:
        """Write ``results.json`` with the run metadata embedded; returns its path.

        The document shape is ``{"metadata": {...}, "status": ..., "results":
        {...}}`` so every results file (and any baseline copied from it) is
        self-describing. Pydantic results are serialized in JSON mode, so a
        model carrying ``datetime``/``UUID``/``Decimal`` fields round-trips
        instead of blowing up in ``json.dumps``.
        """
        self._payload = (
            results.model_dump(mode="json") if isinstance(results, BaseModel) else results
        )
        path = self._write_doc(RUN_STATUS_COMPLETED)
        self._results_written = True
        return path

    def _write_doc(self, status: str, *, error: str | None = None) -> Path:
        doc: dict[str, Any] = {
            "metadata": self.metadata.model_dump(),
            "status": status,
            "results": self._payload,
        }
        if error is not None:
            doc["error"] = error
        path = self.path / "results.json"
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        return path

    def write_per_item_breakdowns(self, markdown: str) -> Path:
        """Write the optional ``per_item_breakdowns.md``; returns its path."""
        path = self.path / "per_item_breakdowns.md"
        path.write_text(markdown, encoding="utf-8")
        return path

    def __enter__(self) -> ReportRun:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            # Keep whatever the caller managed to write, but mark the run failed:
            # an aborted eval must never look like a clean run with no findings.
            try:
                self._write_doc(RUN_STATUS_FAILED, error=exc_type.__name__)
            except (TypeError, ValueError):
                # The retained payload is what could not be serialized. Drop it
                # rather than raise out of __exit__ — an exception here would
                # replace the caller's real exception *and* leave a stale
                # results.json still claiming the run completed.
                self._payload = {}
                self._write_doc(RUN_STATUS_FAILED, error=exc_type.__name__)
        elif not self._results_written:
            self._write_doc(RUN_STATUS_COMPLETED)


def latest_run_dir(reports_root: Path | None = None) -> Path | None:
    """The most recent run directory under ``reports_root``, or ``None``.

    Run directory names start with a colon-free ISO timestamp, so plain
    lexicographic order is chronological. Dot-prefixed entries (e.g. the
    ``.judge_cache`` directory) are not run dirs and are skipped.
    """
    root = REPORTS_ROOT if reports_root is None else reports_root
    if not root.is_dir():
        return None
    candidates = sorted(
        entry for entry in root.iterdir() if entry.is_dir() and not entry.name.startswith(".")
    )
    return candidates[-1] if candidates else None


def _update_run_results(run_dir: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    """Read-modify-write an *existing* run's ``results.json`` (DECISIONS #7).

    ``judge-alignment`` and ``confidence-review`` run after ``extraction``
    against the run dir it created; constructing a ``ReportRun`` would mint a
    new empty timestamped dir and orphan their sections. ``mutate`` receives
    the ``results`` payload (inside the ``{"metadata", "results"}`` envelope)
    and edits it in place; ``metadata``/``status`` and the sections other
    phases own are preserved untouched.
    """
    path = run_dir / "results.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    mutate(doc["results"])
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _append_run_summary(run_dir: Path, block: str) -> Path:
    """Append a block to an existing run's ``summary.md`` (creates it if absent)."""
    path = run_dir / "summary.md"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + block.rstrip("\n") + "\n", encoding="utf-8")
    return path


def save_as_baseline(
    report_path: Path, baseline_name: str, *, baselines_root: Path | None = None
) -> Path:
    """Copy a run's ``results.json`` into the committed baselines dir.

    Fixed baseline shape: ``{"baseline_set_at": <ISO now>, "run_label": <from
    the source metadata>, "source": <verbatim results.json doc>}`` — the header
    keys sit at top level and the entire source doc is nested under ``source``,
    so the copied provenance is never clobbered or duplicated by the header.

    Refuses to promote a run finalized as ``failed``: a baseline is the
    reference every later run is judged against, so silently blessing a crashed
    run would turn its empty results into "the expected numbers".

    ``baseline_name`` must be a single safe path segment. Epic 16 surfaces it as
    a user-supplied ``--name``, and an unchecked name is pasted straight into a
    path: ``../reports/results`` would escape the baselines dir and an absolute
    path would discard it entirely, silently clobbering an unrelated file.
    """
    if not _SAFE_BASELINE_NAME.fullmatch(baseline_name):
        raise ValueError(
            f"invalid baseline name {baseline_name!r}: expected a single path segment "
            f"of letters, digits, '.', '_' or '-'"
        )
    root = BASELINES_ROOT if baselines_root is None else baselines_root
    source_doc = json.loads((report_path / "results.json").read_text(encoding="utf-8"))
    if source_doc.get("status") == RUN_STATUS_FAILED:
        raise ValueError(
            f"refusing to baseline a failed run: {report_path} "
            f"(error: {source_doc.get('error', 'unknown')})"
        )
    baseline = {
        "baseline_set_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_label": source_doc["metadata"]["run_label"],
        "source": source_doc,
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{baseline_name}.json"
    path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    return path


class DiffResult(BaseModel):
    """Regression-diff result; the extraction branch fills ``changes`` (Epic 15).

    Retrieval metrics join in Epic 16. The field set is fixed — downstream
    phases fill it, they do not reshape it.
    """

    baseline_path: str
    current_path: str
    status: str = "not_implemented"
    summary: str
    changes: list[dict[str, Any]] = []


# Absolute tolerance for regression flagging (DECISIONS #5): suppresses LLM
# nondeterminism / floating-point jitter so [REGRESSION] means a real drop.
# Only a move *beyond* the tolerance counts, and binary floating point makes
# nominally-exact boundary deltas overshoot (0.91 - 0.92 == -0.010000000000000009),
# so the comparison carries a representation-error slack.
_DIFF_TOLERANCE = 0.01
_TOLERANCE_SLACK = 1e-9

_COUNT_KEYS = (
    "fixtures",
    "recipes_extracted",
    "extraction_failures",
    "over_split_fixtures",
    "ready",
    "needs_review",
)

_DIRECTION_HIGHER = "higher_is_better"
_DIRECTION_INFO = "informational"

FLAG_REGRESSION = "[REGRESSION]"


def _scalar_metrics(results: dict[str, Any]) -> dict[str, float | int | None]:
    """Flatten an extraction ``results`` payload to its comparable scalar metrics.

    Per-field objective accuracy (15.1), item counts (15.1), judge pass rate
    (15.2), and judge-human agreement (15.3). Calibration is deliberately a
    review view, not a diffed metric (see ``confidence-review``).
    """
    metrics: dict[str, float | int | None] = {}
    aggregate = results.get("aggregate") or {}
    for field, value in (aggregate.get("field_accuracy") or {}).items():
        metrics[f"field_accuracy.{field}"] = value
    for count_key in _COUNT_KEYS:
        if count_key in aggregate:
            metrics[f"count.{count_key}"] = aggregate[count_key]
    judge = results.get("judge") or {}
    if judge:
        metrics["judge.pass_rate"] = judge.get("pass_rate")
    agreement = results.get("agreement") or {}
    if agreement:
        metrics["agreement.rate"] = agreement.get("agreement_rate")
    return metrics


def _metric_direction(metric: str) -> str:
    return _DIRECTION_INFO if metric.startswith("count.") else _DIRECTION_HIGHER


def _lost_coverage_is_a_regression(metric: str) -> bool:
    """Whether a metric the baseline measured, and this run did not, is a drop.

    Objective accuracy stops being measurable only when the fixtures it covers
    stopped producing scores — e.g. every extraction was rejected or truncated,
    which leaves the run ``completed`` with survivor-only (or no) accuracy. Left
    as plain ``missing`` that scenario prints "No regressions detected" for a
    total extraction collapse. Judge pass rate and judge-human agreement are
    different: they are absent whenever the optional ``--judge`` /
    ``judge-alignment`` steps simply were not run, so their absence stays
    informational.
    """
    return metric.startswith("field_accuracy.")


def _diff_extraction(
    current: dict[str, Any], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per-metric deltas between two unwrapped extraction ``results`` payloads.

    Direction per DECISIONS #5: accuracy / pass-rate / agreement are
    higher-is-better and flag ``[REGRESSION]`` on a drop beyond the tolerance;
    counts are informational context. A metric present on only one side is
    ``new`` / ``missing`` and never crashes — except that an *accuracy* metric
    the baseline carried and this run cannot measure is a regression, not a
    shrug (see :func:`_lost_coverage_is_a_regression`).
    """
    current_metrics = _scalar_metrics(current)
    baseline_metrics = _scalar_metrics(baseline)
    changes: list[dict[str, Any]] = []
    for metric in sorted(set(current_metrics) | set(baseline_metrics)):
        baseline_value = baseline_metrics.get(metric)
        current_value = current_metrics.get(metric)
        direction = _metric_direction(metric)
        delta: float | None = None
        if baseline_value is None and current_value is None:
            flag = "missing"
        elif baseline_value is None:
            flag = "new"
        elif current_value is None:
            flag = FLAG_REGRESSION if _lost_coverage_is_a_regression(metric) else "missing"
        else:
            delta = current_value - baseline_value
            threshold = _DIFF_TOLERANCE + _TOLERANCE_SLACK
            if direction == _DIRECTION_INFO:
                flag = "info"
            elif delta < -threshold:
                flag = FLAG_REGRESSION
            elif delta > threshold:
                flag = "improved"
            else:
                flag = "unchanged"
        changes.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "current": current_value,
                "delta": delta,
                "direction": direction,
                "flag": flag,
            }
        )
    return changes


def _render_diff_summary(changes: list[dict[str, Any]]) -> str:
    def fmt(value: float | int | None) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, int):
            return str(value)
        return f"{value:.2f}"

    lines = ["Extraction diff vs baseline:"]
    for change in changes:
        rendered = f"{change['metric']}: {fmt(change['baseline'])} -> {fmt(change['current'])}"
        if change["delta"] is not None and change["direction"] == _DIRECTION_HIGHER:
            rendered += f" (delta {change['delta']:+.2f})"
        if change["flag"] == FLAG_REGRESSION:
            if change["current"] is None:
                rendered += " (no longer measured)"
            rendered = f"{FLAG_REGRESSION} {rendered}"
        elif change["flag"] in ("new", "missing", "improved"):
            rendered += f" ({change['flag']})"
        lines.append(f"  {rendered}")
    regressions = [change for change in changes if change["flag"] == FLAG_REGRESSION]
    lines.append(
        f"{len(regressions)} regression(s) detected."
        if regressions
        else "No regressions detected."
    )
    return "\n".join(lines)


def _refuse_failed(doc: dict[str, Any], path: Path, kind: str) -> None:
    """Refuse to diff a run finalized as ``failed`` (mirrors ``save_as_baseline``).

    A crashed run's ``results`` is empty or partial, so every metric would
    unwrap to ``None`` — read as "missing", never a regression — and the diff
    would return ``ok`` / "No regressions detected" for a run that never
    produced numbers. A false green here is worse than no diff at all.
    """
    if doc.get("status") == RUN_STATUS_FAILED:
        raise ValueError(
            f"refusing to diff a failed {kind}: {path} "
            f"(error: {doc.get('error', 'unknown')})"
        )


def diff_against_baseline(baseline_path: Path, current_report_path: Path) -> DiffResult:
    """Diff a run's extraction metrics against a committed baseline (doc 12 § 9).

    ``current_report_path`` is the run *directory* (mirroring
    ``save_as_baseline``, which reads ``report_path / "results.json"``). The two
    inputs are wrapped differently and are unwrapped before comparing:
    ``current = doc["results"]`` (the ``{"metadata", "results"}`` envelope) but
    ``baseline = doc["source"]["results"]`` (``save_as_baseline`` nests the
    whole results document under ``source``). Missing inputs raise
    ``FileNotFoundError`` — a diff against a nonexistent baseline or report is
    a caller error, not a "no changes" result — and a run finalized as
    ``failed`` on either side raises ``ValueError`` rather than diffing its
    empty metrics into a clean bill of health.
    """
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline not found: {baseline_path}")
    if not current_report_path.exists():
        raise FileNotFoundError(f"report not found: {current_report_path}")
    current_results_path = current_report_path / "results.json"
    if not current_results_path.is_file():
        raise FileNotFoundError(f"report results not found: {current_results_path}")
    current_doc = json.loads(current_results_path.read_text(encoding="utf-8"))
    baseline_doc = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_source = baseline_doc.get("source") or {}
    _refuse_failed(current_doc, current_report_path, "report")
    _refuse_failed(baseline_source, baseline_path, "baseline")
    current = current_doc.get("results") or {}
    baseline = baseline_source.get("results") or {}
    changes = _diff_extraction(current, baseline)
    has_regressions = any(change["flag"] == FLAG_REGRESSION for change in changes)
    return DiffResult(
        baseline_path=str(baseline_path),
        current_path=str(current_report_path),
        status="regressions_detected" if has_regressions else "ok",
        summary=_render_diff_summary(changes),
        changes=changes,
    )
