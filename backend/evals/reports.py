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
from dataclasses import dataclass
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
    "RetrievalDiff",
    "RunMetadata",
    "SettingsLike",
    "build_metadata",
    "diff_against_baseline",
    "diff_retrieval",
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
    results_path = report_path / "results.json"
    if not results_path.is_file():
        # A report *directory* is the contract; pointing at results.json itself
        # would otherwise die with an unhandled NotADirectoryError.
        raise FileNotFoundError(f"report results not found: {results_path}")
    source_doc = _load_json_object(results_path, "report")
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
    """Placeholder regression-diff result; Epics 15/16 fill ``changes``/metrics."""

    baseline_path: str
    current_path: str
    status: str = "not_implemented"
    summary: str
    changes: list[dict[str, Any]] = []


# --- retrieval regression diff (Epic 16 Phase 16.2) --------------------------

#: Display labels for the four retrieval measures, in headline order.
_RETRIEVAL_METRIC_LABELS = (
    ("ndcg_cut_10", "NDCG@10"),
    ("recall_5", "Recall@5"),
    ("recall_10", "Recall@10"),
    ("recip_rank", "MRR"),
)

#: ``run``-block fields that make two runs incomparable when they differ
#: (Epic 18's rerank stage changes item ordering materially).
_COMPARABILITY_FIELDS = ("query_set", "mode", "reranking_enabled", "embedding_model")

#: Per-query regression thresholds (fixed by the epic): an expected item that
#: dropped out of the top-k, or whose rank worsened by at least this much.
_RANK_DROP_THRESHOLD = 3


@dataclass(frozen=True)
class RetrievalDiff:
    """Pure diff of two unwrapped retrieval results payloads.

    ``diff_against_baseline`` projects this onto :class:`DiffResult`; keeping
    the rich shape separate leaves ``DiffResult`` (shared with Epic 15)
    untouched.
    """

    overall: dict[str, dict[str, float]]
    per_query_ndcg_delta: dict[str, float]
    added_query_ids: list[str]
    removed_query_ids: list[str]
    regressions: list[dict[str, Any]]
    worst_queries: list[dict[str, Any]]
    warnings: list[str]
    headline: str
    status: str


def _unwrap(doc: dict[str, Any]) -> dict[str, Any]:
    """Normalise either diff input to the inner results payload.

    The two inputs are differently-shaped envelopes: a baseline file is
    ``{"baseline_set_at", "run_label", "source": {"metadata", ..., "results"}}``
    while a report's ``results.json`` is ``{"metadata", ..., "results"}`` —
    and ``report_type`` lives *inside* the inner payload, so top-level reads
    would silently dispatch nothing. A bare payload passes through unchanged.
    """
    if "baseline_set_at" in doc and isinstance(doc.get("source"), dict):
        source = doc["source"]
        results = source.get("results")
        return results if isinstance(results, dict) else source
    if "metadata" in doc and isinstance(doc.get("results"), dict):
        results_doc: dict[str, Any] = doc["results"]
        return results_doc
    return doc


def _run_status(doc: dict[str, Any]) -> str | None:
    """The finalized run status carried by either diff-input envelope shape.

    ``_unwrap`` deliberately discards the envelope, but ``status`` is the one
    envelope field the diff must not ignore: a run finalized as ``failed``
    keeps whatever partial payload it had managed to write, so diffing it
    would report a confident "no change" (or a fabricated regression) over
    results that were never produced.
    """
    if "baseline_set_at" in doc and isinstance(doc.get("source"), dict):
        doc = doc["source"]
    status = doc.get("status")
    return status if isinstance(status, str) else None


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read a JSON **object** from ``path``, or raise a caller-facing ``ValueError``."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {label} {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(
            f"expected a JSON object in {label} {path}, got {type(doc).__name__}"
        )
    return doc


def _headline_tag(delta: float, threshold: float) -> str:
    # Strictly past the threshold tags; |delta| == threshold is "no change".
    # The epsilon keeps float noise (0.595 - 0.600 != -0.005 exactly) from
    # flipping the boundary case.
    epsilon = 1e-9
    if delta < -(threshold + epsilon):
        return f"[REGRESSION {delta:+.2f}]"
    if delta > threshold + epsilon:
        return f"[IMPROVEMENT {delta:+.2f}]"
    return "[no change]"


def _query_regressions(
    query_id: str,
    baseline_ranks: dict[str, Any],
    current_ranks: dict[str, Any],
    *,
    baseline_k: int,
    current_k: int,
) -> list[dict[str, Any]]:
    """Flag expected items that left the top-k or dropped ≥ 3 ranks."""
    entries: list[dict[str, Any]] = []
    for item_id, baseline_rank in baseline_ranks.items():
        if baseline_rank is None or baseline_rank > baseline_k:
            continue  # was not in the baseline top-k: nothing to regress from
        current_rank = current_ranks.get(item_id)
        if current_rank is None or current_rank > current_k:
            reason = "dropped_from_top_k"
        elif current_rank - baseline_rank >= _RANK_DROP_THRESHOLD:
            reason = "rank_drop"
        else:
            continue
        entries.append(
            {
                "query_id": query_id,
                "item_id": item_id,
                "baseline_rank": baseline_rank,
                "current_rank": current_rank,
                "reason": reason,
            }
        )
    return entries


def diff_retrieval(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    headline_threshold: float = 0.005,
    top_n: int = 5,
) -> RetrievalDiff:
    """Diff two **unwrapped** retrieval payloads; pure, no I/O.

    Overall deltas are ``current − baseline`` per measure; per-query NDCG@10
    deltas cover the baseline∩current query-id intersection (ids present in
    only one run are reported as added/removed, never as a delta against
    zero). The headline tags a metric only strictly past
    ``headline_threshold``; per-query regression thresholds are fixed
    (dropped from top-k, or rank-drop ≥ 3). A ``run``-block mismatch on
    query_set / mode / reranking_enabled / embedding_model prepends a
    prominent warning — such runs are not comparable.
    """
    baseline_run = baseline.get("run", {})
    current_run = current.get("run", {})
    warnings = [
        f"WARNING: runs are not comparable — {field} differs "
        f"(baseline={baseline_run.get(field)!r}, current={current_run.get(field)!r})"
        for field in _COMPARABILITY_FIELDS
        if baseline_run.get(field) != current_run.get(field)
    ]

    overall: dict[str, dict[str, float]] = {}
    headline_lines = list(warnings)
    tags: dict[str, str] = {}
    for measure, label in _RETRIEVAL_METRIC_LABELS:
        baseline_value = float(baseline.get("aggregate", {}).get(measure, 0.0))
        current_value = float(current.get("aggregate", {}).get(measure, 0.0))
        delta = current_value - baseline_value
        overall[measure] = {
            "baseline": baseline_value,
            "current": current_value,
            "delta": delta,
        }
        tag = _headline_tag(delta, headline_threshold)
        tags[measure] = tag
        headline_lines.append(f"{label}: {baseline_value:.2f} → {current_value:.2f} {tag}")

    baseline_queries = baseline.get("per_query", {})
    current_queries = current.get("per_query", {})
    shared_ids = sorted(set(baseline_queries) & set(current_queries))
    added = sorted(set(current_queries) - set(baseline_queries))
    removed = sorted(set(baseline_queries) - set(current_queries))

    per_query_ndcg_delta: dict[str, float] = {}
    regressions: list[dict[str, Any]] = []
    baseline_k = int(baseline_run.get("k", 10))
    current_k = int(current_run.get("k", 10))
    for query_id in shared_ids:
        baseline_query = baseline_queries[query_id]
        current_query = current_queries[query_id]
        per_query_ndcg_delta[query_id] = float(
            current_query["metrics"]["ndcg_cut_10"]
        ) - float(baseline_query["metrics"]["ndcg_cut_10"])
        regressions.extend(
            _query_regressions(
                query_id,
                baseline_query.get("expected_item_ranks", {}),
                current_query.get("expected_item_ranks", {}),
                baseline_k=baseline_k,
                current_k=current_k,
            )
        )

    worst_queries = [
        {"query_id": query_id, "delta": delta}
        for query_id, delta in sorted(per_query_ndcg_delta.items(), key=lambda entry: entry[1])
        if delta < 0
    ][:top_n]

    if regressions or any(tag.startswith("[REGRESSION") for tag in tags.values()):
        status = "regression"
    elif any(tag.startswith("[IMPROVEMENT") for tag in tags.values()):
        status = "improvement"
    else:
        status = "no_change"

    return RetrievalDiff(
        overall=overall,
        per_query_ndcg_delta=per_query_ndcg_delta,
        added_query_ids=added,
        removed_query_ids=removed,
        regressions=regressions,
        worst_queries=worst_queries,
        warnings=warnings,
        headline="\n".join(headline_lines),
        status=status,
    )


def _render_retrieval_summary(diff: RetrievalDiff) -> str:
    """The full printable diff block: headline, regressions, worst queries."""
    lines = [diff.headline]
    if diff.regressions:
        lines += ["", "Per-query regressions:"]
        lines += [
            f"- {entry['query_id']}: {entry['item_id']} "
            f"rank {entry['baseline_rank']} → {entry['current_rank']} ({entry['reason']})"
            for entry in diff.regressions
        ]
    if diff.worst_queries:
        lines += ["", "Biggest NDCG@10 drops:"]
        lines += [
            f"- {entry['query_id']}: {entry['delta']:+.4f}" for entry in diff.worst_queries
        ]
    if diff.added_query_ids:
        lines += ["", f"Queries only in current run: {', '.join(diff.added_query_ids)}"]
    if diff.removed_query_ids:
        lines += ["", f"Queries only in baseline run: {', '.join(diff.removed_query_ids)}"]
    return "\n".join(lines)


def diff_against_baseline(baseline_path: Path, current_report_path: Path) -> DiffResult:
    """Diff a report run against a committed baseline, dispatching by type.

    The signature is asymmetric by design: ``baseline_path`` is a JSON
    **file** (``save_as_baseline`` output) while ``current_report_path`` is a
    report **directory** whose ``results.json`` is loaded. Both envelopes are
    normalised via :func:`_unwrap` before dispatch on the inner payload's
    ``report_type`` — retrieval pairs route to :func:`diff_retrieval`; other
    types (extraction is Epic 15) keep the placeholder result. Missing inputs
    raise ``FileNotFoundError`` — a diff against a nonexistent baseline or
    report is a caller error, not a "no changes" result.

    Three further input errors raise instead of falling through to the
    placeholder — each would otherwise be reported as a clean, exit-0 diff:
    a run finalized ``failed`` on either side, a ``report_type`` mismatch
    between the two sides, and malformed / non-object JSON.
    """
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline not found: {baseline_path}")
    if not current_report_path.is_dir():
        # A report *directory* is the contract; pointing at results.json
        # itself would otherwise die with an unhandled NotADirectoryError.
        raise FileNotFoundError(f"report directory not found: {current_report_path}")
    results_path = current_report_path / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"report results not found: {results_path}")

    baseline_doc = _load_json_object(baseline_path, "baseline")
    current_doc = _load_json_object(results_path, "report")
    for label, path, doc in (
        ("baseline", baseline_path, baseline_doc),
        ("report", results_path, current_doc),
    ):
        if _run_status(doc) == RUN_STATUS_FAILED:
            raise ValueError(f"refusing to diff a failed run: {label} {path}")

    baseline_payload = _unwrap(baseline_doc)
    current_payload = _unwrap(current_doc)
    baseline_type = baseline_payload.get("report_type")
    current_type = current_payload.get("report_type")
    if baseline_type != current_type:
        raise ValueError(
            f"report type mismatch: baseline is {baseline_type!r}, "
            f"report is {current_type!r}"
        )
    if baseline_type == "retrieval":
        diff = diff_retrieval(baseline_payload, current_payload)
        return DiffResult(
            baseline_path=str(baseline_path),
            current_path=str(current_report_path),
            status=diff.status,
            summary=_render_retrieval_summary(diff),
            changes=diff.regressions,
        )
    return DiffResult(
        baseline_path=str(baseline_path),
        current_path=str(current_report_path),
        summary="diff not implemented yet — Epics 15/16 fill this in",
    )
