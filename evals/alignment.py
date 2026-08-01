"""Judge-alignment workflow: human-vs-judge agreement (Epic 15 Phase 15.3,
integrity fixes in Epic 20 Phase 20.1).

Implements the doc-12 § 5 alignment loop over a *prior extraction run*: the
run's ``results.json`` supplies each fixture's persisted extracted artifact
(serialized as ``{"items": [...]}`` by the shared
``serialize_extracted_artifact`` helper), the human and the LLM judge both rate
exactly that byte-identical string, agreement is computed over the fixtures
both sides rated, and disagreements are listed with **both** critiques for
prompt iteration. Alignment never re-extracts — a fixture with no persisted
artifact in the run (extraction failed, hard validation failed, or a run
written before artifacts were persisted) is recorded *unrated* and the human
is never prompted for it (DECISIONS #4).

:func:`load_alignment_run` is the public pre-flight validator (DECISIONS #9):
every unusable run shape — no run at all, a missing dir or ``results.json``,
malformed JSON, a ``failed`` run, a retrieval run, a fixture-set mismatch, or
a fixture edited since the run — raises ``ValueError`` *before* the first
human prompt (and, in the CLI, before ``get_settings()`` or a provider exist).

Judge ratings replay from the Phase-15.2 cache under the run's *recorded*
provenance — ``fixture_content_hash`` and ``extraction_prompt_version`` come
from ``results.json``, never from the current working tree — plus the judged
artifact's hash (DECISIONS #2/#10); the injected provider is only consulted on
a cache miss, and a miss writes the entry back with ``artifact_hash`` stamped
into its metadata, exactly like the driver's own cache writes.

Records are set- and provider-scoped: the composite id is
``<fixture_set>__<fixture>__<judge>__<extraction_model>`` (each part through
``slugify_key_part``), where the model is the run's ``metadata.llm_model`` —
the *artifact's* producer, not the judge's provider (DECISIONS #6). Both
critiques plus provenance live in ``run_metadata`` under the documented keys
``{"judge_name", "judge_version", "model", "fixture_name", "fixture_set",
"human_critique", "judge_critique", "rated_at", "artifact_hash",
"judge_dimension", "extraction_provider", "extraction_model", "aligned_run"}``.

Human ratings are reused only while the judge version **and** the artifact
hash both match (DECISIONS #5) — a re-extraction that changed the artifact
re-prompts, and an old record without a hash re-prompts. The human is told
which judge dimension they are rating: ``_judge_dimension`` lifts the
``Rate exactly ONE subjective dimension: … ?`` sentence verbatim from the
judge prompt, falling back to the judge name when absent (DECISIONS #11). On
the unrated path an existing record carrying a human rating is left on disk
untouched — a collected human rating is never destroyed, and no verdict is
asserted about an artifact that no longer exists (DECISIONS #4).

The interactive prompt is an injected callable (DECISIONS #3) — tests script
it; only the CLI uses the real ``typer.prompt``. The agreement metric is
written into the *existing* extraction run's ``results.json`` at
``results.agreement`` via the read-modify-write helper (DECISIONS #7), never
via a new ``ReportRun``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import typer
from pydantic import BaseModel

from evals.extraction import (
    artifact_hash,
    build_judge_cache_key,
    serialize_extracted_artifact,
)
from evals.fixtures import (
    load_judge_alignment,
    load_judge_prompt,
    load_recipe_fixtures,
    save_judge_alignment,
)
from evals.judge_cache import JudgeCache, slugify_key_part
from evals.judges import JudgeError, JudgeRating, load_judge
from evals.models import JudgeAlignmentRecord, RecipeFixture
from evals.reports import _append_run_summary, _update_run_results
from rag_recipes.providers.llm.base import LLMProvider

__all__ = [
    "AlignmentDisagreement",
    "AlignmentReport",
    "PromptHuman",
    "load_alignment_run",
    "run_judge_alignment",
]

HumanRating = Literal["pass", "fail"]

PromptHuman = Callable[[RecipeFixture, str, str, str], tuple[HumanRating, str]]
"""Collect a human ``("pass"|"fail", critique)`` for one fixture.

Called as ``(fixture, artifact, judge_name, dimension)``: ``artifact`` is the
exact serialized string the judge rates (byte-identical), and ``dimension`` is
the judge's dimension sentence so both sides answer the same question
(DECISIONS #11). The rating shares ``JudgeRating.rating``'s domain — anything
else would make the human-vs-judge comparison silently never match.
"""

_DIMENSION_MARKER = "Rate exactly ONE subjective dimension:"


class AlignmentDisagreement(BaseModel):
    """One fixture where human and judge disagree — both critiques attached."""

    fixture_name: str
    human_rating: str
    judge_rating: str
    human_critique: str
    judge_critique: str


class AlignmentReport(BaseModel):
    """Aggregate outcome of one judge-alignment pass over a fixture set."""

    judge_name: str
    judge_version: str
    model: str
    agreement_rate: float | None
    rated: int
    agreements: int
    disagreements: list[AlignmentDisagreement]
    unrated: list[str]
    records: list[JudgeAlignmentRecord]

    def agreement_payload(self) -> dict[str, object]:
        """The ``results.agreement`` section written into the run's report."""
        return {
            "judge_name": self.judge_name,
            "judge_version": self.judge_version,
            "model": self.model,
            "agreement_rate": self.agreement_rate,
            "rated": self.rated,
            "agreements": self.agreements,
            "disagreements": [d.model_dump() for d in self.disagreements],
            "unrated": list(self.unrated),
        }


def _judge_dimension(prompt_text: str, judge_name: str) -> str:
    """The judge's dimension sentence, lifted verbatim from its prompt.

    The extraction rule is a parsing contract against the committed prompts
    (DECISIONS #11): the substring from the literal marker ``Rate exactly ONE
    subjective dimension:`` up to and including the first ``?`` after it, with
    all whitespace/newlines collapsed to single spaces and ``**`` emphasis
    stripped. The sentence is *not* a line — it starts mid-line and wraps.
    Absent the marker (or a ``?``), falls back to the judge name alone.
    """
    start = prompt_text.find(_DIMENSION_MARKER)
    if start == -1:
        return judge_name
    end = prompt_text.find("?", start)
    if end == -1:
        return judge_name
    sentence = prompt_text[start : end + 1]
    return re.sub(r"\s+", " ", sentence).replace("**", "").strip()


def _default_prompt_human(
    fixture: RecipeFixture, artifact: str, judge_name: str, dimension: str
) -> tuple[HumanRating, str]:
    """Interactive typer prompt — the CLI path; tests inject a scripted fake."""
    typer.echo(f"\n=== {fixture.name} ===")
    typer.echo(f"Judge dimension ({judge_name}): {dimension}")
    typer.echo("\n--- Source ---")
    typer.echo(fixture.source_md)
    typer.echo("--- Extracted artifact (exactly what the judge rates) ---")
    typer.echo(artifact)
    while True:
        raw = typer.prompt("Rating (pass/fail)").strip().lower()
        if raw in ("pass", "fail"):
            break
        typer.echo("Please answer 'pass' or 'fail'.")
    critique = typer.prompt("Critique")
    return cast(HumanRating, raw), critique


def _composite_id(
    fixture_set: str, fixture_name: str, judge_name: str, extraction_model: str
) -> str:
    """Set-, judge- and extraction-model-scoped record id (DECISIONS #6).

    ``extraction_model`` is the run's ``metadata.llm_model`` — the model that
    produced the rated *artifact*, not the judge's provider — so 20.3's
    dual-provider session keeps one record per (set, fixture, judge, provider).
    Every part goes through the shared ``slugify_key_part``.
    """
    parts = (fixture_set, fixture_name, judge_name, extraction_model)
    return "__".join(slugify_key_part(part) for part in parts)


def _run_fix_hint(fixture_set: str) -> str:
    return f"run `rag-evals extraction --fixtures {fixture_set} --label <label>` first"


def _per_fixture_index(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The run's per-fixture entries keyed by fixture name, read defensively.

    Every key of a foreign ``results.json`` is read with ``.get()`` and its
    *shape* checked (DECISIONS #9): a payload whose ``per_fixture`` is ``null``
    or a scalar — a hand-edited or truncated file — would otherwise raise
    ``TypeError`` from the iteration, and the CLI catches only
    ``(FileNotFoundError, ValueError)``, so a bad-run shape would surface as a
    traceback instead of exit 2. An unusable ``per_fixture`` degrades to "no
    entries", which routes every fixture down the unrated path.
    """
    entries = results.get("per_fixture")
    if not isinstance(entries, list):
        return {}
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str):
                index[name] = entry
    return index


def load_alignment_run(
    run_dir: Path | None, fixture_set: str, fixtures: list[RecipeFixture]
) -> dict[str, Any]:
    """Validate an extraction run for alignment; returns its ``results`` payload.

    The whole pre-flight validation pass (DECISIONS #9): called by
    ``run_judge_alignment`` before its fixture loop *and* by the CLI before
    ``get_settings()``/provider construction, so every unusable-run shape dies
    with a clean ``ValueError`` (CLI exit 2) instead of a traceback — and
    before the human has rated anything. Checks, in order: a run dir exists,
    its ``results.json`` exists and parses, the run did not finalize
    ``failed``, the payload is an extraction run for exactly ``fixture_set``,
    its ``per_fixture`` (when present) is a list rather than some other JSON
    value, and no loaded fixture's ``content_hash()`` drifted from the run-recorded
    ``fixture_content_hash`` (all drifted fixtures reported in one message —
    a fixture with *no* recorded hash is not drift; it degrades to unrated in
    the loop, so a pre-change run never fails here).
    """
    if run_dir is None:
        raise ValueError(
            f"no extraction run to align against; {_run_fix_hint(fixture_set)}"
        )
    results_path = run_dir / "results.json"
    if not results_path.is_file():
        raise ValueError(
            f"no usable extraction run at {run_dir}: results.json is missing; "
            f"{_run_fix_hint(fixture_set)}"
        )
    try:
        doc = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed results.json in {run_dir}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"malformed results.json in {run_dir}: expected a JSON object")
    if doc.get("status") == "failed":
        raise ValueError(
            f"the run at {run_dir} finalized as failed "
            f"(error: {doc.get('error', 'unknown')}); {_run_fix_hint(fixture_set)}"
        )
    results = doc.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"malformed results.json in {run_dir}: no results payload")
    # Retrieval payloads carry an explicit report_type and no fixture_set;
    # extraction payloads omit report_type entirely. Check the explicit
    # discriminator first, the absent fixture_set as the backstop.
    if results.get("report_type") == "retrieval" or results.get("fixture_set") is None:
        raise ValueError(
            f"the run at {run_dir} is not an extraction run; "
            f"{_run_fix_hint(fixture_set)}"
        )
    if results.get("fixture_set") != fixture_set:
        raise ValueError(
            f"the run at {run_dir} evaluated fixture set "
            f"{results.get('fixture_set')!r}, not {fixture_set!r}; "
            f"{_run_fix_hint(fixture_set)}"
        )
    # Shape, not just syntax: `"per_fixture": null` (or a scalar) is valid JSON
    # that the drift scan would iterate into a TypeError — a traceback where the
    # contract promises exit 2. Refusing beats degrading here: the alternative
    # would write `unrated` records into the committed `judge_alignment/` dir
    # and an `agreement` section into a run whose payload we cannot read.
    if "per_fixture" in results and not isinstance(results["per_fixture"], list):
        raise ValueError(
            f"malformed results.json in {run_dir}: per_fixture is not a list; "
            f"{_run_fix_hint(fixture_set)}"
        )
    per_fixture = _per_fixture_index(results)
    drifted = [
        fixture.name
        for fixture in fixtures
        if (recorded := (per_fixture.get(fixture.name) or {}).get("fixture_content_hash"))
        is not None
        and recorded != fixture.content_hash()
    ]
    if drifted:
        raise ValueError(
            f"fixtures {', '.join(sorted(drifted))} changed since the run at "
            f"{run_dir}; re-run extraction before aligning"
        )
    return results


def _reusable_human_rating(
    existing: JudgeAlignmentRecord | None, judge_version: str, current_artifact_hash: str
) -> tuple[HumanRating, str] | None:
    """A stored human rating, reused only for the same judge version AND artifact.

    A human rating is a statement about one specific displayed artifact
    (DECISIONS #5): a record whose ``artifact_hash`` is missing (pre-20.1) or
    differs (the run was re-extracted) re-prompts.
    """
    if (
        existing is not None
        and existing.human_rating in ("pass", "fail")
        and existing.run_metadata.get("judge_version") == judge_version
        and existing.run_metadata.get("artifact_hash") == current_artifact_hash
    ):
        return (
            cast(HumanRating, existing.human_rating),
            str(existing.run_metadata.get("human_critique", "")),
        )
    return None


async def run_judge_alignment(
    judge: str,
    fixture_set: str,
    *,
    llm_provider: LLMProvider,
    prompt_human: PromptHuman | None = None,
    report_path: Path | None = None,
    root: Path | None = None,
    judge_cache_root: Path | None = None,
) -> AlignmentReport:
    """Align human and judge ratings over a run's persisted artifacts.

    ``report_path`` names the extraction run whose persisted artifacts are
    rated — it is required in effect: ``load_alignment_run`` raises for
    ``None`` or any unusable run (DECISIONS #9). ``llm_provider`` is injected,
    never constructed (only the CLI builds a real one), and is consulted only
    on a judge-cache miss — alignment performs **zero** extraction calls.
    ``root`` covers fixtures, judge prompts, and alignment records;
    ``judge_cache_root`` the 15.2 cache — both injectable so tests stay in
    ``tmp_path``. The agreement section is merged into the run's
    ``results.json``/``summary.md``.
    """
    ask_human = _default_prompt_human if prompt_human is None else prompt_human
    judge_runner = load_judge(judge, llm_provider, root=root)
    dimension = _judge_dimension(load_judge_prompt(judge, root=root).text, judge_runner.name)
    cache = JudgeCache(root=judge_cache_root)
    fixtures = load_recipe_fixtures(fixture_set, root=root)
    if not fixtures:
        # Same false-green as the driver: an absent/mistyped set would silently
        # write an empty `agreement` section (rate n/a) over the run's real one.
        raise ValueError(
            f"recipe fixture set {fixture_set!r} is empty or does not exist; "
            f"nothing to align"
        )

    results = load_alignment_run(report_path, fixture_set, fixtures)
    assert report_path is not None  # load_alignment_run raised otherwise
    # The envelope was just validated; metadata identifies the artifact's
    # producer (the *extraction* provider/model, not the judge's — DECISIONS #6).
    envelope = json.loads((report_path / "results.json").read_text(encoding="utf-8"))
    run_meta = envelope.get("metadata") or {}
    extraction_provider = str(run_meta.get("llm_provider", "unknown"))
    extraction_model = str(run_meta.get("llm_model", "unknown"))
    prompt_version = results.get("extraction_prompt_version")
    per_fixture = _per_fixture_index(results)

    records: list[JudgeAlignmentRecord] = []
    disagreements: list[AlignmentDisagreement] = []
    unrated: list[str] = []
    agreements = 0
    rated = 0

    def base_metadata() -> dict[str, Any]:
        return {
            "judge_name": judge_runner.name,
            "judge_version": judge_runner.version,
            "model": judge_runner.model,
            "fixture_set": fixture_set,
            "judge_dimension": dimension,
            "extraction_provider": extraction_provider,
            "extraction_model": extraction_model,
            "aligned_run": str(report_path),
            "rated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    for fixture in fixtures:
        composite_id = _composite_id(
            fixture_set, fixture.name, judge_runner.name, extraction_model
        )
        existing = load_judge_alignment(judge_runner.name, composite_id, root=root)
        entry = per_fixture.get(fixture.name) or {}
        recipes_payload = entry.get("recipes")
        content_hash = entry.get("fixture_content_hash")

        if not recipes_payload or content_hash is None or prompt_version is None:
            # No usable persisted artifact (extraction failed, hard validation
            # failed, missing from the run, or a pre-20.1 run without the
            # provenance keys): unrated, and the human is never prompted —
            # a source-only human rating is exactly the divergence this phase
            # removes (DECISIONS #4).
            unrated.append(fixture.name)
            if existing is not None and existing.human_rating in ("pass", "fail"):
                # Never destroy a collected human rating, and never assert one
                # about a vanished artifact: leave the record untouched.
                continue
            reason = (
                "run predates persisted artifacts"
                if recipes_payload
                else f"no persisted artifact for {fixture.name!r} in this run"
            )
            record = JudgeAlignmentRecord(
                fixture_id=composite_id,
                human_rating=None,
                judge_rating=None,
                agreement_status="unrated",
                run_metadata={
                    **base_metadata(),
                    "fixture_name": fixture.name,
                    "unrated_reason": reason,
                },
            )
            save_judge_alignment(record, root=root)
            records.append(record)
            continue

        artifact = serialize_extracted_artifact(recipes_payload)
        current_hash = artifact_hash(artifact)
        reused = _reusable_human_rating(existing, judge_runner.version, current_hash)
        human_rating, human_critique = (
            reused
            if reused is not None
            else ask_human(fixture, artifact, judge_runner.name, dimension)
        )
        if human_rating not in ("pass", "fail"):
            raise ValueError(
                f"human rating must be 'pass' or 'fail', got {human_rating!r} "
                f"for fixture {fixture.name!r}"
            )

        # Cache key from the run's recorded provenance, not the working tree
        # (DECISIONS #2): the entry's content hash and the run's prompt version.
        key = build_judge_cache_key(
            fixture_set=fixture_set,
            fixture_id=fixture.name,
            fixture_content_hash=str(content_hash),
            extraction_prompt_version=str(prompt_version),
            artifact_hash=current_hash,
            judge=judge_runner,
        )
        judge_rating: JudgeRating | None = cache.get(key)
        if judge_rating is None:
            try:
                judge_rating = await judge_runner.judge(
                    artifact,
                    json.dumps(fixture.expected, indent=2),
                    fixture.source_md,
                )
            except JudgeError:
                judge_rating = None
            else:
                # Alignment is a first-class cache producer: stamp the artifact
                # hash so its files are as self-describing as the driver's.
                judge_rating.metadata["artifact_hash"] = current_hash
                cache.put(judge_rating, key=key)

        if judge_rating is None:
            agreement_status = "unrated"
            unrated.append(fixture.name)
        else:
            rated += 1
            if judge_rating.rating == human_rating:
                agreement_status = "agree"
                agreements += 1
            else:
                agreement_status = "disagree"
                disagreements.append(
                    AlignmentDisagreement(
                        fixture_name=fixture.name,
                        human_rating=human_rating,
                        judge_rating=judge_rating.rating,
                        human_critique=human_critique,
                        judge_critique=judge_rating.critique,
                    )
                )

        record = JudgeAlignmentRecord(
            fixture_id=composite_id,
            human_rating=human_rating,
            judge_rating=judge_rating.rating if judge_rating is not None else None,
            agreement_status=agreement_status,
            run_metadata={
                **base_metadata(),
                "fixture_name": fixture.name,
                "human_critique": human_critique,
                "judge_critique": judge_rating.critique if judge_rating is not None else None,
                "artifact_hash": current_hash,
            },
        )
        save_judge_alignment(record, root=root)
        records.append(record)

    report = AlignmentReport(
        judge_name=judge_runner.name,
        judge_version=judge_runner.version,
        model=judge_runner.model,
        agreement_rate=agreements / rated if rated else None,
        rated=rated,
        agreements=agreements,
        disagreements=disagreements,
        unrated=unrated,
        records=records,
    )

    payload = report.agreement_payload()

    def _merge(results_payload: dict[str, object]) -> None:
        results_payload["agreement"] = payload

    _update_run_results(report_path, _merge)
    rate = report.agreement_rate
    rendered = "n/a" if rate is None else f"{rate:.2f}"
    _append_run_summary(
        report_path,
        f"Judge-human agreement ({report.judge_name}, {report.judge_version}): {rendered}",
    )

    return report
