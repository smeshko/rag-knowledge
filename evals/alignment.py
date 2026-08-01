"""Judge-alignment workflow: human-vs-judge agreement (Epic 15 Phase 15.3).

Implements the doc-12 § 5 alignment loop: a human rates each fixture's
extraction pass/fail with a critique, the LLM judge rates the same fixtures
(replayed from the Phase-15.2 cache — the injected provider is only consulted
on a cache miss, in which case the fixture's extraction is re-driven through
the synthetic-window path to give the judge something to rate), agreement is
computed over the fixtures both sides rated, and disagreements are listed with
**both** critiques for prompt iteration.

Epic-14 model gaps absorbed here (DECISIONS #6): ``save_judge_alignment``
writes by ``record.fixture_id`` alone and ``load_judge_alignment`` ignores its
``name`` argument, so records are namespaced with the composite id
``<fixture_name>__<judge_name>`` — two judges aligned on one fixture never
overwrite each other. ``JudgeAlignmentRecord`` has no critique fields, so both
critiques (plus judge version/model/fixture name/timestamp) live in
``run_metadata`` under the documented keys ``{"judge_name", "judge_version",
"model", "fixture_name", "human_critique", "judge_critique", "rated_at"}``.

Human ratings are version-keyed (DECISIONS #2): a stored rating is reused only
while the judge version matches; a version bump re-prompts and overwrites the
record (prior-version ratings are not retained). The interactive prompt is an
injected callable (DECISIONS #3) — tests script it; only the CLI uses the real
``typer.prompt``. The agreement metric is written into the *existing*
extraction run's ``results.json`` at ``results.agreement`` via the
read-modify-write helper (DECISIONS #7), never via a new ``ReportRun``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import typer
from pydantic import BaseModel

from evals.extraction import (
    _build_synthetic_window,
    _extract_recipes,
    artifact_hash,
    build_judge_cache_key,
    extraction_prompt_version,
)
from evals.fixtures import load_judge_alignment, load_recipe_fixtures, save_judge_alignment
from evals.judge_cache import JudgeCache
from evals.judges import Judge, JudgeError, JudgeRating, load_judge
from evals.models import JudgeAlignmentRecord, RecipeFixture
from evals.reports import _append_run_summary, _update_run_results
from rag_recipes.providers.llm.base import LLMProvider

__all__ = [
    "AlignmentDisagreement",
    "AlignmentReport",
    "PromptHuman",
    "run_judge_alignment",
]

HumanRating = Literal["pass", "fail"]

PromptHuman = Callable[[RecipeFixture], tuple[HumanRating, str]]
"""Collect a human ``("pass"|"fail", critique)`` for one fixture.

The rating shares ``JudgeRating.rating``'s domain — anything else would make
the human-vs-judge comparison silently never match.
"""


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


def _default_prompt_human(fixture: RecipeFixture) -> tuple[HumanRating, str]:
    """Interactive typer prompt — the CLI path; tests inject a scripted fake."""
    typer.echo(f"\n=== {fixture.name} ===")
    typer.echo(fixture.source_md)
    while True:
        raw = typer.prompt("Rating (pass/fail)").strip().lower()
        if raw in ("pass", "fail"):
            break
        typer.echo("Please answer 'pass' or 'fail'.")
    critique = typer.prompt("Critique")
    return cast(HumanRating, raw), critique


def _composite_id(fixture_name: str, judge_name: str) -> str:
    """Judge-scoped alignment-record id (DECISIONS #6 — no overwrite across judges)."""
    return f"{fixture_name}__{judge_name}"


async def _judge_rating_for(
    judge: Judge,
    cache: JudgeCache,
    fixture: RecipeFixture,
    fixture_set: str,
    llm_provider: LLMProvider,
) -> JudgeRating | None:
    """The judge's rating for a fixture: cache replay, LLM only on a miss.

    Interim Epic 20.1 state (rewritten in the alignment task): this still
    re-drives the fixture's extraction through the synthetic-window path and
    judges (and caches) ``recipes[0]`` only — the extraction now runs *before*
    the cache lookup because the judged artifact's hash is a key part
    (DECISIONS #10). The prompt version comes from the
    ``extraction_prompt_version()`` accessor, never a second import of the
    constant (DECISIONS #2). Extraction failure or ``JudgeError`` yields
    ``None`` — the fixture is *unrated*, never a silent pass/fail.
    """
    window = _build_synthetic_window(fixture.name, fixture.source_md)
    recipes, _error = await _extract_recipes(window, fixture.name, llm_provider)
    if not recipes:
        return None
    extracted = json.dumps(recipes[0].model_dump(mode="json", by_alias=True), indent=2)
    key = build_judge_cache_key(
        fixture_set=fixture_set,
        fixture_id=fixture.name,
        fixture_content_hash=fixture.content_hash(),
        extraction_prompt_version=extraction_prompt_version(),
        artifact_hash=artifact_hash(extracted),
        judge=judge,
    )
    rating = cache.get(key)
    if rating is not None:
        return rating
    try:
        rating = await judge.judge(
            extracted,
            json.dumps(fixture.expected, indent=2),
            fixture.source_md,
        )
    except JudgeError:
        return None
    cache.put(rating, key=key)
    return rating


def _reusable_human_rating(
    existing: JudgeAlignmentRecord | None, judge_version: str
) -> tuple[HumanRating, str] | None:
    """A stored human rating, reused only while the judge version matches."""
    if (
        existing is not None
        and existing.human_rating in ("pass", "fail")
        and existing.run_metadata.get("judge_version") == judge_version
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
    """Walk a fixture set collecting human ratings and computing judge agreement.

    ``llm_provider`` is injected, never constructed (only the CLI builds a real
    one). ``root`` covers fixtures, judge prompts, and alignment records;
    ``judge_cache_root`` the 15.2 cache — both injectable so tests stay in
    ``tmp_path``. When ``report_path`` names an extraction run directory, the
    agreement section is merged into its ``results.json``/``summary.md``.
    """
    ask_human = _default_prompt_human if prompt_human is None else prompt_human
    judge_runner = load_judge(judge, llm_provider, root=root)
    cache = JudgeCache(root=judge_cache_root)
    fixtures = load_recipe_fixtures(fixture_set, root=root)
    if not fixtures:
        # Same false-green as the driver: an absent/mistyped set would silently
        # write an empty `agreement` section (rate n/a) over the run's real one.
        raise ValueError(
            f"recipe fixture set {fixture_set!r} is empty or does not exist; "
            f"nothing to align"
        )

    records: list[JudgeAlignmentRecord] = []
    disagreements: list[AlignmentDisagreement] = []
    unrated: list[str] = []
    agreements = 0
    rated = 0

    for fixture in fixtures:
        composite_id = _composite_id(fixture.name, judge_runner.name)
        existing = load_judge_alignment(judge_runner.name, composite_id, root=root)
        reused = _reusable_human_rating(existing, judge_runner.version)
        human_rating, human_critique = reused if reused is not None else ask_human(fixture)
        if human_rating not in ("pass", "fail"):
            raise ValueError(
                f"human rating must be 'pass' or 'fail', got {human_rating!r} "
                f"for fixture {fixture.name!r}"
            )

        judge_rating = await _judge_rating_for(
            judge_runner, cache, fixture, fixture_set, llm_provider
        )
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
                "judge_name": judge_runner.name,
                "judge_version": judge_runner.version,
                "model": judge_runner.model,
                "fixture_name": fixture.name,
                "human_critique": human_critique,
                "judge_critique": judge_rating.critique if judge_rating is not None else None,
                "rated_at": datetime.now(UTC).isoformat(timespec="seconds"),
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

    if report_path is not None:
        payload = report.agreement_payload()

        def _merge(results: dict[str, object]) -> None:
            results["agreement"] = payload

        _update_run_results(report_path, _merge)
        rate = report.agreement_rate
        rendered = "n/a" if rate is None else f"{rate:.2f}"
        _append_run_summary(
            report_path,
            f"Judge-human agreement ({report.judge_name}, {report.judge_version}): {rendered}",
        )

    return report
