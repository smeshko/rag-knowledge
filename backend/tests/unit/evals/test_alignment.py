"""Offline tests for the judge-alignment workflow (Epic 15 Phase 15.3).

No stdin, no TTY, no API key, no ``live`` marker: the human prompt is a
scripted callable, the judge side is served from a seeded 15.2 cache (or a
``FakeLLMProvider`` for the cache-miss path), and every root points at
``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from evals.alignment import run_judge_alignment
from evals.extraction import run_extraction_eval
from evals.fixtures import load_judge_alignment
from evals.judge_cache import JudgeCache
from evals.judges import JUDGE_SCHEMA_VERSION, JudgeRating
from evals.models import RecipeFixture

from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputResponse, TokenUsage
from tests.unit.evals.eval_utils import (
    SOUP_EXPECTED,
    SOUP_SOURCE,
    STEW_EXPECTED,
    STEW_SOURCE,
    THRESHOLDS,
    SettingsStandIn,
    request_hash,
    stew_output,
    write_fixture,
    write_judge_prompt,
    write_smoke_set,
)


class _ScriptedPrompt:
    """A scripted ``prompt_human``: canned (rating, critique) per fixture name."""

    def __init__(self, answers: dict[str, tuple[str, str]]) -> None:
        self._answers = answers
        self.asked: list[str] = []

    def __call__(self, fixture: RecipeFixture) -> tuple[Any, str]:
        self.asked.append(fixture.name)
        return self._answers[fixture.name]


def _rating(fixture: str, rating: str, judge: str = "summary_quality") -> JudgeRating:
    return JudgeRating(
        judge_name=judge,
        judge_version="v1",
        rating=rating,  # type: ignore[arg-type]
        critique=f"judge critique for {fixture}",
        metadata={"provider": "fake", "model": "fake-model"},
    )


def _seed(tmp_path: Path, *, judges: tuple[str, ...] = ("summary_quality",)) -> Path:
    """Fixture root with the smoke pair and one prompt per judge."""
    root = tmp_path / "fixtures"
    write_fixture(root, "smoke", "bean-stew", STEW_SOURCE, STEW_EXPECTED)
    write_fixture(root, "smoke", "tomato-soup", SOUP_SOURCE, SOUP_EXPECTED)
    for judge in judges:
        write_judge_prompt(root, judge)
    return root


def _seed_cache(cache_root: Path, ratings: dict[str, str], judge: str = "summary_quality") -> None:
    cache = JudgeCache(root=cache_root)
    for fixture, rating in ratings.items():
        cache.put(_rating(fixture, rating, judge), fixture_id=fixture)


async def test_agreement_math_and_disagreements_with_both_critiques(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    cache_root = tmp_path / "cache"
    _seed_cache(cache_root, {"bean-stew": "pass", "tomato-soup": "fail"})
    prompt = _ScriptedPrompt(
        {
            "bean-stew": ("pass", "human: looks right"),
            "tomato-soup": ("pass", "human: soup is fine"),
        }
    )
    provider = FakeLLMProvider()  # must never be consulted: both ratings cached

    report = await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=provider,
        prompt_human=prompt,
        root=root,
        judge_cache_root=cache_root,
    )

    assert provider.calls == ()
    assert report.rated == 2
    assert report.agreements == 1
    assert report.agreement_rate == pytest.approx(0.5)
    assert report.unrated == []
    (disagreement,) = report.disagreements
    assert disagreement.fixture_name == "tomato-soup"
    assert disagreement.human_rating == "pass"
    assert disagreement.judge_rating == "fail"
    assert disagreement.human_critique == "human: soup is fine"
    assert disagreement.judge_critique == "judge critique for tomato-soup"


async def test_records_persist_with_composite_id_and_critiques_in_run_metadata(
    tmp_path: Path,
) -> None:
    root = _seed(tmp_path)
    cache_root = tmp_path / "cache"
    _seed_cache(cache_root, {"bean-stew": "pass", "tomato-soup": "fail"})
    prompt = _ScriptedPrompt(
        {
            "bean-stew": ("pass", "human: looks right"),
            "tomato-soup": ("fail", "human: steps missing"),
        }
    )
    await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=FakeLLMProvider(),
        prompt_human=prompt,
        root=root,
        judge_cache_root=cache_root,
    )
    record = load_judge_alignment("summary_quality", "bean-stew__summary_quality", root=root)
    assert record is not None
    assert record.fixture_id == "bean-stew__summary_quality"
    assert record.human_rating == "pass"
    assert record.judge_rating == "pass"
    assert record.agreement_status == "agree"
    assert record.run_metadata["judge_name"] == "summary_quality"
    assert record.run_metadata["judge_version"] == "v1"
    assert record.run_metadata["model"] == "fake-model"
    assert record.run_metadata["fixture_name"] == "bean-stew"
    assert record.run_metadata["human_critique"] == "human: looks right"
    assert record.run_metadata["judge_critique"] == "judge critique for bean-stew"
    assert "rated_at" in record.run_metadata


async def test_two_judges_on_one_fixture_keep_both_records(tmp_path: Path) -> None:
    # Epic 14's save_judge_alignment writes by fixture_id alone, so without the
    # composite id the second judge would clobber the first (DECISIONS #6).
    root = _seed(tmp_path, judges=("summary_quality", "boundary_correctness"))
    cache_root = tmp_path / "cache"
    _seed_cache(cache_root, {"bean-stew": "pass", "tomato-soup": "pass"}, "summary_quality")
    _seed_cache(cache_root, {"bean-stew": "fail", "tomato-soup": "fail"}, "boundary_correctness")
    answers = {"bean-stew": ("pass", "ok"), "tomato-soup": ("pass", "ok")}
    for judge in ("summary_quality", "boundary_correctness"):
        await run_judge_alignment(
            judge,
            "smoke",
            llm_provider=FakeLLMProvider(),
            prompt_human=_ScriptedPrompt(dict(answers)),
            root=root,
            judge_cache_root=cache_root,
        )
    summary = load_judge_alignment("summary_quality", "bean-stew__summary_quality", root=root)
    boundary = load_judge_alignment(
        "boundary_correctness", "bean-stew__boundary_correctness", root=root
    )
    assert summary is not None and summary.judge_rating == "pass"
    assert boundary is not None and boundary.judge_rating == "fail"


async def test_rerun_reuses_human_ratings_and_version_bump_reprompts(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    cache_root = tmp_path / "cache"
    _seed_cache(cache_root, {"bean-stew": "pass", "tomato-soup": "pass"})
    answers = {"bean-stew": ("pass", "ok"), "tomato-soup": ("fail", "meh")}
    kwargs: dict[str, Any] = {
        "llm_provider": FakeLLMProvider(),
        "root": root,
        "judge_cache_root": cache_root,
    }

    first_prompt = _ScriptedPrompt(dict(answers))
    await run_judge_alignment(
        "summary_quality", "smoke", prompt_human=first_prompt, **kwargs
    )
    assert sorted(first_prompt.asked) == ["bean-stew", "tomato-soup"]

    # Re-run, same judge version: stored human ratings are reused, no re-prompt.
    second_prompt = _ScriptedPrompt(dict(answers))
    report = await run_judge_alignment(
        "summary_quality", "smoke", prompt_human=second_prompt, **kwargs
    )
    assert second_prompt.asked == []
    by_id = {record.fixture_id: record for record in report.records}
    assert by_id["tomato-soup__summary_quality"].human_rating == "fail"

    # Bump the prompt version: the stored ratings are stale, so it re-prompts.
    write_judge_prompt_v2 = (
        "# Summary quality judge\n# version: v2\n\nRate harder.\n\n"
        "{extracted_output}\n{expected_output}\n{source_text}\n"
    )
    (root / "judge_prompts" / "summary_quality.md").write_text(
        write_judge_prompt_v2, encoding="utf-8"
    )
    _seed_cache(cache_root, {"bean-stew": "pass", "tomato-soup": "pass"})
    cache = JudgeCache(root=cache_root)
    for fixture in ("bean-stew", "tomato-soup"):
        v2 = _rating(fixture, "pass").model_copy(update={"judge_version": "v2"})
        cache.put(v2, fixture_id=fixture)
    third_prompt = _ScriptedPrompt(dict(answers))
    await run_judge_alignment(
        "summary_quality", "smoke", prompt_human=third_prompt, **kwargs
    )
    assert sorted(third_prompt.asked) == ["bean-stew", "tomato-soup"]


async def test_unrated_fixture_is_excluded_from_agreement_and_persisted(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    cache_root = tmp_path / "cache"
    _seed_cache(cache_root, {"bean-stew": "pass"})  # tomato-soup: cache miss
    # On the miss the provider rejects the extraction call, so the judge never
    # gets anything to rate → unrated, not a crash and not a fail.
    provider = FakeLLMProvider(
        default_output=StructuredOutputResponse(
            output_json=None,
            parse_error="rejected",
            raw_text="",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            provider="fake",
            model="fake-model",
        )
    )
    prompt = _ScriptedPrompt(
        {"bean-stew": ("pass", "ok"), "tomato-soup": ("pass", "fine")}
    )
    report = await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=provider,
        prompt_human=prompt,
        root=root,
        judge_cache_root=cache_root,
    )
    assert report.rated == 1
    assert report.agreement_rate == pytest.approx(1.0)
    assert report.unrated == ["tomato-soup"]
    record = load_judge_alignment("summary_quality", "tomato-soup__summary_quality", root=root)
    assert record is not None
    assert record.human_rating == "pass"
    assert record.judge_rating is None
    assert record.agreement_status == "unrated"


async def test_cache_miss_extracts_judges_and_caches(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    cache_root = tmp_path / "cache"
    write_fixture_root_only = root  # single fixture set: drop the soup fixture
    (root / "synthetic_recipes" / "smoke" / "tomato-soup" / "source.md").unlink()
    (root / "synthetic_recipes" / "smoke" / "tomato-soup" / "expected.json").unlink()
    (root / "synthetic_recipes" / "smoke" / "tomato-soup").rmdir()
    provider = FakeLLMProvider(
        {request_hash("bean-stew", STEW_SOURCE): stew_output()},
        default_output={"rating": "pass", "critique": "judged live"},
    )
    report = await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=provider,
        prompt_human=_ScriptedPrompt({"bean-stew": ("pass", "ok")}),
        root=write_fixture_root_only,
        judge_cache_root=cache_root,
    )
    assert report.rated == 1
    assert report.agreement_rate == pytest.approx(1.0)
    judge_calls = [c for c in provider.calls if c.schema_version == JUDGE_SCHEMA_VERSION]
    assert len(judge_calls) == 1
    cached = JudgeCache(root=cache_root).get("bean-stew", "summary_quality", "v1", "fake-model")
    assert cached is not None
    assert cached.critique == "judged live"


async def test_empty_fixture_set_raises_instead_of_writing_an_empty_agreement(
    tmp_path: Path,
) -> None:
    root = _seed(tmp_path)
    with pytest.raises(ValueError, match="empty or does not exist"):
        await run_judge_alignment(
            "summary_quality",
            "typo",
            llm_provider=FakeLLMProvider(),
            prompt_human=_ScriptedPrompt({}),
            root=root,
            judge_cache_root=tmp_path / "cache",
        )


async def test_agreement_section_merges_into_existing_run_results(tmp_path: Path) -> None:
    fixtures_root = tmp_path / "fixtures"
    provider = write_smoke_set(
        fixtures_root, judge_output={"rating": "pass", "critique": "OK."}
    )
    write_judge_prompt(fixtures_root)
    run = await run_extraction_eval(
        "smoke",
        "aligned-eval",
        llm_provider=provider,
        judge="summary_quality",
        fixtures_root=fixtures_root,
        reports_root=tmp_path / "reports",
        thresholds=THRESHOLDS,
        settings=SettingsStandIn(),
        judge_cache_root=tmp_path / "cache",
    )
    before = json.loads((run.path / "results.json").read_text(encoding="utf-8"))

    report = await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=provider,
        prompt_human=_ScriptedPrompt(
            {"bean-stew": ("pass", "ok"), "tomato-soup": ("fail", "steps missing")}
        ),
        report_path=run.path,
        root=fixtures_root,
        judge_cache_root=tmp_path / "cache",
    )
    assert report.agreement_rate == pytest.approx(0.5)

    after = json.loads((run.path / "results.json").read_text(encoding="utf-8"))
    # metadata and the 15.1/15.2 sections are untouched; agreement is filled.
    assert after["metadata"] == before["metadata"]
    assert after["status"] == before["status"]
    assert after["results"]["per_fixture"] == before["results"]["per_fixture"]
    assert after["results"]["aggregate"] == before["results"]["aggregate"]
    assert after["results"]["judge"] == before["results"]["judge"]
    agreement = after["results"]["agreement"]
    assert agreement["judge_name"] == "summary_quality"
    assert agreement["agreement_rate"] == pytest.approx(0.5)
    assert agreement["rated"] == 2
    (disagreement,) = agreement["disagreements"]
    assert disagreement["human_critique"] == "steps missing"
    assert disagreement["judge_critique"] == "OK."

    summary = (run.path / "summary.md").read_text(encoding="utf-8")
    assert "Judge-human agreement (summary_quality, v1): 0.50" in summary
