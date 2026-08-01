"""Offline tests for the judge-alignment workflow (Epic 15.3, Epic 20.1).

No stdin, no TTY, no API key, no ``live`` marker: a prior extraction run is
seeded hermetically via ``run_extraction_eval`` with a ``FakeLLMProvider``,
alignment is then driven against that run with a *judge-only* provider (no
extraction responses at all — alignment never extracts), the human prompt is a
scripted callable, and every root points at ``tmp_path``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from evals.alignment import load_alignment_run, run_judge_alignment
from evals.extraction import (
    artifact_hash,
    run_extraction_eval,
    serialize_extracted_artifact,
)
from evals.fixtures import load_judge_alignment, save_judge_alignment
from evals.judge_cache import JudgeCache, JudgeCacheKey
from evals.judges import JUDGE_SCHEMA_VERSION
from evals.models import JudgeAlignmentRecord, RecipeFixture

from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputResponse, TokenUsage
from tests.unit.evals.eval_utils import (
    DIMENSION_JUDGE_PROMPT,
    DIMENSION_SENTENCE,
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

STEW_ID = "smoke__bean-stew__summary_quality__gpt-4.1"
SOUP_ID = "smoke__tomato-soup__summary_quality__gpt-4.1"


class _ScriptedPrompt:
    """A scripted ``prompt_human`` recording everything it was shown."""

    def __init__(self, answers: dict[str, tuple[str, str]]) -> None:
        self._answers = answers
        self.asked: list[str] = []
        self.artifacts: dict[str, str] = {}
        self.judge_names: dict[str, str] = {}
        self.dimensions: dict[str, str] = {}

    def __call__(
        self, fixture: RecipeFixture, artifact: str, judge_name: str, dimension: str
    ) -> tuple[Any, str]:
        self.asked.append(fixture.name)
        self.artifacts[fixture.name] = artifact
        self.judge_names[fixture.name] = judge_name
        self.dimensions[fixture.name] = dimension
        return self._answers[fixture.name]


def _judge_provider(rating: str = "pass", critique: str = "judge ok") -> FakeLLMProvider:
    """A provider with NO extraction responses — it can only serve judge calls."""
    return FakeLLMProvider(default_output={"rating": rating, "critique": critique})


def _judge_calls(provider: FakeLLMProvider) -> list[Any]:
    return [c for c in provider.calls if c.schema_version == JUDGE_SCHEMA_VERSION]


async def _seed_run(
    fixtures_root: Path,
    reports_root: Path,
    provider: FakeLLMProvider,
    *,
    fixture_set: str = "smoke",
    label: str = "seed",
    settings: SettingsStandIn | None = None,
    judge: str | None = None,
    judge_cache_root: Path | None = None,
) -> Path:
    """Produce a real extraction run dir for alignment to rate."""
    run = await run_extraction_eval(
        fixture_set,
        label,
        llm_provider=provider,
        judge=judge,
        fixtures_root=fixtures_root,
        reports_root=reports_root,
        thresholds=THRESHOLDS,
        settings=settings or SettingsStandIn(),
        judge_cache_root=judge_cache_root,
    )
    return run.path


def _results(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "results.json").read_text(encoding="utf-8"))["results"]


def _persisted_artifact(run_dir: Path, fixture_name: str) -> str:
    results = _results(run_dir)
    entry = next(e for e in results["per_fixture"] if e["name"] == fixture_name)
    return serialize_extracted_artifact(entry["recipes"])


def _run_key(
    run_dir: Path, fixture_name: str, judge: str = "summary_quality", model: str = "fake-model"
) -> JudgeCacheKey:
    """The cache key alignment builds from the run's recorded provenance."""
    results = _results(run_dir)
    entry = next(e for e in results["per_fixture"] if e["name"] == fixture_name)
    return JudgeCacheKey(
        fixture_set=results["fixture_set"],
        fixture_id=fixture_name,
        fixture_content_hash=entry["fixture_content_hash"],
        extraction_prompt_version=results["extraction_prompt_version"],
        artifact_hash=artifact_hash(serialize_extracted_artifact(entry["recipes"])),
        judge_name=judge,
        judge_version="v1",
        model=model,
    )


_ANSWERS = {"bean-stew": ("pass", "human: looks right"), "tomato-soup": ("pass", "human: fine")}


async def test_alignment_judges_persisted_artifact_with_zero_extraction_calls(
    tmp_path: Path,
) -> None:
    fixtures_root = tmp_path / "fixtures"
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", write_smoke_set(fixtures_root))
    write_judge_prompt(fixtures_root)
    provider = _judge_provider()

    report = await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=provider,
        prompt_human=_ScriptedPrompt(dict(_ANSWERS)),
        report_path=run_dir,
        root=fixtures_root,
        judge_cache_root=tmp_path / "cache",
    )

    # Zero extraction calls: the provider has no extraction responses, and
    # every call it did receive was a judge call.
    assert len(provider.calls) == 2
    assert all(c.schema_version == JUDGE_SCHEMA_VERSION for c in provider.calls)
    assert report.rated == 2
    assert report.unrated == []
    # The miss was cached under the run-provenance key with the artifact hash
    # stamped into the rating's metadata (self-describing cache file).
    key = _run_key(run_dir, "bean-stew")
    cached = JudgeCache(root=tmp_path / "cache").get(key)
    assert cached is not None
    assert cached.metadata["artifact_hash"] == key.artifact_hash


async def test_byte_identity_between_human_judge_and_persisted_artifact(
    tmp_path: Path,
) -> None:
    fixtures_root = tmp_path / "fixtures"
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", write_smoke_set(fixtures_root))
    write_judge_prompt(fixtures_root)
    provider = _judge_provider()
    prompt = _ScriptedPrompt(dict(_ANSWERS))

    await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=provider,
        prompt_human=prompt,
        report_path=run_dir,
        root=fixtures_root,
        judge_cache_root=tmp_path / "cache",
    )

    for name in ("bean-stew", "tomato-soup"):
        persisted = _persisted_artifact(run_dir, name)
        # The human saw exactly the persisted serialization…
        assert prompt.artifacts[name] == persisted
        # …and the same string is embedded verbatim in a judge request.
        assert any(persisted in call.input for call in _judge_calls(provider))


async def test_second_alignment_run_replays_cache_and_human_with_zero_llm_calls(
    tmp_path: Path,
) -> None:
    fixtures_root = tmp_path / "fixtures"
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", write_smoke_set(fixtures_root))
    write_judge_prompt(fixtures_root)
    kwargs: dict[str, Any] = {
        "report_path": run_dir,
        "root": fixtures_root,
        "judge_cache_root": tmp_path / "cache",
    }
    await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=_judge_provider(),
        prompt_human=_ScriptedPrompt(dict(_ANSWERS)),
        **kwargs,
    )

    fresh_provider = FakeLLMProvider()  # no canned outputs at all
    fresh_prompt = _ScriptedPrompt(dict(_ANSWERS))
    report = await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=fresh_provider,
        prompt_human=fresh_prompt,
        **kwargs,
    )
    assert fresh_provider.calls == ()  # judge replayed from cache
    assert fresh_prompt.asked == []  # human ratings reused (version + artifact match)
    assert report.rated == 2


async def test_agreement_math_and_disagreements_with_both_critiques(tmp_path: Path) -> None:
    fixtures_root = tmp_path / "fixtures"
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", write_smoke_set(fixtures_root))
    write_judge_prompt(fixtures_root)
    cache = JudgeCache(root=tmp_path / "cache")
    for fixture, rating in (("bean-stew", "pass"), ("tomato-soup", "fail")):
        key = _run_key(run_dir, fixture)
        cache.put(
            _seed_rating(fixture, rating, artifact=key.artifact_hash),
            key=key,
        )
    provider = FakeLLMProvider()  # must never be consulted: both ratings cached
    prompt = _ScriptedPrompt(
        {
            "bean-stew": ("pass", "human: looks right"),
            "tomato-soup": ("pass", "human: soup is fine"),
        }
    )

    report = await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=provider,
        prompt_human=prompt,
        report_path=run_dir,
        root=fixtures_root,
        judge_cache_root=tmp_path / "cache",
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


def _seed_rating(fixture: str, rating: str, *, artifact: str, judge: str = "summary_quality"):
    from evals.judges import JudgeRating

    return JudgeRating(
        judge_name=judge,
        judge_version="v1",
        rating=rating,  # type: ignore[arg-type]
        critique=f"judge critique for {fixture}",
        metadata={"provider": "fake", "model": "fake-model", "artifact_hash": artifact},
    )


async def test_records_persist_with_composite_id_and_provenance_in_run_metadata(
    tmp_path: Path,
) -> None:
    fixtures_root = tmp_path / "fixtures"
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", write_smoke_set(fixtures_root))
    write_judge_prompt(fixtures_root, text=DIMENSION_JUDGE_PROMPT)

    await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=_judge_provider(critique="judge: crisp"),
        prompt_human=_ScriptedPrompt(dict(_ANSWERS)),
        report_path=run_dir,
        root=fixtures_root,
        judge_cache_root=tmp_path / "cache",
    )

    record = load_judge_alignment("summary_quality", STEW_ID, root=fixtures_root)
    assert record is not None
    assert record.fixture_id == STEW_ID
    assert record.human_rating == "pass"
    assert record.judge_rating == "pass"
    assert record.agreement_status == "agree"
    meta = record.run_metadata
    assert meta["judge_name"] == "summary_quality"
    assert meta["judge_version"] == "v1"
    assert meta["model"] == "fake-model"  # the judge's provider model
    assert meta["fixture_name"] == "bean-stew"
    assert meta["fixture_set"] == "smoke"
    assert meta["human_critique"] == "human: looks right"
    assert meta["judge_critique"] == "judge: crisp"
    assert meta["artifact_hash"] == artifact_hash(_persisted_artifact(run_dir, "bean-stew"))
    assert meta["judge_dimension"] == DIMENSION_SENTENCE
    assert meta["extraction_provider"] == "openai"
    assert meta["extraction_model"] == "gpt-4.1"  # the run's metadata.llm_model
    assert meta["aligned_run"] == str(run_dir)
    assert "rated_at" in meta


async def test_two_judges_on_one_fixture_keep_both_records(tmp_path: Path) -> None:
    fixtures_root = tmp_path / "fixtures"
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", write_smoke_set(fixtures_root))
    for judge in ("summary_quality", "boundary_correctness"):
        write_judge_prompt(fixtures_root, judge)
    answers = {"bean-stew": ("pass", "ok"), "tomato-soup": ("pass", "ok")}
    for judge, rating in (("summary_quality", "pass"), ("boundary_correctness", "fail")):
        await run_judge_alignment(
            judge,
            "smoke",
            llm_provider=_judge_provider(rating=rating),
            prompt_human=_ScriptedPrompt(dict(answers)),
            report_path=run_dir,
            root=fixtures_root,
            judge_cache_root=tmp_path / "cache",
        )
    summary = load_judge_alignment("summary_quality", STEW_ID, root=fixtures_root)
    boundary = load_judge_alignment(
        "boundary_correctness",
        "smoke__bean-stew__boundary_correctness__gpt-4.1",
        root=fixtures_root,
    )
    assert summary is not None and summary.judge_rating == "pass"
    assert boundary is not None and boundary.judge_rating == "fail"


async def test_two_fixture_sets_with_same_named_fixture_keep_both_records(
    tmp_path: Path,
) -> None:
    fixtures_root = tmp_path / "fixtures"
    provider = write_smoke_set(fixtures_root)
    write_fixture(fixtures_root, "smoke2", "bean-stew", STEW_SOURCE, STEW_EXPECTED)
    write_judge_prompt(fixtures_root)
    run_a = await _seed_run(fixtures_root, tmp_path / "reports", provider, fixture_set="smoke")
    run_b = await _seed_run(fixtures_root, tmp_path / "reports", provider, fixture_set="smoke2")
    answers = {"bean-stew": ("pass", "ok"), "tomato-soup": ("pass", "ok")}
    for fixture_set, run_dir in (("smoke", run_a), ("smoke2", run_b)):
        await run_judge_alignment(
            "summary_quality",
            fixture_set,
            llm_provider=_judge_provider(),
            prompt_human=_ScriptedPrompt(dict(answers)),
            report_path=run_dir,
            root=fixtures_root,
            judge_cache_root=tmp_path / "cache",
        )
    record_a = load_judge_alignment("summary_quality", STEW_ID, root=fixtures_root)
    record_b = load_judge_alignment(
        "summary_quality", "smoke2__bean-stew__summary_quality__gpt-4.1", root=fixtures_root
    )
    assert record_a is not None and record_a.run_metadata["fixture_set"] == "smoke"
    assert record_b is not None and record_b.run_metadata["fixture_set"] == "smoke2"


async def test_two_extraction_models_keep_both_records(tmp_path: Path) -> None:
    # The id discriminates on the run's metadata.llm_model (the artifact's
    # producer) — NOT the judge provider's model, which is "fake-model" in both
    # alignments here and would collapse the two records.
    fixtures_root = tmp_path / "fixtures"
    provider = write_smoke_set(fixtures_root)
    write_judge_prompt(fixtures_root)
    other_settings = SettingsStandIn()
    other_settings.llm_model = "other-model"
    run_a = await _seed_run(fixtures_root, tmp_path / "reports", provider)
    run_b = await _seed_run(
        fixtures_root, tmp_path / "reports", provider, settings=other_settings
    )
    for run_dir in (run_a, run_b):
        await run_judge_alignment(
            "summary_quality",
            "smoke",
            llm_provider=_judge_provider(),
            prompt_human=_ScriptedPrompt(dict(_ANSWERS)),
            report_path=run_dir,
            root=fixtures_root,
            judge_cache_root=tmp_path / "cache",
        )
    record_a = load_judge_alignment("summary_quality", STEW_ID, root=fixtures_root)
    record_b = load_judge_alignment(
        "summary_quality", "smoke__bean-stew__summary_quality__other-model", root=fixtures_root
    )
    assert record_a is not None and record_a.run_metadata["extraction_model"] == "gpt-4.1"
    assert record_b is not None and record_b.run_metadata["extraction_model"] == "other-model"


async def test_dimension_falls_back_to_judge_name_without_marker(tmp_path: Path) -> None:
    # eval_utils.JUDGE_PROMPT has no "Rate exactly ONE subjective dimension:"
    # sentence, so the human is told the judge name alone.
    fixtures_root = tmp_path / "fixtures"
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", write_smoke_set(fixtures_root))
    write_judge_prompt(fixtures_root)  # the dimension-less prompt
    prompt = _ScriptedPrompt(dict(_ANSWERS))

    await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=_judge_provider(),
        prompt_human=prompt,
        report_path=run_dir,
        root=fixtures_root,
        judge_cache_root=tmp_path / "cache",
    )

    assert prompt.judge_names["bean-stew"] == "summary_quality"
    assert prompt.dimensions["bean-stew"] == "summary_quality"
    record = load_judge_alignment("summary_quality", STEW_ID, root=fixtures_root)
    assert record is not None
    assert record.run_metadata["judge_dimension"] == "summary_quality"


async def test_human_prompt_receives_the_dimension_sentence(tmp_path: Path) -> None:
    fixtures_root = tmp_path / "fixtures"
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", write_smoke_set(fixtures_root))
    write_judge_prompt(fixtures_root, text=DIMENSION_JUDGE_PROMPT)
    prompt = _ScriptedPrompt(dict(_ANSWERS))

    await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=_judge_provider(),
        prompt_human=prompt,
        report_path=run_dir,
        root=fixtures_root,
        judge_cache_root=tmp_path / "cache",
    )

    assert prompt.judge_names["bean-stew"] == "summary_quality"
    assert prompt.dimensions["bean-stew"] == DIMENSION_SENTENCE


async def test_pre_change_run_degrades_to_unrated_without_crash(tmp_path: Path) -> None:
    # A results.json written before Epic 20.1 has no recipes /
    # fixture_content_hash / extraction_prompt_version keys: alignment must
    # neither crash nor exit-2 on the drift check — every fixture degrades to
    # unrated and the human is never prompted.
    fixtures_root = tmp_path / "fixtures"
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", write_smoke_set(fixtures_root))
    write_judge_prompt(fixtures_root)
    results_path = run_dir / "results.json"
    doc = json.loads(results_path.read_text(encoding="utf-8"))
    del doc["results"]["extraction_prompt_version"]
    for entry in doc["results"]["per_fixture"]:
        del entry["recipes"]
        del entry["fixture_content_hash"]
        del entry["scored_recipe_index"]
    results_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    prompt = _ScriptedPrompt(dict(_ANSWERS))
    provider = FakeLLMProvider()

    report = await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=provider,
        prompt_human=prompt,
        report_path=run_dir,
        root=fixtures_root,
        judge_cache_root=tmp_path / "cache",
    )

    assert prompt.asked == []
    assert provider.calls == ()
    assert report.rated == 0
    assert sorted(report.unrated) == ["bean-stew", "tomato-soup"]
    record = load_judge_alignment("summary_quality", STEW_ID, root=fixtures_root)
    assert record is not None
    assert record.human_rating is None
    assert record.agreement_status == "unrated"


async def _seed_failed_soup_run(tmp_path: Path) -> tuple[Path, Path]:
    """A run where tomato-soup's extraction was rejected (no persisted artifact)."""
    fixtures_root = tmp_path / "fixtures"
    write_smoke_set(fixtures_root)  # writes both fixtures; provider rebuilt below
    provider = FakeLLMProvider(
        {request_hash("bean-stew", STEW_SOURCE): stew_output()},
        default_output=StructuredOutputResponse(
            output_json=None,
            parse_error="rejected",
            raw_text="",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            provider="fake",
            model="fake-model",
        ),
    )
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", provider)
    write_judge_prompt(fixtures_root)
    return fixtures_root, run_dir


async def test_unrated_fixture_skips_human_and_writes_a_fresh_record(tmp_path: Path) -> None:
    fixtures_root, run_dir = await _seed_failed_soup_run(tmp_path)
    prompt = _ScriptedPrompt(dict(_ANSWERS))

    report = await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=_judge_provider(),
        prompt_human=prompt,
        report_path=run_dir,
        root=fixtures_root,
        judge_cache_root=tmp_path / "cache",
    )

    assert prompt.asked == ["bean-stew"]  # the human is never prompted for soup
    assert report.rated == 1
    assert report.unrated == ["tomato-soup"]
    record = load_judge_alignment("summary_quality", SOUP_ID, root=fixtures_root)
    assert record is not None
    assert record.human_rating is None
    assert record.judge_rating is None
    assert record.agreement_status == "unrated"
    assert "no persisted artifact" in record.run_metadata["unrated_reason"]


async def test_unrated_path_preserves_a_stored_human_rating(tmp_path: Path) -> None:
    # DECISIONS #4: a record carrying a collected human rating is left on disk
    # byte-identical — nothing written — while the fixture still counts unrated.
    fixtures_root, run_dir = await _seed_failed_soup_run(tmp_path)
    stored = JudgeAlignmentRecord(
        fixture_id=SOUP_ID,
        human_rating="pass",
        judge_rating=None,
        agreement_status="unrated",
        run_metadata={"judge_version": "v1", "human_critique": "from a 20.3 session"},
    )
    record_path = save_judge_alignment(stored, root=fixtures_root)
    before = record_path.read_bytes()
    prompt = _ScriptedPrompt(dict(_ANSWERS))

    report = await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=_judge_provider(),
        prompt_human=prompt,
        report_path=run_dir,
        root=fixtures_root,
        judge_cache_root=tmp_path / "cache",
    )

    assert record_path.read_bytes() == before  # untouched on disk
    assert prompt.asked == ["bean-stew"]
    assert report.unrated == ["tomato-soup"]


async def test_judge_error_counts_fixture_as_unrated_but_keeps_human_rating(
    tmp_path: Path,
) -> None:
    fixtures_root = tmp_path / "fixtures"
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", write_smoke_set(fixtures_root))
    write_judge_prompt(fixtures_root)
    # A malformed verdict raises JudgeError → unrated, never a silent pass/fail.
    provider = FakeLLMProvider(default_output={"rating": "maybe", "critique": "Hmm."})

    report = await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=provider,
        prompt_human=_ScriptedPrompt(dict(_ANSWERS)),
        report_path=run_dir,
        root=fixtures_root,
        judge_cache_root=tmp_path / "cache",
    )

    assert report.rated == 0
    assert sorted(report.unrated) == ["bean-stew", "tomato-soup"]
    record = load_judge_alignment("summary_quality", STEW_ID, root=fixtures_root)
    assert record is not None
    assert record.human_rating == "pass"  # the human did rate the artifact
    assert record.judge_rating is None
    assert record.agreement_status == "unrated"


async def test_rerun_reuses_human_ratings_and_judge_version_bump_reprompts(
    tmp_path: Path,
) -> None:
    fixtures_root = tmp_path / "fixtures"
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", write_smoke_set(fixtures_root))
    write_judge_prompt(fixtures_root)
    kwargs: dict[str, Any] = {
        "report_path": run_dir,
        "root": fixtures_root,
        "judge_cache_root": tmp_path / "cache",
    }
    first_prompt = _ScriptedPrompt(dict(_ANSWERS))
    await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=_judge_provider(),
        prompt_human=first_prompt,
        **kwargs,
    )
    assert sorted(first_prompt.asked) == ["bean-stew", "tomato-soup"]

    # Same judge version, same artifacts: stored human ratings reused.
    second_prompt = _ScriptedPrompt(dict(_ANSWERS))
    await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=_judge_provider(),
        prompt_human=second_prompt,
        **kwargs,
    )
    assert second_prompt.asked == []

    # Bump the prompt version: stored ratings are stale → re-prompt.
    (fixtures_root / "judge_prompts" / "summary_quality.md").write_text(
        "# Summary quality judge\n# version: v2\n\nRate harder.\n\n"
        "{extracted_output}\n{expected_output}\n{source_text}\n",
        encoding="utf-8",
    )
    third_prompt = _ScriptedPrompt(dict(_ANSWERS))
    await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=_judge_provider(),
        prompt_human=third_prompt,
        **kwargs,
    )
    assert sorted(third_prompt.asked) == ["bean-stew", "tomato-soup"]


async def test_a_reseeded_run_with_a_different_artifact_reprompts_that_fixture(
    tmp_path: Path,
) -> None:
    # Human-rating reuse binds to the artifact hash (DECISIONS #5): change the
    # persisted artifact for one fixture and only that fixture re-prompts.
    fixtures_root = tmp_path / "fixtures"
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", write_smoke_set(fixtures_root))
    write_judge_prompt(fixtures_root)
    kwargs: dict[str, Any] = {
        "report_path": run_dir,
        "root": fixtures_root,
        "judge_cache_root": tmp_path / "cache",
    }
    await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=_judge_provider(),
        prompt_human=_ScriptedPrompt(dict(_ANSWERS)),
        **kwargs,
    )

    results_path = run_dir / "results.json"
    doc = json.loads(results_path.read_text(encoding="utf-8"))
    for entry in doc["results"]["per_fixture"]:
        if entry["name"] == "bean-stew":
            entry["recipes"][0]["title"] = "Bean Stew (re-extracted)"
    results_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    prompt = _ScriptedPrompt(dict(_ANSWERS))
    await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=_judge_provider(),
        prompt_human=prompt,
        **kwargs,
    )
    assert prompt.asked == ["bean-stew"]  # soup's artifact is unchanged: reused


async def test_warm_cache_entry_for_a_different_artifact_is_not_served(
    tmp_path: Path,
) -> None:
    # DECISIONS #10 at the alignment layer: an entry keyed on another artifact
    # hash misses, the judge is re-called for the persisted artifact, and the
    # older entry survives on disk.
    fixtures_root = tmp_path / "fixtures"
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", write_smoke_set(fixtures_root))
    write_judge_prompt(fixtures_root)
    cache = JudgeCache(root=tmp_path / "cache")
    stale_key = dataclasses.replace(_run_key(run_dir, "bean-stew"), artifact_hash="0" * 64)
    stale = _seed_rating("bean-stew", "fail", artifact="0" * 64)
    cache.put(stale, key=stale_key)
    provider = _judge_provider(critique="fresh judgment")

    await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=provider,
        prompt_human=_ScriptedPrompt(dict(_ANSWERS)),
        report_path=run_dir,
        root=fixtures_root,
        judge_cache_root=tmp_path / "cache",
    )

    assert len(_judge_calls(provider)) == 2  # both fixtures freshly judged
    assert cache.get(stale_key) == stale  # the stale entry survives
    fresh = cache.get(_run_key(run_dir, "bean-stew"))
    assert fresh is not None
    assert fresh.critique == "fresh judgment"
    record = load_judge_alignment("summary_quality", STEW_ID, root=fixtures_root)
    assert record is not None
    assert record.judge_rating == "pass"  # never the stale "fail"


async def test_a_fixture_changed_after_the_run_refuses_before_any_prompt(
    tmp_path: Path,
) -> None:
    fixtures_root = tmp_path / "fixtures"
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", write_smoke_set(fixtures_root))
    write_judge_prompt(fixtures_root)
    (fixtures_root / "synthetic_recipes" / "smoke" / "bean-stew" / "source.md").write_text(
        STEW_SOURCE + "\nEdited after the run.\n", encoding="utf-8"
    )
    prompt = _ScriptedPrompt(dict(_ANSWERS))

    with pytest.raises(ValueError, match="bean-stew"):
        await run_judge_alignment(
            "summary_quality",
            "smoke",
            llm_provider=FakeLLMProvider(),
            prompt_human=prompt,
            report_path=run_dir,
            root=fixtures_root,
            judge_cache_root=tmp_path / "cache",
        )
    assert prompt.asked == []  # pre-flight: the human never rated anything


async def test_every_unusable_run_shape_raises_value_error(tmp_path: Path) -> None:
    fixtures_root = tmp_path / "fixtures"
    provider = write_smoke_set(fixtures_root)
    write_fixture(fixtures_root, "smoke2", "bean-stew", STEW_SOURCE, STEW_EXPECTED)
    write_judge_prompt(fixtures_root)
    run_dir = await _seed_run(fixtures_root, tmp_path / "reports", provider)

    async def align(report_path: Path | None, fixture_set: str = "smoke") -> None:
        await run_judge_alignment(
            "summary_quality",
            fixture_set,
            llm_provider=FakeLLMProvider(),
            prompt_human=_ScriptedPrompt(dict(_ANSWERS)),
            report_path=report_path,
            root=fixtures_root,
            judge_cache_root=tmp_path / "cache",
        )

    # No run at all.
    with pytest.raises(ValueError, match="no extraction run to align against"):
        await align(None)
    # Run dir absent.
    with pytest.raises(ValueError, match="results.json is missing"):
        await align(tmp_path / "nope")
    # Run dir exists but results.json is absent.
    (tmp_path / "empty-run").mkdir()
    with pytest.raises(ValueError, match="results.json is missing"):
        await align(tmp_path / "empty-run")
    # Malformed JSON.
    (tmp_path / "bad-run").mkdir()
    (tmp_path / "bad-run" / "results.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed results.json"):
        await align(tmp_path / "bad-run")
    # A failed run.
    (tmp_path / "failed-run").mkdir()
    (tmp_path / "failed-run" / "results.json").write_text(
        json.dumps(
            {"metadata": {}, "status": "failed", "error": "ValueError", "results": {}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finalized as failed"):
        await align(tmp_path / "failed-run")
    # A retrieval run (explicit report_type; no fixture_set).
    (tmp_path / "retrieval-run").mkdir()
    (tmp_path / "retrieval-run" / "results.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "status": "completed",
                "results": {"report_type": "retrieval", "aggregate": {}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not an extraction run"):
        await align(tmp_path / "retrieval-run")
    # A fixture-set mismatch: smoke2 fixtures against the smoke run.
    with pytest.raises(ValueError, match="not 'smoke2'"):
        await align(run_dir, fixture_set="smoke2")


def test_load_alignment_run_returns_the_results_payload(tmp_path: Path) -> None:
    fixture = RecipeFixture(name="bean-stew", source_md=STEW_SOURCE, expected=STEW_EXPECTED)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {
        "fixture_set": "smoke",
        "extraction_prompt_version": "recipe-extraction-v1",
        "per_fixture": [
            {
                "name": "bean-stew",
                "status": "scored",
                "fixture_content_hash": fixture.content_hash(),
                "recipes": [{"title": "Bean Stew"}],
            }
        ],
    }
    (run_dir / "results.json").write_text(
        json.dumps({"metadata": {}, "status": "completed", "results": payload}),
        encoding="utf-8",
    )
    assert load_alignment_run(run_dir, "smoke", [fixture]) == payload


async def test_empty_fixture_set_raises_instead_of_writing_an_empty_agreement(
    tmp_path: Path,
) -> None:
    fixtures_root = tmp_path / "fixtures"
    write_smoke_set(fixtures_root)
    write_judge_prompt(fixtures_root)
    with pytest.raises(ValueError, match="empty or does not exist"):
        await run_judge_alignment(
            "summary_quality",
            "typo",
            llm_provider=FakeLLMProvider(),
            prompt_human=_ScriptedPrompt({}),
            report_path=None,
            root=fixtures_root,
            judge_cache_root=tmp_path / "cache",
        )


async def test_agreement_section_merges_into_existing_run_results(tmp_path: Path) -> None:
    fixtures_root = tmp_path / "fixtures"
    provider = write_smoke_set(
        fixtures_root, judge_output={"rating": "pass", "critique": "OK."}
    )
    write_judge_prompt(fixtures_root)
    run_dir = await _seed_run(
        fixtures_root,
        tmp_path / "reports",
        provider,
        label="aligned-eval",
        judge="summary_quality",
        judge_cache_root=tmp_path / "cache",
    )
    before = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    judge_calls_before = len(_judge_calls(provider))

    report = await run_judge_alignment(
        "summary_quality",
        "smoke",
        llm_provider=provider,
        prompt_human=_ScriptedPrompt(
            {"bean-stew": ("pass", "ok"), "tomato-soup": ("fail", "steps missing")}
        ),
        report_path=run_dir,
        root=fixtures_root,
        judge_cache_root=tmp_path / "cache",
    )
    assert report.agreement_rate == pytest.approx(0.5)
    # The driver's --judge pass filled the cache for these artifacts, so
    # alignment replays it: zero new judge calls (the expected warm path).
    assert len(_judge_calls(provider)) == judge_calls_before

    after = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
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

    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "Judge-human agreement (summary_quality, v1): 0.50" in summary
