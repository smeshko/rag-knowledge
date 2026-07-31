"""Offline tests for the extraction-eval driver (Epic 15 Phases 15.1/15.2).

Hermetic by construction: fixtures live under ``tmp_path``, reports are written
to ``tmp_path``, thresholds and settings are injected, and every extraction and
judge call is driven by ``FakeLLMProvider`` — no real API, no key, no ``live``
marker. Shared payload/fixture builders live in ``eval_utils``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evals.cli
import evals.fixtures
import evals.judge_cache
import evals.reports
import pytest
from evals.cli import app
from evals.extraction import run_extraction_eval
from evals.judges import JUDGE_SCHEMA_VERSION
from typer.testing import CliRunner

from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputResponse, TokenUsage
from tests.unit.evals.eval_utils import (
    STEW_EXPECTED,
    STEW_SOURCE,
    THRESHOLDS,
    SettingsStandIn,
    write_fixture,
    write_judge_prompt,
    write_smoke_set,
)


async def _run_smoke_eval(tmp_path: Path, **kwargs: Any):  # noqa: ANN202
    fixtures_root = tmp_path / "fixtures"
    provider = write_smoke_set(fixtures_root)
    return await run_extraction_eval(
        "smoke",
        "smoke-eval",
        llm_provider=provider,
        fixtures_root=fixtures_root,
        reports_root=tmp_path / "reports",
        thresholds=THRESHOLDS,
        settings=SettingsStandIn(),
        **kwargs,
    )


async def test_results_json_envelope_and_reserved_slots(tmp_path: Path) -> None:
    run = await _run_smoke_eval(tmp_path)
    doc = json.loads((run.path / "results.json").read_text(encoding="utf-8"))
    assert set(doc) == {"metadata", "status", "results"}
    assert doc["status"] == "completed"
    results = doc["results"]
    assert results["fixture_set"] == "smoke"
    assert results["judge"] is None
    assert results["agreement"] is None
    assert results["calibration"] is None


async def test_per_fixture_scores_and_ready_needs_review_split(tmp_path: Path) -> None:
    run = await _run_smoke_eval(tmp_path)
    results = json.loads((run.path / "results.json").read_text(encoding="utf-8"))["results"]

    by_name = {entry["name"]: entry for entry in results["per_fixture"]}
    assert set(by_name) == {"bean-stew", "tomato-soup"}

    stew = by_name["bean-stew"]
    assert stew["status"] == "scored"
    assert stew["review_status"] == "ready"
    assert stew["warnings"] == []
    assert stew["missing_fields"] == []
    assert stew["scores"]["title"] == {"exact": True, "normalized": True}
    assert stew["scores"]["yield"] is True
    assert stew["scores"]["prep_time"]["match"] is True
    assert stew["scores"]["ingredient_count"] is True
    assert stew["scores"]["step_count"] is True
    assert stew["scores"]["ingredients_detail"]["f1"] == pytest.approx(1.0)
    assert stew["scores"]["source_span_ids"]["f1"] == pytest.approx(1.0)

    soup = by_name["tomato-soup"]
    assert soup["review_status"] == "needs_review"
    assert soup["warnings"] == ["no_steps"]
    assert soup["missing_fields"] == ["total_time", "steps"]
    assert soup["scores"]["step_count"] is False
    assert soup["scores"]["total_time"]["match"] is False

    aggregate = results["aggregate"]
    assert aggregate["fixtures"] == 2
    assert aggregate["recipes_extracted"] == 2
    assert aggregate["extraction_failures"] == 0
    assert aggregate["ready"] == 1
    assert aggregate["needs_review"] == 1
    assert aggregate["average_confidence"] == pytest.approx(0.8)
    accuracy = aggregate["field_accuracy"]
    assert accuracy["title_exact"] == pytest.approx(1.0)
    assert accuracy["title_normalized"] == pytest.approx(1.0)
    assert accuracy["yield"] == pytest.approx(1.0)
    assert accuracy["step_count"] == pytest.approx(0.5)
    assert accuracy["total_time"] == pytest.approx(0.5)
    assert aggregate["missing_field_counts"] == {"total_time": 1, "steps": 1}


async def test_summary_renders_doc12_extraction_report_shape(tmp_path: Path) -> None:
    run = await _run_smoke_eval(tmp_path)
    summary = (run.path / "summary.md").read_text(encoding="utf-8")
    assert "Fixture set: smoke (2 fixtures)" in summary
    assert "Recipes extracted: 2" in summary
    assert "Ready: 1" in summary
    assert "Needs review: 1" in summary
    assert "Average confidence: 0.80" in summary
    assert "Objective field accuracy:" in summary
    assert "  title: 1.00" in summary
    assert "  yield: 1.00" in summary
    assert "  ingredients (count match): 1.00" in summary
    assert "  ingredients (per-field): 1.00" in summary
    assert "  steps (count match): 0.50" in summary
    assert "Missing ingredients: 0" in summary
    assert "Missing steps: 1" in summary


async def test_no_judge_leaves_judge_slot_null(tmp_path: Path) -> None:
    run = await _run_smoke_eval(tmp_path)
    results = json.loads((run.path / "results.json").read_text(encoding="utf-8"))["results"]
    assert results["judge"] is None


async def test_unknown_judge_name_surfaces_missing_prompt(tmp_path: Path) -> None:
    # Since Phase 15.2 the judge seam is live: a judge name without a committed
    # prompt is a caller error, surfaced as the loader's FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        await _run_smoke_eval(tmp_path, judge="completeness")


async def test_rejected_extraction_is_counted_not_crashed(tmp_path: Path) -> None:
    fixtures_root = tmp_path / "fixtures"
    write_fixture(fixtures_root, "smoke", "bean-stew", STEW_SOURCE, STEW_EXPECTED)
    provider = FakeLLMProvider(
        default_output=StructuredOutputResponse(
            output_json=None,
            parse_error="max_tokens truncation",
            raw_text="{\"items\": [",
            usage=TokenUsage(input_tokens=10, output_tokens=10),
            provider="fake",
            model="fake-model",
        )
    )
    run = await run_extraction_eval(
        "smoke",
        "rejected-eval",
        llm_provider=provider,
        fixtures_root=fixtures_root,
        reports_root=tmp_path / "reports",
        thresholds=THRESHOLDS,
        settings=SettingsStandIn(),
    )
    results = json.loads((run.path / "results.json").read_text(encoding="utf-8"))["results"]
    entry = results["per_fixture"][0]
    assert entry["status"] == "extraction_failed"
    assert entry["error"] == "max_tokens truncation"
    aggregate = results["aggregate"]
    assert aggregate["extraction_failures"] == 1
    assert aggregate["recipes_extracted"] == 0
    assert aggregate["ready"] == 0
    assert aggregate["needs_review"] == 0
    assert aggregate["average_confidence"] is None


async def test_a_candidate_production_would_discard_is_not_scored_or_counted_ready(
    tmp_path: Path,
) -> None:
    # persist.py runs validate_hard first and persists nothing when it fires, so
    # a candidate citing a span that is not in the window must not be blessed
    # `ready` and handed a perfect span-provenance score by the eval.
    from tests.unit.evals.eval_utils import request_hash, stew_output

    fixtures_root = tmp_path / "fixtures"
    write_fixture(fixtures_root, "smoke", "bean-stew", STEW_SOURCE, STEW_EXPECTED)
    bad = stew_output()
    bad["items"][0]["source_span_ids"] = ["span_hallucinated"]
    provider = FakeLLMProvider({request_hash("bean-stew", STEW_SOURCE): bad})
    run = await run_extraction_eval(
        "smoke",
        "hard-fail-eval",
        llm_provider=provider,
        fixtures_root=fixtures_root,
        reports_root=tmp_path / "reports",
        thresholds=THRESHOLDS,
        settings=SettingsStandIn(),
    )
    results = json.loads((run.path / "results.json").read_text(encoding="utf-8"))["results"]
    entry = results["per_fixture"][0]
    assert entry["status"] == "hard_validation_failed"
    assert "source_span_not_in_window" in entry["failures"]
    assert "scores" not in entry
    aggregate = results["aggregate"]
    assert aggregate["hard_validation_failures"] == 1
    assert aggregate["ready"] == 0
    assert aggregate["needs_review"] == 0
    assert aggregate["extraction_success_rate"] == pytest.approx(0.0)
    assert aggregate["field_accuracy"]["title_normalized"] is None


async def test_survivor_only_accuracy_is_caught_by_the_coverage_rate(tmp_path: Path) -> None:
    # A run where one fixture scores perfectly and the other is rejected keeps
    # accuracy at 1.00; the coverage rate is what makes the collapse visible.
    from tests.unit.evals.eval_utils import SOUP_SOURCE, request_hash, stew_output

    fixtures_root = tmp_path / "fixtures"
    provider = write_smoke_set(fixtures_root)
    provider = FakeLLMProvider(
        {request_hash("bean-stew", STEW_SOURCE): stew_output()},
        default_output=StructuredOutputResponse(
            output_json=None,
            parse_error="max_tokens truncation",
            raw_text="",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            provider="fake",
            model="fake-model",
        ),
    )
    assert SOUP_SOURCE  # the soup fixture is the one that gets rejected
    run = await run_extraction_eval(
        "smoke",
        "survivor-eval",
        llm_provider=provider,
        fixtures_root=fixtures_root,
        reports_root=tmp_path / "reports",
        thresholds=THRESHOLDS,
        settings=SettingsStandIn(),
    )
    aggregate = json.loads((run.path / "results.json").read_text(encoding="utf-8"))["results"][
        "aggregate"
    ]
    assert aggregate["field_accuracy"]["title_normalized"] == pytest.approx(1.0)
    assert aggregate["extraction_success_rate"] == pytest.approx(0.5)


async def test_a_fixture_split_across_items_is_counted_not_collapsed(tmp_path: Path) -> None:
    # Each synthetic fixture holds exactly one recipe, so an extractor that
    # emits two items for it is a boundary failure the report must surface —
    # not silently collapse to "1 recipe extracted".
    from tests.unit.evals.eval_utils import request_hash, stew_output

    fixtures_root = tmp_path / "fixtures"
    write_fixture(fixtures_root, "smoke", "bean-stew", STEW_SOURCE, STEW_EXPECTED)
    split = stew_output()
    split["items"].append(json.loads(json.dumps(split["items"][0])))
    provider = FakeLLMProvider({request_hash("bean-stew", STEW_SOURCE): split})
    run = await run_extraction_eval(
        "smoke",
        "split-eval",
        llm_provider=provider,
        fixtures_root=fixtures_root,
        reports_root=tmp_path / "reports",
        thresholds=THRESHOLDS,
        settings=SettingsStandIn(),
    )
    results = json.loads((run.path / "results.json").read_text(encoding="utf-8"))["results"]
    assert results["per_fixture"][0]["recipes_returned"] == 2
    assert results["aggregate"]["recipes_extracted"] == 2
    assert results["aggregate"]["fixtures"] == 1
    assert results["aggregate"]["over_split_fixtures"] == 1
    assert "Fixtures split across items: 1" in (run.path / "summary.md").read_text(
        encoding="utf-8"
    )


async def test_explicit_nulls_in_expected_json_score_as_misses_not_a_crash(
    tmp_path: Path,
) -> None:
    # `expected.json` is an opaque dict, so a fixture may write explicit nulls;
    # `.get(key, default)` returns None for those, which used to blow up in
    # normalize_title / list().
    fixtures_root = tmp_path / "fixtures"
    provider = write_smoke_set(fixtures_root)
    (fixtures_root / "synthetic_recipes" / "smoke" / "bean-stew" / "expected.json").write_text(
        json.dumps(
            {
                "item_type": "recipe",
                "title": None,
                "source_span_ids": None,
                "structured_data": None,
            }
        ),
        encoding="utf-8",
    )
    run = await run_extraction_eval(
        "smoke",
        "null-eval",
        llm_provider=provider,
        fixtures_root=fixtures_root,
        reports_root=tmp_path / "reports",
        thresholds=THRESHOLDS,
        settings=SettingsStandIn(),
    )
    results = json.loads((run.path / "results.json").read_text(encoding="utf-8"))["results"]
    stew = next(entry for entry in results["per_fixture"] if entry["name"] == "bean-stew")
    assert stew["status"] == "scored"
    assert stew["scores"]["title"] == {"exact": False, "normalized": False}
    assert stew["scores"]["ingredient_count"] is False
    assert stew["missing_fields"] == []


async def test_empty_fixture_set_raises_instead_of_writing_a_green_report(
    tmp_path: Path,
) -> None:
    # A mistyped --fixtures must not produce a completed report with zero
    # fixtures whose baseline diff then prints "No regressions detected".
    reports_root = tmp_path / "reports"
    with pytest.raises(ValueError, match="empty or does not exist"):
        await run_extraction_eval(
            "typo",
            "empty-eval",
            llm_provider=FakeLLMProvider(),
            fixtures_root=tmp_path / "fixtures",
            reports_root=reports_root,
            thresholds=THRESHOLDS,
            settings=SettingsStandIn(),
        )
    assert not reports_root.exists()


def test_cli_extraction_exits_non_zero_on_an_unknown_fixture_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evals.fixtures, "FIXTURES_ROOT", tmp_path / "fixtures")
    monkeypatch.setattr(evals.reports, "REPORTS_ROOT", tmp_path / "reports")
    monkeypatch.setattr("rag_recipes.config.get_settings", _CliSettingsStandIn)
    monkeypatch.setattr(evals.cli, "_build_llm_provider", lambda settings: FakeLLMProvider())

    result = runner.invoke(app, ["extraction", "--fixtures", "typo", "--label", "smoke"])

    assert result.exit_code == 2
    assert "report: " not in result.output


async def test_baseline_diff_is_invoked_and_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline_path = tmp_path / "baselines" / "extraction.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(json.dumps({"baseline_set_at": "2026-01-01T00:00:00+00:00"}))
    await _run_smoke_eval(tmp_path, baseline_path=baseline_path)
    captured = capsys.readouterr()
    assert "diff" in captured.out


async def test_no_baseline_is_handled_without_error(tmp_path: Path) -> None:
    run = await _run_smoke_eval(
        tmp_path, baseline_path=tmp_path / "baselines" / "extraction.json"
    )
    assert (run.path / "results.json").is_file()


# --- judge integration (Epic 15 Phase 15.2) ----------------------------------


def _judge_calls(provider: FakeLLMProvider) -> list[Any]:
    return [call for call in provider.calls if call.schema_version == JUDGE_SCHEMA_VERSION]


async def test_judge_records_ratings_and_pass_rate(tmp_path: Path) -> None:
    fixtures_root = tmp_path / "fixtures"
    provider = write_smoke_set(fixtures_root, judge_output={"rating": "pass", "critique": "OK."})
    write_judge_prompt(fixtures_root)
    run = await run_extraction_eval(
        "smoke",
        "judged-eval",
        llm_provider=provider,
        judge="summary_quality",
        fixtures_root=fixtures_root,
        reports_root=tmp_path / "reports",
        thresholds=THRESHOLDS,
        settings=SettingsStandIn(),
        judge_cache_root=tmp_path / "judge-cache",
    )
    results = json.loads((run.path / "results.json").read_text(encoding="utf-8"))["results"]
    judge = results["judge"]
    assert judge["name"] == "summary_quality"
    assert judge["version"] == "v1"
    assert judge["model"] == "fake-model"
    assert judge["pass_rate"] == pytest.approx(1.0)
    assert judge["rated"] == 2
    assert judge["unrated"] == 0
    assert set(judge["per_fixture"]) == {"bean-stew", "tomato-soup"}
    stew = judge["per_fixture"]["bean-stew"]
    assert stew["status"] == "rated"
    assert stew["rating"] == "pass"
    assert stew["critique"] == "OK."
    assert stew["metadata"]["model"] == "fake-model"
    # Objective scoring is unchanged by the judge integration.
    assert results["aggregate"]["ready"] == 1
    assert results["aggregate"]["needs_review"] == 1

    summary = (run.path / "summary.md").read_text(encoding="utf-8")
    assert "Judge pass rate (summary_quality, v1): 1.00" in summary


async def test_second_judge_run_replays_cache_without_new_calls(tmp_path: Path) -> None:
    fixtures_root = tmp_path / "fixtures"
    provider = write_smoke_set(fixtures_root, judge_output={"rating": "pass", "critique": "OK."})
    write_judge_prompt(fixtures_root)
    kwargs: dict[str, Any] = {
        "llm_provider": provider,
        "judge": "summary_quality",
        "fixtures_root": fixtures_root,
        "reports_root": tmp_path / "reports",
        "thresholds": THRESHOLDS,
        "settings": SettingsStandIn(),
        "judge_cache_root": tmp_path / "judge-cache",
    }
    await run_extraction_eval("smoke", "first", **kwargs)
    assert len(_judge_calls(provider)) == 2  # first run judges both fixtures
    run = await run_extraction_eval("smoke", "second", **kwargs)
    assert len(_judge_calls(provider)) == 2  # cache hit: no new judge calls
    results = json.loads((run.path / "results.json").read_text(encoding="utf-8"))["results"]
    assert results["judge"]["pass_rate"] == pytest.approx(1.0)
    assert results["judge"]["rated"] == 2


async def test_judge_error_counts_fixture_as_unrated(tmp_path: Path) -> None:
    fixtures_root = tmp_path / "fixtures"
    # A malformed verdict (the Anthropic path is non-strict, so reachable) must
    # surface as un-rated, never as a silent pass or fail.
    provider = write_smoke_set(
        fixtures_root, judge_output={"rating": "maybe", "critique": "Hmm."}
    )
    write_judge_prompt(fixtures_root)
    run = await run_extraction_eval(
        "smoke",
        "unrated-eval",
        llm_provider=provider,
        judge="summary_quality",
        fixtures_root=fixtures_root,
        reports_root=tmp_path / "reports",
        thresholds=THRESHOLDS,
        settings=SettingsStandIn(),
        judge_cache_root=tmp_path / "judge-cache",
    )
    results = json.loads((run.path / "results.json").read_text(encoding="utf-8"))["results"]
    judge = results["judge"]
    assert judge["rated"] == 0
    assert judge["unrated"] == 2
    assert judge["pass_rate"] is None
    for entry in judge["per_fixture"].values():
        assert entry["status"] == "unrated"
        assert "verdict" in entry["error"]
    summary = (run.path / "summary.md").read_text(encoding="utf-8")
    assert "Judge pass rate (summary_quality, v1): n/a" in summary


# --- CLI wiring (Epic 15 Phase 15.1 TASK-003) --------------------------------

runner = CliRunner()


class _CliSettingsStandIn(SettingsStandIn):
    """Extends the metadata stand-in with the soft-validation threshold fields."""

    def __init__(self) -> None:
        super().__init__()
        self.extraction_min_overall_confidence = 0.5
        self.extraction_min_boundary_confidence = 0.5
        self.extraction_min_normalization_confidence = 0.5
        self.extraction_min_recipe_chars = 50
        self.extraction_max_recipe_chars = 20_000


def test_cli_extraction_runs_offline_with_injected_fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures_root = tmp_path / "fixtures"
    provider = write_smoke_set(fixtures_root, judge_output={"rating": "pass", "critique": "OK."})
    write_judge_prompt(fixtures_root)
    monkeypatch.setattr(evals.fixtures, "FIXTURES_ROOT", fixtures_root)
    monkeypatch.setattr(evals.reports, "REPORTS_ROOT", tmp_path / "reports")
    monkeypatch.setattr(evals.judge_cache, "CACHE_ROOT", tmp_path / "judge-cache")
    monkeypatch.setattr("rag_recipes.config.get_settings", _CliSettingsStandIn)
    monkeypatch.setattr(evals.cli, "_build_llm_provider", lambda settings: provider)

    result = runner.invoke(
        app,
        ["extraction", "--fixtures", "smoke", "--label", "smoke", "--judge", "summary_quality"],
    )

    assert result.exit_code == 0
    assert "report: " in result.output
    report_path = Path(result.output.split("report: ", 1)[1].strip())
    assert report_path.parent == tmp_path / "reports"
    assert (report_path / "results.json").is_file()
    assert (report_path / "summary.md").is_file()


def test_cli_extraction_help_shows_flags_as_options_not_positionals() -> None:
    result = runner.invoke(app, ["extraction", "--help"])
    assert result.exit_code == 0
    for option in ("--fixtures", "--label", "--judge"):
        assert option in result.output
    # A bare-typed param would render as a positional metavar in the usage line.
    assert "LABEL" not in result.output
    assert "FIXTURES" not in result.output
