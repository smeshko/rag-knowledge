"""Offline tests for the extraction-eval driver (Epic 15 Phase 15.1).

Hermetic by construction: fixtures live under ``tmp_path``, reports are written
to ``tmp_path``, thresholds and settings are injected, and every extraction is
driven by ``FakeLLMProvider`` — no real API, no key, no ``live`` marker.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evals.cli
import evals.fixtures
import evals.reports
import pytest
from evals.cli import app
from evals.extraction import _build_synthetic_window, run_extraction_eval, synthetic_span_id
from typer.testing import CliRunner

from rag_recipes.ingestion.pipeline.extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    _render_prompt,
    build_recipe_v1_json_schema,
)
from rag_recipes.ingestion.pipeline.windows import format_window_for_llm
from rag_recipes.ingestion.validation import SoftValidationThresholds
from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import (
    StructuredOutputRequest,
    StructuredOutputResponse,
    TokenUsage,
)

_THRESHOLDS = SoftValidationThresholds(
    min_overall_confidence=0.5,
    min_boundary_confidence=0.5,
    min_normalization_confidence=0.5,
    min_recipe_chars=50,
    max_recipe_chars=20_000,
)


class _SettingsStandIn:
    """Lightweight ``SettingsLike`` stand-in (exactly the five read attributes)."""

    def __init__(self) -> None:
        self.embedding_provider = "openai"
        self.embedding_model = "text-embedding-3-small"
        self.llm_provider = "openai"
        self.llm_model = "gpt-4.1"
        self.anthropic_llm_model = "claude-sonnet-4-6"


def _ingredient_payload(
    position: int,
    raw_text: str,
    quantity_value: float | None,
    unit: str | None,
    item: str | None,
    preparation: str | None = None,
) -> dict[str, Any]:
    return {
        "position": position,
        "raw_text": raw_text,
        "quantity_text": None if quantity_value is None else str(quantity_value),
        "quantity_value": quantity_value,
        "unit_raw": unit,
        "unit_normalized": unit,
        "item_text": item,
        "item_normalized": item,
        "preparation": preparation,
        "notes": None,
        "confidence": {
            "overall": 0.9,
            "quantity": 0.9,
            "unit": 0.9,
            "item": 0.9,
            "normalization": 0.9,
        },
    }


def _step_payload(step_number: int, text: str, span_id: str) -> dict[str, Any]:
    return {
        "step_number": step_number,
        "text": text,
        "source_span_ids": [span_id],
        "confidence": {"overall": 0.9, "ordering": 0.9},
    }


def _recipe_output(
    *,
    title: str,
    span_id: str,
    body_text: str,
    ingredients: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    yield_: str | None = "serves 4",
    prep_time: str | None = "15 minutes",
    cook_time: str | None = "30 minutes",
    total_time: str | None = "45 minutes",
    overall_confidence: float = 0.9,
) -> dict[str, Any]:
    """A full, valid ``recipe.v1`` provider payload for one recipe."""
    return {
        "items": [
            {
                "item_type": "recipe",
                "title": title,
                "summary": "A simple, comforting dish.",
                "body_text": body_text,
                "source_span_ids": [span_id],
                "structured_data": {
                    "schema": "recipe.v1",
                    "yield": yield_,
                    "prep_time": prep_time,
                    "cook_time": cook_time,
                    "total_time": total_time,
                    "ingredients_text": "\n".join(i["raw_text"] for i in ingredients),
                    "ingredients": ingredients,
                    "steps_text": "\n".join(s["text"] for s in steps),
                    "steps": steps,
                },
                "confidence": {
                    "overall": overall_confidence,
                    "boundary": 0.9,
                    "fields": {
                        "title": 0.9,
                        "summary": 0.9,
                        "yield": 0.9,
                        "ingredients": 0.9,
                        "steps": 0.9,
                    },
                },
                "warnings": [],
            }
        ]
    }


def _write_fixture(
    fixtures_root: Path, fixture_set: str, name: str, source_md: str, expected: dict[str, Any]
) -> None:
    fixture_dir = fixtures_root / "synthetic_recipes" / fixture_set / name
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "source.md").write_text(source_md, encoding="utf-8")
    (fixture_dir / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")


def _request_hash(name: str, source_md: str) -> str:
    """The ``FakeLLMProvider`` hash for the driver's rendered fixture prompt."""
    window = _build_synthetic_window(name, source_md)
    request = StructuredOutputRequest(
        provider="fake",
        model="fake-model",
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        input=_render_prompt(format_window_for_llm(window)),
        json_schema=build_recipe_v1_json_schema(),
    )
    return FakeLLMProvider.request_hash(request)


_STEW_SOURCE = "# Bean Stew\n\nA hearty stew of white beans and onion, simmered slowly.\n"
_STEW_SPAN = synthetic_span_id("bean-stew")
_STEW_BODY = (
    "Dice the onion and soften it in olive oil. Add the white beans and stock, "
    "then simmer gently for half an hour until thick and creamy."
)
_STEW_INGREDIENTS = [
    _ingredient_payload(1, "1 onion, diced", 1.0, None, "onion", "diced"),
    _ingredient_payload(2, "2 cups white beans", 2.0, "cup", "white beans"),
]
_STEW_STEPS = [
    _step_payload(1, "Soften the onion.", _STEW_SPAN),
    _step_payload(2, "Simmer the beans.", _STEW_SPAN),
]
_STEW_EXPECTED = {
    "item_type": "recipe",
    "title": "Bean Stew",
    "source_span_ids": [_STEW_SPAN],
    "structured_data": {
        "schema": "recipe.v1",
        "yield": "serves 4",
        "prep_time": "15 min",
        "cook_time": "30 min",
        "total_time": "45 min",
        "ingredients": [
            {
                "raw_text": "1 onion, diced",
                "quantity_value": 1.0,
                "unit_normalized": None,
                "item_normalized": "onion",
                "preparation": "diced",
            },
            {
                "raw_text": "2 cups white beans",
                "quantity_value": 2.0,
                "unit_normalized": "cup",
                "item_normalized": "white beans",
                "preparation": None,
            },
        ],
        "steps": [{"text": "Soften the onion."}, {"text": "Simmer the beans."}],
    },
}

_SOUP_SOURCE = "# Tomato Soup\n\nA quick tomato soup.\n"
_SOUP_SPAN = synthetic_span_id("tomato-soup")
_SOUP_INGREDIENTS = [
    _ingredient_payload(1, "4 tomatoes, chopped", 4.0, None, "tomato", "chopped"),
]
_SOUP_EXPECTED = {
    "item_type": "recipe",
    "title": "Tomato Soup",
    "source_span_ids": [_SOUP_SPAN],
    "structured_data": {
        "schema": "recipe.v1",
        "yield": "serves 2",
        "prep_time": "10 min",
        "cook_time": "20 min",
        "total_time": "30 min",
        "ingredients": [
            {
                "raw_text": "4 tomatoes, chopped",
                "quantity_value": 4.0,
                "unit_normalized": None,
                "item_normalized": "tomato",
                "preparation": "chopped",
            },
        ],
        "steps": [{"text": "Chop the tomatoes."}, {"text": "Simmer and blend."}],
    },
}


def _write_smoke_set(fixtures_root: Path) -> FakeLLMProvider:
    """Two fixtures: a clean ``ready`` stew and a step-less ``needs_review`` soup."""
    _write_fixture(fixtures_root, "smoke", "bean-stew", _STEW_SOURCE, _STEW_EXPECTED)
    _write_fixture(fixtures_root, "smoke", "tomato-soup", _SOUP_SOURCE, _SOUP_EXPECTED)
    stew_output = _recipe_output(
        title="Bean Stew",
        span_id=_STEW_SPAN,
        body_text=_STEW_BODY,
        ingredients=_STEW_INGREDIENTS,
        steps=_STEW_STEPS,
        prep_time="15 minutes",
        cook_time="30 minutes",
        total_time="45 minutes",
    )
    # The soup extraction misses every step (fires the `no_steps` soft rule →
    # needs_review) and drops the total_time the golden expects.
    soup_output = _recipe_output(
        title="Tomato Soup",
        span_id=_SOUP_SPAN,
        body_text="Chop the tomatoes, then simmer and blend until smooth and silky.",
        ingredients=_SOUP_INGREDIENTS,
        steps=[],
        yield_="serves 2",
        prep_time="10 minutes",
        cook_time="20 minutes",
        total_time=None,
        overall_confidence=0.7,
    )
    return FakeLLMProvider(
        {
            _request_hash("bean-stew", _STEW_SOURCE): stew_output,
            _request_hash("tomato-soup", _SOUP_SOURCE): soup_output,
        }
    )


async def _run_smoke_eval(tmp_path: Path, **kwargs: Any):  # noqa: ANN202
    fixtures_root = tmp_path / "fixtures"
    provider = _write_smoke_set(fixtures_root)
    return await run_extraction_eval(
        "smoke",
        "smoke-eval",
        llm_provider=provider,
        fixtures_root=fixtures_root,
        reports_root=tmp_path / "reports",
        thresholds=_THRESHOLDS,
        settings=_SettingsStandIn(),
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


async def test_judge_parameter_is_accepted_and_inert(tmp_path: Path) -> None:
    run = await _run_smoke_eval(tmp_path, judge="completeness")
    results = json.loads((run.path / "results.json").read_text(encoding="utf-8"))["results"]
    assert results["judge"] is None


async def test_rejected_extraction_is_counted_not_crashed(tmp_path: Path) -> None:
    fixtures_root = tmp_path / "fixtures"
    _write_fixture(fixtures_root, "smoke", "bean-stew", _STEW_SOURCE, _STEW_EXPECTED)
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
        thresholds=_THRESHOLDS,
        settings=_SettingsStandIn(),
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


async def test_baseline_diff_is_invoked_and_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline_path = tmp_path / "baselines" / "extraction.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(json.dumps({"baseline_set_at": "2026-01-01T00:00:00+00:00"}))
    await _run_smoke_eval(tmp_path, baseline_path=baseline_path)
    captured = capsys.readouterr()
    assert "diff not implemented yet" in captured.out


async def test_no_baseline_is_handled_without_error(tmp_path: Path) -> None:
    run = await _run_smoke_eval(
        tmp_path, baseline_path=tmp_path / "baselines" / "extraction.json"
    )
    assert (run.path / "results.json").is_file()


# --- CLI wiring (Epic 15 Phase 15.1 TASK-003) --------------------------------

runner = CliRunner()


class _CliSettingsStandIn(_SettingsStandIn):
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
    provider = _write_smoke_set(fixtures_root)
    monkeypatch.setattr(evals.fixtures, "FIXTURES_ROOT", fixtures_root)
    monkeypatch.setattr(evals.reports, "REPORTS_ROOT", tmp_path / "reports")
    monkeypatch.setattr("rag_recipes.config.get_settings", _CliSettingsStandIn)
    monkeypatch.setattr(evals.cli, "_build_llm_provider", lambda settings: provider)

    result = runner.invoke(
        app,
        ["extraction", "--fixtures", "smoke", "--label", "smoke", "--judge", "completeness"],
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
