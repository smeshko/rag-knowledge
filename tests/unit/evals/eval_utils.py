"""Shared offline-eval test helpers (Epic 15).

Builders for canned ``recipe.v1`` provider payloads, tmp-path fixture trees,
judge prompts, and the ``FakeLLMProvider`` request hash for the driver's
rendered extraction prompt — used by ``test_extraction_eval.py`` and
``test_alignment.py``. Everything here is hermetic: no DB, no real provider,
no repo-path writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.extraction import _build_synthetic_window, synthetic_span_id

from rag_recipes.ingestion.pipeline.extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    _render_prompt,
    build_recipe_v1_json_schema,
)
from rag_recipes.ingestion.pipeline.windows import format_window_for_llm
from rag_recipes.ingestion.validation import SoftValidationThresholds
from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest

THRESHOLDS = SoftValidationThresholds(
    min_overall_confidence=0.5,
    min_boundary_confidence=0.5,
    min_normalization_confidence=0.5,
    min_recipe_chars=50,
    max_recipe_chars=20_000,
)

JUDGE_PROMPT = (
    "# Summary quality judge\n"
    "# version: v1\n\n"
    "Rate the summary.\n\n"
    "## Extracted output\n\n{extracted_output}\n\n"
    "## Expected (golden) values\n\n{expected_output}\n\n"
    "## Source text\n\n{source_text}\n"
)

# Mirrors the committed judge prompts' dimension sentence (Epic 20.1): starts
# mid-line after the framing sentence, wraps across lines, carries ** emphasis,
# and ends at the first "?". JUDGE_PROMPT above has NO such sentence, so it
# exercises the judge-name fallback; this one exercises the extraction rule.
DIMENSION_JUDGE_PROMPT = (
    "# Summary quality judge\n"
    "# version: v1\n\n"
    "You are an exacting culinary editor reviewing the output of a "
    "recipe-extraction pipeline. Rate exactly ONE subjective dimension: "
    "**summary quality** — does the\nextracted summary capture the recipe's "
    "character? Rate fail when it misleads.\n\n"
    "## Extracted output\n\n{extracted_output}\n\n"
    "## Expected (golden) values\n\n{expected_output}\n\n"
    "## Source text\n\n{source_text}\n"
)

# What `_judge_dimension` lifts from DIMENSION_JUDGE_PROMPT: marker through the
# first "?", whitespace collapsed, "**" emphasis stripped.
DIMENSION_SENTENCE = (
    "Rate exactly ONE subjective dimension: summary quality — does the "
    "extracted summary capture the recipe's character?"
)


class SettingsStandIn:
    """Lightweight ``SettingsLike`` stand-in (exactly the five read attributes)."""

    def __init__(self) -> None:
        self.embedding_provider = "openai"
        self.embedding_model = "text-embedding-3-small"
        self.llm_provider = "openai"
        self.llm_model = "gpt-4.1"
        self.anthropic_llm_model = "claude-sonnet-4-6"


def ingredient_payload(
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


def step_payload(step_number: int, text: str, span_id: str) -> dict[str, Any]:
    return {
        "step_number": step_number,
        "text": text,
        "source_span_ids": [span_id],
        "confidence": {"overall": 0.9, "ordering": 0.9},
    }


def recipe_output(
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


def golden_to_recipe_output(
    name: str, source_md: str, expected: dict[str, Any]
) -> dict[str, Any]:
    """The provider payload an *ideal* extractor would return for one golden.

    Built from that fixture's own ``expected.json`` — the golden-replay run
    (Epic 23.2 TASK-004) feeds these back through the real driver so the loader,
    hard/soft validation, and every objective scorer traverse the whole set with
    no live call. A perfect score therefore proves the golden is well formed and
    fully reachable; it says nothing about whether the golden matches the
    recipe — that is exactly and only the human verification pass.
    """
    del source_md  # part of the adapter contract (window identity), unused here
    structured = expected["structured_data"]
    span_id = synthetic_span_id(name)
    ingredients = [
        ingredient_payload(
            position,
            ingredient["raw_text"],
            ingredient["quantity_value"],
            ingredient["unit_normalized"],
            ingredient["item_normalized"],
            ingredient["preparation"],
        )
        for position, ingredient in enumerate(structured["ingredients"], start=1)
    ]
    steps = [
        step_payload(number, step["text"], span_id)
        for number, step in enumerate(structured["steps"], start=1)
    ]
    return recipe_output(
        title=expected["title"],
        span_id=span_id,
        # Soft validation bounds recipe length; the joined method is the recipe.
        body_text="\n".join(step["text"] for step in structured["steps"]),
        ingredients=ingredients,
        steps=steps,
        yield_=structured["yield"],
        prep_time=structured["prep_time"],
        cook_time=structured["cook_time"],
        total_time=structured["total_time"],
    )


def write_fixture(
    fixtures_root: Path, fixture_set: str, name: str, source_md: str, expected: dict[str, Any]
) -> None:
    fixture_dir = fixtures_root / "synthetic_recipes" / fixture_set / name
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "source.md").write_text(source_md, encoding="utf-8")
    (fixture_dir / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")


def write_judge_prompt(
    fixtures_root: Path, name: str = "summary_quality", text: str = JUDGE_PROMPT
) -> None:
    prompts = fixtures_root / "judge_prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / f"{name}.md").write_text(text, encoding="utf-8")


def request_hash(name: str, source_md: str) -> str:
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


STEW_SOURCE = "# Bean Stew\n\nA hearty stew of white beans and onion, simmered slowly.\n"
STEW_SPAN = synthetic_span_id("bean-stew")
STEW_BODY = (
    "Dice the onion and soften it in olive oil. Add the white beans and stock, "
    "then simmer gently for half an hour until thick and creamy."
)
STEW_INGREDIENTS = [
    ingredient_payload(1, "1 onion, diced", 1.0, None, "onion", "diced"),
    ingredient_payload(2, "2 cups white beans", 2.0, "cup", "white beans"),
]
STEW_STEPS = [
    step_payload(1, "Soften the onion.", STEW_SPAN),
    step_payload(2, "Simmer the beans.", STEW_SPAN),
]
STEW_EXPECTED = {
    "item_type": "recipe",
    "title": "Bean Stew",
    "source_span_ids": [STEW_SPAN],
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

SOUP_SOURCE = "# Tomato Soup\n\nA quick tomato soup.\n"
SOUP_SPAN = synthetic_span_id("tomato-soup")
SOUP_INGREDIENTS = [
    ingredient_payload(1, "4 tomatoes, chopped", 4.0, None, "tomato", "chopped"),
]
SOUP_EXPECTED = {
    "item_type": "recipe",
    "title": "Tomato Soup",
    "source_span_ids": [SOUP_SPAN],
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


def stew_output() -> dict[str, Any]:
    """The clean ``ready`` extraction for the bean-stew fixture."""
    return recipe_output(
        title="Bean Stew",
        span_id=STEW_SPAN,
        body_text=STEW_BODY,
        ingredients=STEW_INGREDIENTS,
        steps=STEW_STEPS,
    )


def soup_output() -> dict[str, Any]:
    """The step-less ``needs_review`` extraction for the tomato-soup fixture."""
    return recipe_output(
        title="Tomato Soup",
        span_id=SOUP_SPAN,
        body_text="Chop the tomatoes, then simmer and blend until smooth and silky.",
        ingredients=SOUP_INGREDIENTS,
        steps=[],
        yield_="serves 2",
        prep_time="10 minutes",
        cook_time="20 minutes",
        total_time=None,
        overall_confidence=0.7,
    )


def write_smoke_set(
    fixtures_root: Path, judge_output: dict[str, Any] | None = None
) -> FakeLLMProvider:
    """Two fixtures: a clean ``ready`` stew and a step-less ``needs_review`` soup.

    Extraction responses are keyed by request hash; ``judge_output`` (if given)
    becomes the fake's ``default_output``, served to any non-extraction request
    — i.e. the judge calls, whose distinct ``prompt_version``/``schema_version``
    guarantee their hashes never collide with the extraction ones.
    """
    write_fixture(fixtures_root, "smoke", "bean-stew", STEW_SOURCE, STEW_EXPECTED)
    write_fixture(fixtures_root, "smoke", "tomato-soup", SOUP_SOURCE, SOUP_EXPECTED)
    return FakeLLMProvider(
        {
            request_hash("bean-stew", STEW_SOURCE): stew_output(),
            request_hash("tomato-soup", SOUP_SOURCE): soup_output(),
        },
        default_output=judge_output,
    )
