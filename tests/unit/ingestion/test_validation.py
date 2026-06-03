"""Unit tests for the pure validation layer in ingestion.validation.

No DB, no provider, no ``Settings`` singleton: ``ExtractedRecipe`` candidates are
built directly from the 9.2 Pydantic models (bypassing the LLM parse so the
defence-in-depth hard rules can be exercised even where 9.2's schema would have
forbidden the shape), and a ``Window`` is built from detached ``SourceSpan``s.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from rag_recipes.ingestion.pipeline.extraction import (
    ExtractedIngredient,
    ExtractedRecipe,
    ExtractedStep,
    IngredientConfidence,
    RecipeConfidence,
    RecipeFieldConfidence,
    RecipeV1StructuredData,
    StepConfidence,
)
from rag_recipes.ingestion.pipeline.windows import Window
from rag_recipes.ingestion.validation import (
    HardValidationError,
    HardValidationFailure,
    validate_hard,
)
from rag_recipes.storage.models.source_span import SourceSpan


def _make_span(page: int, text: str | None = None) -> SourceSpan:
    """Build a detached per-page SourceSpan (mirrors test_windows)."""
    body = text if text is not None else f"text of page {page}"
    return SourceSpan(
        id=f"span_{page:03d}",
        text=body,
        text_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        locator={"type": "pdf_page_range", "page_start": page, "page_end": page},
    )


def _make_window() -> Window:
    return Window(spans=(_make_span(1), _make_span(2)))


def _ingredient_confidence(**overrides: float) -> IngredientConfidence:
    base = dict(overall=0.9, quantity=0.9, unit=0.9, item=0.9, normalization=0.9)
    base.update(overrides)
    return IngredientConfidence(**base)  # type: ignore[arg-type]


def _make_ingredient(
    position: int = 1,
    raw_text: str = "2 tbsp olive oil",
    confidence: IngredientConfidence | None = None,
) -> ExtractedIngredient:
    return ExtractedIngredient(
        position=position,
        raw_text=raw_text,
        quantity_text="2",
        quantity_value=2.0,
        unit_raw="tbsp",
        unit_normalized="tablespoon",
        item_text="olive oil",
        item_normalized="olive oil",
        preparation=None,
        notes=None,
        confidence=confidence or _ingredient_confidence(),
    )


def _make_step(
    step_number: int = 1,
    text: str = "Heat the oil in a large pot.",
    source_span_ids: list[str] | None = None,
    confidence: StepConfidence | None = None,
) -> ExtractedStep:
    return ExtractedStep(
        step_number=step_number,
        text=text,
        source_span_ids=source_span_ids if source_span_ids is not None else ["span_002"],
        confidence=confidence or StepConfidence(overall=0.9, ordering=0.9),
    )


def _make_structured_data(
    ingredients: list[ExtractedIngredient] | None = None,
    steps: list[ExtractedStep] | None = None,
) -> RecipeV1StructuredData:
    return RecipeV1StructuredData(
        schema_="recipe.v1",
        yield_="Serves 4",
        prep_time=None,
        cook_time=None,
        total_time=None,
        ingredients_text=None,
        ingredients=ingredients if ingredients is not None else [_make_ingredient()],
        steps_text=None,
        steps=steps if steps is not None else [_make_step()],
    )


def _recipe_confidence(**overrides: Any) -> RecipeConfidence:
    fields = overrides.pop("fields", None) or RecipeFieldConfidence(
        title=0.9, summary=0.9, yield_=0.9, ingredients=0.9, steps=0.9
    )
    base = dict(overall=0.9, boundary=0.9)
    base.update(overrides)
    return RecipeConfidence(fields=fields, **base)  # type: ignore[arg-type]


def _make_recipe(**overrides: Any) -> ExtractedRecipe:
    base: dict[str, Any] = dict(
        item_type="recipe",
        title="Tomato and White Bean Soup",
        summary="A simple soup with pantry ingredients.",
        body_text="x" * 500,
        source_span_ids=["span_001"],
        structured_data=_make_structured_data(),
        confidence=_recipe_confidence(),
        warnings=[],
    )
    base.update(overrides)
    return ExtractedRecipe(**base)


def _codes(failures: list[HardValidationFailure]) -> list[str]:
    return [f.code for f in failures]


def test_clean_candidate_passes_hard() -> None:
    assert validate_hard(_make_recipe(), _make_window()) == []


def test_item_type_not_recipe() -> None:
    failures = validate_hard(_make_recipe(item_type="note"), _make_window())
    assert _codes(failures) == ["item_type_not_recipe"]


@pytest.mark.parametrize("title", ["", "   ", "\t\n"])
def test_missing_or_blank_title(title: str) -> None:
    failures = validate_hard(_make_recipe(title=title), _make_window())
    assert _codes(failures) == ["missing_title"]


def test_empty_source_span_ids() -> None:
    failures = validate_hard(_make_recipe(source_span_ids=[]), _make_window())
    assert _codes(failures) == ["missing_source_span_ids"]


def test_source_span_not_in_window_item_level() -> None:
    failures = validate_hard(_make_recipe(source_span_ids=["span_999"]), _make_window())
    assert _codes(failures) == ["source_span_not_in_window"]
    assert "span_999" in failures[0].message


def test_source_span_not_in_window_step_level() -> None:
    steps = [_make_step(source_span_ids=["span_888"])]
    recipe = _make_recipe(structured_data=_make_structured_data(steps=steps))
    failures = validate_hard(recipe, _make_window())
    assert _codes(failures) == ["source_span_not_in_window"]
    assert "span_888" in failures[0].message


def test_confidence_out_of_range_item_level() -> None:
    recipe = _make_recipe(confidence=_recipe_confidence(overall=1.5))
    failures = validate_hard(recipe, _make_window())
    assert _codes(failures) == ["confidence_out_of_range"]


def test_confidence_out_of_range_ingredient_level() -> None:
    ingredient = _make_ingredient(confidence=_ingredient_confidence(normalization=1.5))
    recipe = _make_recipe(structured_data=_make_structured_data(ingredients=[ingredient]))
    failures = validate_hard(recipe, _make_window())
    assert _codes(failures) == ["confidence_out_of_range"]


def test_confidence_out_of_range_step_level() -> None:
    step = _make_step(confidence=StepConfidence(overall=0.9, ordering=-0.1))
    recipe = _make_recipe(structured_data=_make_structured_data(steps=[step]))
    failures = validate_hard(recipe, _make_window())
    assert _codes(failures) == ["confidence_out_of_range"]


@pytest.mark.parametrize("raw_text", ["", "   "])
def test_ingredient_missing_raw_text(raw_text: str) -> None:
    ingredient = _make_ingredient(position=3, raw_text=raw_text)
    recipe = _make_recipe(structured_data=_make_structured_data(ingredients=[ingredient]))
    failures = validate_hard(recipe, _make_window())
    assert _codes(failures) == ["ingredient_missing_raw_text"]
    assert "3" in failures[0].message


def test_validate_hard_collects_all_failures() -> None:
    recipe = _make_recipe(item_type="note", title="   ")
    failures = validate_hard(recipe, _make_window())
    assert set(_codes(failures)) == {"item_type_not_recipe", "missing_title"}


def test_hard_validation_error_carries_failures() -> None:
    failures = [HardValidationFailure(code="missing_title", message="blank")]
    error = HardValidationError(failures)
    assert error.failures == failures
    assert "missing_title" in str(error)
