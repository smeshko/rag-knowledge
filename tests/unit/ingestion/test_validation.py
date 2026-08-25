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
)
from rag_recipes.ingestion.pipeline.windows import Window
from rag_recipes.ingestion.validation import (
    HardValidationError,
    HardValidationFailure,
    SoftValidationThresholds,
    SoftValidationWarning,
    validate_hard,
    validate_soft,
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
    base = dict(normalization=0.9)
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
) -> ExtractedStep:
    return ExtractedStep(
        step_number=step_number,
        text=text,
        source_span_ids=source_span_ids if source_span_ids is not None else ["span_002"],
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
        ingredients=ingredients if ingredients is not None else [_make_ingredient()],
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


# --- soft validation -------------------------------------------------------

# Defaults mirror Settings: 0.5 confidence floors, 200..20000 char bounds.
_THRESHOLDS = SoftValidationThresholds(
    min_overall_confidence=0.5,
    min_boundary_confidence=0.5,
    min_normalization_confidence=0.5,
    min_recipe_chars=200,
    max_recipe_chars=20000,
    assembly_min_ingredients=3,
    assembly_max_ingredients=12,
    assembly_max_chars=400,
)


def _soft_codes(warnings: list[SoftValidationWarning]) -> list[str]:
    return [w.code for w in warnings]


def test_clean_candidate_passes_soft() -> None:
    assert validate_soft(_make_recipe(), thresholds=_THRESHOLDS) == []


def test_no_ingredients() -> None:
    recipe = _make_recipe(structured_data=_make_structured_data(ingredients=[]))
    assert _soft_codes(validate_soft(recipe, thresholds=_THRESHOLDS)) == ["no_ingredients"]


def test_no_steps() -> None:
    recipe = _make_recipe(structured_data=_make_structured_data(steps=[]))
    assert _soft_codes(validate_soft(recipe, thresholds=_THRESHOLDS)) == ["no_steps"]


# --- assembly recipes ------------------------------------------------------
#
# The bowl-cookbook genre prints recipes that are a title plus a list of
# components documented elsewhere, with no method at all. ``no_steps`` and
# ``recipe_too_short`` were written for books where a missing method means the
# extraction failed; against this genre they fire on correct output. The
# exemption is bounded on three axes, and each bound is pinned below — the floor
# especially, because the shape it excludes (a method written as prose into
# body_text and never structured) is a real defect that must keep flagging.

_ASSEMBLY_BODY = (
    "THANKSGIVING IN A BOWL\n"
    "Mashed Potatoes (page 46), shredded leftover roasted turkey, roasted "
    "Brussels sprouts (page 43), cranberry sauce"
)


def _assembly(count: int = 4, body: str = _ASSEMBLY_BODY) -> ExtractedRecipe:
    """A step-less candidate with ``count`` components and a short body."""
    ingredients = [_make_ingredient(position=i) for i in range(1, count + 1)]
    return _make_recipe(
        body_text=body,
        structured_data=_make_structured_data(ingredients=ingredients, steps=[]),
    )


def test_assembly_recipe_is_not_flagged() -> None:
    assert validate_soft(_assembly(), thresholds=_THRESHOLDS) == []


def test_assembly_exemption_clears_both_size_rules_together() -> None:
    # An assembly recipe is short *because* it has no method, so waiving
    # no_steps while leaving recipe_too_short would park it in review anyway.
    body = "Bowl\n" + "x" * 100
    assert len(body) < _THRESHOLDS.min_recipe_chars
    assert validate_soft(_assembly(body=body), thresholds=_THRESHOLDS) == []


def test_too_few_components_is_a_missing_method_not_an_assembly() -> None:
    # One ingredient and prose instructions in the body: the extraction eval's
    # tomato-soup shape. Genuinely unstructured output — must keep flagging.
    recipe = _make_recipe(
        body_text="Chop the tomatoes, then simmer and blend until smooth.",
        structured_data=_make_structured_data(
            ingredients=[_make_ingredient(raw_text="4 tomatoes, chopped")], steps=[]
        ),
    )
    assert _soft_codes(validate_soft(recipe, thresholds=_THRESHOLDS)) == [
        "no_steps",
        "recipe_too_short",
    ]


def test_too_many_components_is_a_truncated_recipe_not_an_assembly() -> None:
    # The head half of a recipe split at a window boundary keeps a full
    # ingredient block and loses its method. That is data loss, not a genre.
    codes = _soft_codes(
        validate_soft(
            _assembly(count=_THRESHOLDS.assembly_max_ingredients + 1), thresholds=_THRESHOLDS
        )
    )
    assert "no_steps" in codes


def test_long_body_is_not_an_assembly() -> None:
    # Past the char ceiling there is too much text for a bare component list;
    # whatever it is, a human should look.
    body = "x" * (_THRESHOLDS.assembly_max_chars + 1)
    assert _soft_codes(validate_soft(_assembly(body=body), thresholds=_THRESHOLDS)) == ["no_steps"]


@pytest.mark.parametrize("count", [3, 12])
def test_assembly_bounds_are_inclusive(count: int) -> None:
    assert validate_soft(_assembly(count=count), thresholds=_THRESHOLDS) == []


def test_assembly_exemption_never_reaches_an_empty_candidate() -> None:
    # No ingredients and no steps is not an assembly recipe, it is empty.
    recipe = _make_recipe(
        body_text=_ASSEMBLY_BODY,
        structured_data=_make_structured_data(ingredients=[], steps=[]),
    )
    assert _soft_codes(validate_soft(recipe, thresholds=_THRESHOLDS)) == [
        "no_ingredients",
        "no_steps",
        "recipe_too_short",
    ]


def test_assembly_exemption_does_not_mask_confidence_rules() -> None:
    # Only the two size-shaped rules are waived; a badly-cut recipe still says so.
    recipe = _assembly()
    recipe = recipe.model_copy(update={"confidence": _recipe_confidence(boundary=0.3)})
    assert _soft_codes(validate_soft(recipe, thresholds=_THRESHOLDS)) == [
        "low_boundary_confidence"
    ]


def test_low_overall_confidence() -> None:
    recipe = _make_recipe(confidence=_recipe_confidence(overall=0.3))
    assert _soft_codes(validate_soft(recipe, thresholds=_THRESHOLDS)) == ["low_overall_confidence"]


def test_low_boundary_confidence() -> None:
    recipe = _make_recipe(confidence=_recipe_confidence(boundary=0.3))
    assert _soft_codes(validate_soft(recipe, thresholds=_THRESHOLDS)) == ["low_boundary_confidence"]


def test_recipe_too_short() -> None:
    recipe = _make_recipe(body_text="x" * 10)
    assert _soft_codes(validate_soft(recipe, thresholds=_THRESHOLDS)) == ["recipe_too_short"]


def test_recipe_too_long() -> None:
    recipe = _make_recipe(body_text="x" * 20001)
    assert _soft_codes(validate_soft(recipe, thresholds=_THRESHOLDS)) == ["recipe_too_long"]


def test_low_normalization_confidence() -> None:
    ingredient = _make_ingredient(confidence=_ingredient_confidence(normalization=0.3))
    recipe = _make_recipe(structured_data=_make_structured_data(ingredients=[ingredient]))
    codes = _soft_codes(validate_soft(recipe, thresholds=_THRESHOLDS))
    assert codes == ["low_normalization_confidence"]


def test_low_normalization_uses_lowest_ingredient() -> None:
    # Lowest normalization across present ingredients drives the rule.
    ingredients = [
        _make_ingredient(position=1, confidence=_ingredient_confidence(normalization=0.9)),
        _make_ingredient(position=2, confidence=_ingredient_confidence(normalization=0.2)),
    ]
    recipe = _make_recipe(structured_data=_make_structured_data(ingredients=ingredients))
    assert _soft_codes(validate_soft(recipe, thresholds=_THRESHOLDS)) == [
        "low_normalization_confidence"
    ]


def test_no_ingredients_skips_normalization_rule() -> None:
    # With zero ingredients only no_ingredients fires (no normalization to check).
    recipe = _make_recipe(structured_data=_make_structured_data(ingredients=[]))
    assert _soft_codes(validate_soft(recipe, thresholds=_THRESHOLDS)) == ["no_ingredients"]
