"""Unit tests for the ``review_reasons`` projection (Epic 21.1, D3).

``build_review_reasons`` is pure (no session): it maps the persisted
``structured_data["warnings"]`` codes into ``{code, message}`` pairs for
``needs_review`` items and returns ``[]`` for everything else. The drift
guard at the bottom fires every ``validate_soft`` rule and asserts the
static message map covers every emittable code.
"""

from __future__ import annotations

from rag_recipes.api.review_reasons import (
    SOFT_WARNING_MESSAGES,
    build_review_reasons,
)
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
from rag_recipes.ingestion.validation import SoftValidationThresholds, validate_soft


def test_needs_review_known_codes_project_code_and_message() -> None:
    reasons = build_review_reasons(
        "needs_review",
        {"warnings": ["no_ingredients", "low_overall_confidence"]},
    )
    assert [r.code for r in reasons] == ["no_ingredients", "low_overall_confidence"]
    assert reasons[0].message == SOFT_WARNING_MESSAGES["no_ingredients"]
    assert reasons[1].message == SOFT_WARNING_MESSAGES["low_overall_confidence"]
    assert all(r.message for r in reasons)


def test_unknown_string_projects_as_llm_warning_envelope() -> None:
    reasons = build_review_reasons(
        "needs_review", {"warnings": ["model said something odd"]}
    )
    assert len(reasons) == 1
    assert reasons[0].code == "llm_warning"
    assert reasons[0].message == "model said something odd"


def test_non_string_element_projects_as_llm_warning_and_never_raises() -> None:
    # structured_data is opaque JSONB: a dict element must not 500 the search
    # endpoint via `unhashable type` (D3 isinstance guard).
    reasons = build_review_reasons(
        "needs_review", {"warnings": [{"code": "weird"}, 42]}
    )
    assert [r.code for r in reasons] == ["llm_warning", "llm_warning"]
    assert reasons[0].message == str({"code": "weird"})
    assert reasons[1].message == "42"


def test_ready_status_returns_empty_list_even_with_warnings() -> None:
    assert build_review_reasons("ready", {"warnings": ["no_ingredients"]}) == []


def test_missing_or_none_warnings_key_returns_empty_list() -> None:
    assert build_review_reasons("needs_review", {}) == []
    assert build_review_reasons("needs_review", {"warnings": None}) == []


# --- Drift guard: every validate_soft-emittable code has a map entry ---

_THRESHOLDS = SoftValidationThresholds(
    min_overall_confidence=0.5,
    min_boundary_confidence=0.5,
    min_normalization_confidence=0.5,
    min_recipe_chars=10,
    max_recipe_chars=50,
)


def _confidence(overall: float, boundary: float) -> RecipeConfidence:
    return RecipeConfidence(
        overall=overall,
        boundary=boundary,
        fields=RecipeFieldConfidence(
            title=1.0, summary=1.0, yield_=1.0, ingredients=1.0, steps=1.0
        ),
    )


def _structured(
    ingredients: list[ExtractedIngredient], steps: list[ExtractedStep]
) -> RecipeV1StructuredData:
    return RecipeV1StructuredData(
        yield_=None,
        prep_time=None,
        cook_time=None,
        total_time=None,
        ingredients_text=None,
        ingredients=ingredients,
        steps_text=None,
        steps=steps,
    )


def _recipe(
    *,
    body_text: str,
    structured_data: RecipeV1StructuredData,
    confidence: RecipeConfidence,
) -> ExtractedRecipe:
    return ExtractedRecipe(
        item_type="recipe",
        title="t",
        summary=None,
        body_text=body_text,
        source_span_ids=["span_1"],
        structured_data=structured_data,
        confidence=confidence,
        warnings=[],
    )


def test_soft_warning_messages_covers_every_validate_soft_code() -> None:
    """Fire every soft rule across two candidates; the emitted-code set must
    exactly equal the map's key set (no missing entry, no stale entry).

    Two candidates because the rules are not co-firable on one:
    ``low_normalization_confidence`` needs ingredients (``no_ingredients``
    needs none) and too_short/too_long are an if/elif.
    """
    # A: no ingredients, no steps, low overall, low boundary, too short.
    candidate_a = _recipe(
        body_text="x",
        structured_data=_structured([], []),
        confidence=_confidence(0.0, 0.0),
    )
    # B: ingredients with a low normalization confidence, steps, too long.
    ingredient = ExtractedIngredient(
        position=1,
        raw_text="1 cup beans",
        quantity_text=None,
        quantity_value=None,
        unit_raw=None,
        unit_normalized=None,
        item_text="beans",
        item_normalized="beans",
        preparation=None,
        notes=None,
        confidence=IngredientConfidence(
            overall=1.0, quantity=1.0, unit=1.0, item=1.0, normalization=0.0
        ),
    )
    step = ExtractedStep(
        step_number=1,
        text="cook",
        source_span_ids=["span_1"],
        confidence=StepConfidence(overall=1.0, ordering=1.0),
    )
    candidate_b = _recipe(
        body_text="x" * 100,
        structured_data=_structured([ingredient], [step]),
        confidence=_confidence(1.0, 1.0),
    )

    emitted = {
        warning.code
        for candidate in (candidate_a, candidate_b)
        for warning in validate_soft(candidate, thresholds=_THRESHOLDS)
    }
    assert emitted == set(SOFT_WARNING_MESSAGES)
