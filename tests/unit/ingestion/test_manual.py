"""Unit tests for ``authored_recipe`` — the pure half of ``ingestion.manual``.

No DB, no provider, no ``Settings``. What these assert is that a typed recipe
comes out in exactly the shape an *edited* one does: the same human-authored
row shape, the same derived text blocks, the same warning codes. The shelf
bootstrap is the other half and is exercised in
``tests/integration/test_knowledge_item_create.py``, where there is a database
for it to be idempotent against.
"""

from __future__ import annotations

from rag_recipes.ingestion.editing import RecipeEdit, apply_edit
from rag_recipes.ingestion.manual import authored_recipe, human_confidence
from rag_recipes.ingestion.pipeline.composition import compose_body_text
from rag_recipes.ingestion.validation import SoftValidationThresholds

THRESHOLDS = SoftValidationThresholds(
    min_overall_confidence=0.6,
    min_boundary_confidence=0.6,
    min_normalization_confidence=0.6,
    min_recipe_chars=50,
    max_recipe_chars=20000,
    assembly_min_ingredients=3,
    assembly_max_ingredients=12,
    assembly_max_chars=260,
)

INGREDIENTS = ["500 g ripe tomatoes", "2 cloves garlic", "3 tbsp olive oil"]
STEPS = [
    "Halve the tomatoes and salt them well.",
    "Roast at 200C with the garlic and oil until collapsing, about 40 minutes.",
    "Blitz smooth, then taste again for salt.",
]


def _authored(**overrides: object) -> object:
    kwargs: dict[str, object] = {
        "title": "Roast Tomato Soup",
        "summary": "A soup that tastes of the oven.",
        "yield_": "Serves 4",
        "prep_time": "10 minutes",
        "cook_time": "40 minutes",
        "total_time": "50 minutes",
        "ingredients": list(INGREDIENTS),
        "steps": list(STEPS),
        "thresholds": THRESHOLDS,
    }
    kwargs.update(overrides)
    return authored_recipe(**kwargs)  # type: ignore[arg-type]


def test_scalars_land_on_the_recipe_v1_payload() -> None:
    result = _authored()

    assert result.title == "Roast Tomato Soup"
    assert result.normalized_title == "roast tomato soup"
    assert result.summary == "A soup that tastes of the oven."
    assert result.structured_data["schema"] == "recipe.v1"
    assert result.structured_data["yield"] == "Serves 4"
    assert result.structured_data["prep_time"] == "10 minutes"
    assert result.structured_data["cook_time"] == "40 minutes"
    assert result.structured_data["total_time"] == "50 minutes"


def test_lines_take_the_human_authored_row_shape() -> None:
    """Every line is unmatched against an empty base, so every line is a
    human-authored row — parse nulled, confidences 1.0, ``edited`` set."""
    result = _authored()

    ingredients = result.structured_data["ingredients"]
    assert [row["position"] for row in ingredients] == [1, 2, 3]
    assert [row["raw_text"] for row in ingredients] == INGREDIENTS
    first = ingredients[0]
    assert first["edited"] is True
    assert first["quantity_value"] is None
    assert first["unit_normalized"] is None
    assert first["item_normalized"] is None
    assert first["confidence"] == {
        "overall": 1.0,
        "quantity": 1.0,
        "unit": 1.0,
        "item": 1.0,
        "normalization": 1.0,
    }

    steps = result.structured_data["steps"]
    assert [row["step_number"] for row in steps] == [1, 2, 3]
    assert [row["text"] for row in steps] == STEPS
    # A human wrote it; claiming a page cited it would be a lie.
    assert all(row["source_span_ids"] == [] for row in steps)


def test_text_blocks_and_body_are_composed_from_the_lines() -> None:
    result = _authored()

    assert result.structured_data["ingredients_text"] == "\n".join(INGREDIENTS)
    assert result.structured_data["steps_text"] == "\n".join(STEPS)
    assert result.body_text == compose_body_text(
        title="Roast Tomato Soup", structured=result.structured_data
    )
    # Not merely non-empty: an empty body_text costs the item its recipe_full
    # chunk, which is the chunk search leans on hardest.
    assert "Roast Tomato Soup" in result.body_text
    assert "500 g ripe tomatoes" in result.body_text
    assert "Blitz smooth" in result.body_text


def test_confidence_is_human_throughout() -> None:
    result = _authored()

    assert result.confidence == human_confidence()
    assert result.confidence["overall"] == 1.0
    assert result.confidence["fields"]["ingredients"] == 1.0


def test_human_confidence_is_a_fresh_dict_each_call() -> None:
    """It is written into a JSONB column; two rows must not share one object."""
    first = human_confidence()
    second = human_confidence()

    first["fields"]["title"] = 0.0

    assert second["fields"]["title"] == 1.0


def test_a_complete_recipe_carries_no_warnings() -> None:
    assert _authored().structured_data["warnings"] == []


def test_missing_sections_warn_exactly_as_an_extraction_would() -> None:
    """The codes come from ``validate_soft`` via ``warnings_for_item``, so a
    typed recipe with no method reads the same as an extracted one with none.

    Thirteen ingredients, one past ``assembly_max_ingredients``: a step-less
    recipe inside the assembly window is *exempt* from ``no_steps`` (next test),
    so a smaller list here would prove nothing about the wiring.
    """
    no_steps = _authored(steps=[], ingredients=[f"{n} g of ingredient {n}" for n in range(1, 14)])
    assert "no_steps" in no_steps.structured_data["warnings"]

    no_ingredients = _authored(ingredients=[])
    assert "no_ingredients" in no_ingredients.structured_data["warnings"]


def test_a_typed_assembly_recipe_is_exempt_from_no_steps() -> None:
    """A salad typed as three ingredients and no method is method-free by
    design, not truncated — ``validate_soft``'s assembly exemption, reached
    here for free because the create path defers to it rather than deciding
    for itself what a complete recipe looks like."""
    result = _authored(steps=[])

    assert "no_steps" not in result.structured_data["warnings"]
    assert "recipe_too_short" not in result.structured_data["warnings"]


def test_confidence_warnings_can_never_fire() -> None:
    """The two confidence-derived codes judge whether a recipe was cut out of a
    page correctly. No page was read, so neither has anything to say — and
    ``human_confidence`` is what keeps them quiet rather than luck."""
    warnings = _authored(ingredients=[], steps=[]).structured_data["warnings"]

    assert "low_overall_confidence" not in warnings
    assert "low_boundary_confidence" not in warnings
    assert "low_normalization_confidence" not in warnings


def test_an_empty_recipe_composes_to_its_title_alone() -> None:
    """Accepted, because ``PATCH`` accepts emptying both lists. It must still
    produce a chunkable body rather than an empty string."""
    result = _authored(ingredients=[], steps=[])

    assert result.body_text == "Roast Tomato Soup"
    assert result.structured_data["ingredients"] == []
    assert result.structured_data["steps"] == []


def test_output_matches_editing_the_same_lines_into_an_empty_item() -> None:
    """The claim the module is built on: creating a recipe and editing an empty
    one into the same text produce the same payload. If these ever diverge, a
    manual recipe has drifted into a shape the edit endpoint cannot round-trip.
    """
    created = _authored()
    edited = apply_edit(
        title="Roast Tomato Soup",
        summary="A soup that tastes of the oven.",
        body_text="",
        structured_data={"schema": "recipe.v1"},
        edit=RecipeEdit(
            yield_="Serves 4",
            prep_time="10 minutes",
            cook_time="40 minutes",
            total_time="50 minutes",
            ingredients=list(INGREDIENTS),
            steps=list(STEPS),
        ),
    )

    # `warnings` / `validation_notes` / `validation_thresholds` are the keys the
    # create path adds on top.
    assert {
        key: value
        for key, value in created.structured_data.items()
        if key not in {"warnings", "validation_notes", "validation_thresholds"}
    } == edited.structured_data
    assert created.title == edited.title
    assert created.normalized_title == edited.normalized_title
    assert created.body_text == edited.body_text
