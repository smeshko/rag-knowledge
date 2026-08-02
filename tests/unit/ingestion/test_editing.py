"""Unit tests for the pure edit layer in ``ingestion.editing`` (Epic 22.1).

No DB, no provider, no ``Settings``: rows are built exactly the way
``pipeline/persist`` builds them (``model_dump(mode="json", by_alias=True)`` plus
the soft-validation warning codes), so what these tests feed ``apply_edit`` and
``warnings_for_item`` is byte-for-byte what Postgres holds.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from types import ModuleType
from typing import Any

import pytest

import rag_recipes.ingestion.editing as editing_module
from rag_recipes.ingestion.editing import (
    UNSET,
    RecipeEdit,
    apply_edit,
    warnings_for_item,
)
from rag_recipes.ingestion.pipeline.composition import compose_body_text
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

THRESHOLDS = SoftValidationThresholds(
    min_overall_confidence=0.6,
    min_boundary_confidence=0.6,
    min_normalization_confidence=0.5,
    min_recipe_chars=80,
    max_recipe_chars=600,
)

LONG_BODY = "Tomato Soup\n\n" + ("a rich, slow-simmered tomato soup for a cold evening. " * 3)


# --------------------------------------------------------------------------- #
# Builders — mirror pipeline/persist so a fixture is a real persisted shape.
# --------------------------------------------------------------------------- #


def _ingredient(
    position: int = 1,
    raw_text: str = "2 tbsp olive oil",
    normalization: float = 0.9,
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
        confidence=IngredientConfidence(
            overall=0.9, quantity=0.9, unit=0.9, item=0.9, normalization=normalization
        ),
    )


def _step(step_number: int = 1, text: str = "Heat the oil in a large pot.") -> ExtractedStep:
    return ExtractedStep(
        step_number=step_number,
        text=text,
        source_span_ids=[f"span_{step_number:03d}"],
        confidence=StepConfidence(overall=0.9, ordering=0.9),
    )


def _recipe(
    *,
    title: str = "Tomato Soup",
    summary: str | None = "A simple soup.",
    body_text: str = LONG_BODY,
    ingredients: list[ExtractedIngredient] | None = None,
    steps: list[ExtractedStep] | None = None,
    overall: float = 0.9,
    boundary: float = 0.9,
) -> ExtractedRecipe:
    return ExtractedRecipe(
        item_type="recipe",
        title=title,
        summary=summary,
        body_text=body_text,
        source_span_ids=["span_001", "span_002"],
        structured_data=RecipeV1StructuredData(
            schema_="recipe.v1",
            yield_="Serves 4",
            prep_time=None,
            cook_time=None,
            total_time=None,
            ingredients_text=None,
            ingredients=[_ingredient(), _ingredient(position=2, raw_text="1 onion, diced")]
            if ingredients is None
            else ingredients,
            steps_text=None,
            steps=[_step(), _step(step_number=2, text="Add the tomatoes.")]
            if steps is None
            else steps,
        ),
        confidence=RecipeConfidence(
            overall=overall,
            boundary=boundary,
            fields=RecipeFieldConfidence(
                title=0.9, summary=0.9, yield_=0.9, ingredients=0.9, steps=0.9
            ),
        ),
        warnings=[],
    )


def _persist(extracted: ExtractedRecipe) -> dict[str, Any]:
    """Build the row a clean ingest of ``extracted`` would write (persist.py)."""
    warnings = validate_soft(extracted, thresholds=THRESHOLDS)
    return {
        "title": extracted.title,
        "summary": extracted.summary,
        "body_text": extracted.body_text,
        "source_span_ids": list(extracted.source_span_ids),
        "structured_data": {
            **extracted.structured_data.model_dump(mode="json", by_alias=True),
            "warnings": [w.code for w in warnings],
        },
        "confidence": extracted.confidence.model_dump(mode="json", by_alias=True),
    }


def _edit_fields(row: dict[str, Any]) -> dict[str, Any]:
    """The subset of a row that ``apply_edit`` accepts."""
    return {
        "title": row["title"],
        "summary": row["summary"],
        "body_text": row["body_text"],
        "structured_data": row["structured_data"],
    }


def _raw_texts(structured: dict[str, Any]) -> list[str]:
    return [row["raw_text"] for row in structured["ingredients"]]


# --------------------------------------------------------------------------- #
# apply_edit — scalar fields
# --------------------------------------------------------------------------- #


def test_an_empty_edit_changes_nothing() -> None:
    row = _persist(_recipe())

    result = apply_edit(**_edit_fields(row), edit=RecipeEdit())

    assert RecipeEdit().is_empty()
    assert result.title == row["title"]
    assert result.summary == row["summary"]
    assert result.body_text == row["body_text"]
    assert result.structured_data == row["structured_data"]
    assert result.body_text_rebuilt is False


def test_title_edit_recomputes_normalized_title() -> None:
    row = _persist(_recipe())

    result = apply_edit(**_edit_fields(row), edit=RecipeEdit(title="  Roast   Tomato Soup "))

    assert result.title == "  Roast   Tomato Soup "
    assert result.normalized_title == "roast tomato soup"


def test_body_text_is_byte_identical_after_a_title_only_edit() -> None:
    """LLM prose that lives only in ``body_text`` survives a title fix."""
    row = _persist(_recipe())

    result = apply_edit(**_edit_fields(row), edit=RecipeEdit(title="Roast Tomato Soup"))

    assert result.body_text == row["body_text"]
    assert result.body_text_rebuilt is False


def test_summary_can_be_explicitly_cleared_and_absent_is_not_null() -> None:
    row = _persist(_recipe(summary="A simple soup."))

    cleared = apply_edit(**_edit_fields(row), edit=RecipeEdit(summary=None))
    absent = apply_edit(**_edit_fields(row), edit=RecipeEdit())

    assert cleared.summary is None
    assert absent.summary == "A simple soup."
    assert RecipeEdit().summary is UNSET


def test_scalar_metadata_fields_write_through_to_structured_data() -> None:
    row = _persist(_recipe())

    result = apply_edit(
        **_edit_fields(row),
        edit=RecipeEdit(yield_="Serves 6", prep_time="15 min", cook_time=None),
    )

    assert result.structured_data["yield"] == "Serves 6"
    assert result.structured_data["prep_time"] == "15 min"
    assert result.structured_data["cook_time"] is None
    # Untouched keys pass through, including ones this layer knows nothing about.
    assert result.structured_data["total_time"] == row["structured_data"]["total_time"]
    assert result.structured_data["warnings"] == row["structured_data"]["warnings"]


# --------------------------------------------------------------------------- #
# apply_edit — ingredient and step rows
# --------------------------------------------------------------------------- #


def test_changed_ingredient_line_is_reset_and_siblings_stay_byte_identical() -> None:
    row = _persist(_recipe())
    original = row["structured_data"]["ingredients"]

    result = apply_edit(
        **_edit_fields(row),
        edit=RecipeEdit(ingredients=["2 tbsp olive oil", "2 onions, finely diced"]),
    )
    rows = result.structured_data["ingredients"]

    assert rows[0] == original[0]  # untouched line: byte-identical
    edited = rows[1]
    assert edited["raw_text"] == "2 onions, finely diced"
    assert edited["edited"] is True
    assert edited["position"] == 2
    for parsed in ("quantity_text", "quantity_value", "unit_raw", "unit_normalized"):
        assert edited[parsed] is None
    for parsed in ("item_text", "item_normalized", "preparation", "notes"):
        assert edited[parsed] is None
    assert edited["confidence"] == {
        "overall": 1.0,
        "quantity": 1.0,
        "unit": 1.0,
        "item": 1.0,
        "normalization": 1.0,
    }


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        pytest.param(
            ["2 tbsp olive oil", "1 onion, diced", "1 tsp salt"],
            ["2 tbsp olive oil", "1 onion, diced", "1 tsp salt"],
            id="add",
        ),
        pytest.param(["1 onion, diced"], ["1 onion, diced"], id="remove"),
        pytest.param(
            ["1 onion, diced", "2 tbsp olive oil"],
            ["1 onion, diced", "2 tbsp olive oil"],
            id="reorder",
        ),
    ],
)
def test_ingredient_positions_are_contiguous_and_follow_list_order(
    lines: list[str], expected: list[str]
) -> None:
    row = _persist(_recipe())

    result = apply_edit(**_edit_fields(row), edit=RecipeEdit(ingredients=lines))
    rows = result.structured_data["ingredients"]

    assert [r["raw_text"] for r in rows] == expected
    assert [r["position"] for r in rows] == list(range(1, len(expected) + 1))


def test_a_pure_reorder_keeps_every_row_apart_from_its_position() -> None:
    """Moving a line is not authoring it — the parse and confidences survive."""
    row = _persist(_recipe())
    original = {r["raw_text"]: r for r in row["structured_data"]["ingredients"]}

    result = apply_edit(
        **_edit_fields(row),
        edit=RecipeEdit(ingredients=["1 onion, diced", "2 tbsp olive oil"]),
    )

    for moved in result.structured_data["ingredients"]:
        before = original[moved["raw_text"]]
        assert "edited" not in moved
        assert {k: v for k, v in moved.items() if k != "position"} == {
            k: v for k, v in before.items() if k != "position"
        }


def test_duplicate_lines_are_matched_one_for_one() -> None:
    row = _persist(
        _recipe(ingredients=[_ingredient(), _ingredient(position=2, raw_text="2 tbsp olive oil")])
    )

    result = apply_edit(
        **_edit_fields(row),
        edit=RecipeEdit(ingredients=["2 tbsp olive oil", "2 tbsp olive oil"]),
    )

    assert all("edited" not in r for r in result.structured_data["ingredients"])


def test_added_step_carries_no_span_provenance_and_untouched_steps_keep_theirs() -> None:
    row = _persist(_recipe())
    original = row["structured_data"]["steps"]

    result = apply_edit(
        **_edit_fields(row),
        edit=RecipeEdit(
            steps=["Heat the oil in a large pot.", "Add the tomatoes.", "Simmer for 20 minutes."]
        ),
    )
    rows = result.structured_data["steps"]

    assert rows[0] == original[0]
    assert rows[1] == original[1]
    added = rows[2]
    assert added["step_number"] == 3
    assert added["text"] == "Simmer for 20 minutes."
    assert added["source_span_ids"] == []
    assert added["edited"] is True
    assert added["confidence"] == {"overall": 1.0, "ordering": 1.0}


def test_step_numbers_are_renumbered_from_list_order() -> None:
    row = _persist(_recipe())

    result = apply_edit(
        **_edit_fields(row),
        edit=RecipeEdit(steps=["Add the tomatoes.", "Heat the oil in a large pot."]),
    )

    assert [s["step_number"] for s in result.structured_data["steps"]] == [1, 2]
    assert [s["text"] for s in result.structured_data["steps"]] == [
        "Add the tomatoes.",
        "Heat the oil in a large pot.",
    ]


# --------------------------------------------------------------------------- #
# apply_edit — body_text rebuild
# --------------------------------------------------------------------------- #


def test_body_text_is_rebuilt_when_ingredient_lines_change() -> None:
    row = _persist(_recipe())

    result = apply_edit(
        **_edit_fields(row),
        edit=RecipeEdit(ingredients=["2 tbsp olive oil", "3 ripe tomatoes"]),
    )

    assert result.body_text_rebuilt is True
    assert result.body_text != row["body_text"]
    assert "3 ripe tomatoes" in result.body_text
    assert result.body_text == compose_body_text(
        title=result.title, structured=result.structured_data
    )


def test_editing_lines_refreshes_the_precomputed_text_blocks() -> None:
    """``ingredients_text`` wins over the row list everywhere text is resolved.

    Leaving it at the model's pre-edit blob would index the old ingredients under
    a corrected recipe — the exact failure this epic exists to prevent.
    """
    extracted = _recipe()
    row = _persist(extracted)
    row["structured_data"]["ingredients_text"] = "2 tbsp olive oil\n1 onion, diced"

    result = apply_edit(
        **_edit_fields(row),
        edit=RecipeEdit(ingredients=["2 tbsp olive oil", "3 ripe tomatoes"]),
    )

    assert result.structured_data["ingredients_text"] == "2 tbsp olive oil\n3 ripe tomatoes"
    assert "1 onion, diced" not in result.body_text


def test_editing_one_list_leaves_the_other_list_text_block_untouched() -> None:
    """An ingredients-only edit must not wipe the method prose.

    ``steps_text`` wins over the ``steps`` rows wherever text is resolved, so an
    item flagged ``no_steps`` keeps its whole method in that blob. Rewriting it
    from an empty row list would destroy the item's steps — content the edit
    never named.
    """
    row = _persist(_recipe(steps=[]))
    row["structured_data"]["steps_text"] = "1. Preheat the oven.\n2. Roast for 40 minutes."

    result = apply_edit(
        **_edit_fields(row),
        edit=RecipeEdit(ingredients=["2 tbsp olive oil", "3 ripe tomatoes"]),
    )

    assert result.structured_data["steps_text"] == (
        "1. Preheat the oven.\n2. Roast for 40 minutes."
    )
    assert "Roast for 40 minutes" in result.body_text


def test_editing_steps_leaves_the_ingredients_text_block_untouched() -> None:
    """The symmetric case: section headings in ``ingredients_text`` survive."""
    row = _persist(_recipe())
    row["structured_data"]["ingredients_text"] = (
        "For the soup:\n2 tbsp olive oil\n\nFor the garnish:\nchives"
    )

    result = apply_edit(
        **_edit_fields(row),
        edit=RecipeEdit(steps=["Heat the oil in a large pot.", "Simmer for 20 minutes."]),
    )

    assert result.structured_data["ingredients_text"] == (
        "For the soup:\n2 tbsp olive oil\n\nFor the garnish:\nchives"
    )
    assert "For the garnish:" in result.body_text


def test_the_result_shares_no_mutable_state_with_the_input() -> None:
    """The caller builds a pre-edit snapshot from the same row it passes in.

    If the two shared a nested dict or list, a later in-place tweak of the new
    value would silently rewrite the snapshot of the original extraction.
    """
    row = _persist(_recipe())
    structured_in = row["structured_data"]

    result = apply_edit(**_edit_fields(row), edit=RecipeEdit(title="Renamed"))
    out = result.structured_data

    assert out is not structured_in
    assert out["ingredients"] is not structured_in["ingredients"]
    assert out["steps"] is not structured_in["steps"]
    assert out["ingredients"][0]["confidence"] is not structured_in["ingredients"][0]["confidence"]

    out["ingredients"][0]["confidence"]["overall"] = 0.0
    out["steps"][0]["source_span_ids"].append("span_999")
    assert structured_in["ingredients"][0]["confidence"]["overall"] == 0.9
    assert structured_in["steps"][0]["source_span_ids"] == ["span_001"]


def test_reordered_rows_are_copies_too() -> None:
    row = _persist(_recipe())
    structured_in = row["structured_data"]

    result = apply_edit(
        **_edit_fields(row),
        edit=RecipeEdit(ingredients=["1 onion, diced", "2 tbsp olive oil"]),
    )

    moved = result.structured_data["ingredients"][1]
    assert moved["raw_text"] == "2 tbsp olive oil"
    assert moved["confidence"] is not structured_in["ingredients"][0]["confidence"]


def test_resubmitting_identical_lines_does_not_rebuild_the_body() -> None:
    row = _persist(_recipe())

    result = apply_edit(
        **_edit_fields(row),
        edit=RecipeEdit(ingredients=_raw_texts(row["structured_data"])),
    )

    assert result.body_text_rebuilt is False
    assert result.body_text == row["body_text"]


# --------------------------------------------------------------------------- #
# warnings_for_item — round-trip fidelity
# --------------------------------------------------------------------------- #

_ROUND_TRIP_CASES: list[tuple[str, ExtractedRecipe]] = [
    ("clean", _recipe()),
    ("no_ingredients", _recipe(ingredients=[])),
    ("no_steps", _recipe(steps=[])),
    ("empty_and_short", _recipe(ingredients=[], steps=[], body_text="Soup")),
    ("low_overall", _recipe(overall=0.1)),
    ("low_boundary", _recipe(boundary=0.1)),
    ("low_overall_and_boundary", _recipe(overall=0.2, boundary=0.2)),
    ("too_short", _recipe(body_text="Tomato Soup")),
    ("too_long", _recipe(body_text="x" * 900)),
    ("low_normalization", _recipe(ingredients=[_ingredient(normalization=0.1)])),
    (
        "low_normalization_and_boundary",
        _recipe(ingredients=[_ingredient(normalization=0.1)], boundary=0.1),
    ),
]


@pytest.mark.parametrize(
    "extracted", [c[1] for c in _ROUND_TRIP_CASES], ids=[c[0] for c in _ROUND_TRIP_CASES]
)
def test_round_trip_fidelity_over_a_persisted_row(extracted: ExtractedRecipe) -> None:
    """The adapter is faithful, not merely plausible.

    Over an unedited row, re-deriving the warnings from what Postgres holds must
    reproduce exactly the codes the ingest pipeline stored — otherwise an edit
    would silently rewrite flags it never touched.
    """
    row = _persist(extracted)

    recomputed = warnings_for_item(
        title=row["title"],
        summary=row["summary"],
        body_text=row["body_text"],
        source_span_ids=row["source_span_ids"],
        structured_data=row["structured_data"],
        confidence=row["confidence"],
        thresholds=THRESHOLDS,
    )

    assert recomputed == row["structured_data"]["warnings"]


def test_round_trip_cases_cover_every_soft_warning_code() -> None:
    """Guard: a new soft-validation code must gain a fidelity case here."""
    seen = {
        code
        for _, extracted in _ROUND_TRIP_CASES
        for code in (w.code for w in validate_soft(extracted, thresholds=THRESHOLDS))
    }
    assert seen == {
        "no_ingredients",
        "no_steps",
        "low_overall_confidence",
        "low_boundary_confidence",
        "recipe_too_short",
        "recipe_too_long",
        "low_normalization_confidence",
    }


# --------------------------------------------------------------------------- #
# warnings_for_item — what an edit clears and what it cannot
# --------------------------------------------------------------------------- #


def _warnings_after(row: dict[str, Any], edit: RecipeEdit) -> list[str]:
    result = apply_edit(**_edit_fields(row), edit=edit)
    return warnings_for_item(
        title=result.title,
        summary=result.summary,
        body_text=result.body_text,
        source_span_ids=row["source_span_ids"],
        structured_data=result.structured_data,
        confidence=row["confidence"],
        thresholds=THRESHOLDS,
    )


def test_content_warnings_clear_when_the_edit_resolves_them() -> None:
    row = _persist(_recipe(ingredients=[], steps=[], body_text="Soup"))
    assert set(row["structured_data"]["warnings"]) == {
        "no_ingredients",
        "no_steps",
        "recipe_too_short",
    }

    after = _warnings_after(
        row,
        RecipeEdit(
            ingredients=["2 tbsp olive oil", "3 ripe tomatoes, roughly chopped", "1 onion, diced"],
            steps=[
                "Heat the oil in a large pot over a medium flame.",
                "Add the onion and cook until soft, about ten minutes.",
                "Add the tomatoes and simmer for twenty minutes.",
            ],
        ),
    )

    assert after == []


def test_confidence_warnings_survive_every_content_edit() -> None:
    """Retyping a line does not attest that the recipe was cut out correctly."""
    row = _persist(_recipe(overall=0.1, boundary=0.1, ingredients=[], steps=[]))

    after = _warnings_after(
        row,
        RecipeEdit(
            ingredients=["2 tbsp olive oil", "3 ripe tomatoes"],
            steps=["Heat the oil.", "Add the tomatoes and simmer for twenty minutes."],
        ),
    )

    assert "no_ingredients" not in after
    assert "no_steps" not in after
    assert set(after) == {"low_overall_confidence", "low_boundary_confidence"}


def test_low_normalization_clears_only_once_every_low_line_is_edited() -> None:
    row = _persist(
        _recipe(
            ingredients=[
                _ingredient(position=1, raw_text="2 tbsp olive oil", normalization=0.1),
                _ingredient(position=2, raw_text="1 glug of the good stuff", normalization=0.2),
            ]
        )
    )
    assert "low_normalization_confidence" in row["structured_data"]["warnings"]

    partial = _warnings_after(
        row, RecipeEdit(ingredients=["2 tbsp olive oil", "1 tbsp sherry vinegar"])
    )
    full = _warnings_after(
        row, RecipeEdit(ingredients=["2 tbsp extra-virgin olive oil", "1 tbsp sherry vinegar"])
    )

    assert "low_normalization_confidence" in partial
    assert "low_normalization_confidence" not in full


def test_a_missing_confidence_column_invents_no_warnings() -> None:
    row = _persist(_recipe())

    recomputed = warnings_for_item(
        title=row["title"],
        summary=row["summary"],
        body_text=row["body_text"],
        source_span_ids=row["source_span_ids"],
        structured_data=row["structured_data"],
        confidence=None,
        thresholds=THRESHOLDS,
    )

    assert recomputed == []


def test_warnings_are_recomputed_not_carried_forward() -> None:
    """A stale flag on the row must not survive its own repair."""
    row = _persist(_recipe(ingredients=[]))
    row["structured_data"]["warnings"] = ["no_ingredients", "llm_said_something_odd"]

    after = _warnings_after(
        row,
        RecipeEdit(
            ingredients=["2 tbsp olive oil", "3 ripe tomatoes, roughly chopped", "1 onion, diced"]
        ),
    )

    assert after == []


# --------------------------------------------------------------------------- #
# Purity contract
# --------------------------------------------------------------------------- #


_FORBIDDEN_ROOTS = ("sqlalchemy", "arq", "redis", "fastapi", "httpx")
_FORBIDDEN_MODULES = ("rag_recipes.config", "rag_recipes.providers", "rag_recipes.storage")
# The declared home of the recipe.v1 models (Epic 9) also holds `run_extraction`,
# so it carries a session import. `validation.py` — the module this purity
# contract mirrors — depends on it for exactly the same reason, so it is the
# boundary of the pure layer rather than a violation of it.
_MODEL_HOME = (
    "rag_recipes.ingestion.pipeline.extraction",
    "rag_recipes.ingestion.pipeline.windows",
)


def _direct_imports(module: ModuleType) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_editing_module_imports_no_session_provider_or_settings() -> None:
    """``editing`` is pure, on the same contract as ``validation``.

    Walked **transitively** across first-party modules: importing a pure-looking
    helper out of a module that itself drags in ``get_settings`` would satisfy a
    direct-imports-only check while breaking the contract in fact — which is
    exactly how ``normalize_title`` was first wired.
    """
    seen: set[str] = set()
    queue = [editing_module]
    offenders: list[str] = []
    while queue:
        module = queue.pop()
        if module.__name__ in seen:
            continue
        seen.add(module.__name__)
        for name in _direct_imports(module):
            if name.split(".")[0] in _FORBIDDEN_ROOTS or name.startswith(_FORBIDDEN_MODULES):
                offenders.append(f"{module.__name__} -> {name}")
            elif name.startswith("rag_recipes.") and not name.startswith(_MODEL_HOME):
                queue.append(importlib.import_module(name))

    assert offenders == []


# --------------------------------------------------------------------------- #
# Robustness — a raise here would be a 500 on the edit endpoint
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda s: s["ingredients"][0].update(position=None), id="null-position"),
        pytest.param(lambda s: s["ingredients"][0].update(raw_text=None), id="null-raw-text"),
        pytest.param(lambda s: s["steps"][0].update(step_number=None), id="null-step-number"),
        pytest.param(lambda s: s["steps"][0].update(text=None), id="null-step-text"),
        pytest.param(lambda s: s["steps"][0].update(source_span_ids=[None]), id="junk-span-id"),
        pytest.param(lambda s: s.update(schema=None), id="null-schema"),
        pytest.param(lambda s: s.update(**{"yield": 4}), id="non-string-yield"),
        pytest.param(lambda s: s.update(ingredients="not a list"), id="ingredients-not-a-list"),
        pytest.param(lambda s: s.update(confidence_junk=object()), id="unknown-key"),
    ],
)
def test_junk_in_a_persisted_row_never_raises(mutate: Any) -> None:
    row = _persist(_recipe())
    mutate(row["structured_data"])

    warnings = warnings_for_item(
        title=row["title"],
        summary=row["summary"],
        body_text=row["body_text"],
        source_span_ids=row["source_span_ids"],
        structured_data=row["structured_data"],
        confidence=row["confidence"],
        thresholds=THRESHOLDS,
    )
    edited = apply_edit(**_edit_fields(row), edit=RecipeEdit(steps=["A new step."]))

    assert isinstance(warnings, list)
    assert isinstance(edited.body_text, str)
