"""Unit tests for the shared recipe text composition (Epic 22.1).

These helpers are the single definition of "what text does this item say",
consumed by both ``build_chunks`` and the ``body_text`` rebuild an edit
performs. The drift guard at the bottom is the point of the module.
"""

from __future__ import annotations

from typing import Any

from rag_recipes.ingestion.pipeline.chunking import build_chunks
from rag_recipes.ingestion.pipeline.composition import (
    compose_body_text,
    is_present,
    join_blocks,
    resolve_ingredients_text,
    resolve_steps_text,
)
from rag_recipes.storage.enums import ChunkType, KnowledgeItemStatus
from rag_recipes.storage.models.knowledge_item import KnowledgeItem


def _structured(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema": "recipe.v1",
        "yield": "Serves 4",
        "prep_time": None,
        "cook_time": None,
        "total_time": None,
        "ingredients_text": None,
        "ingredients": [
            {"position": 1, "raw_text": "2 tbsp olive oil"},
            {"position": 2, "raw_text": "1 onion, diced"},
        ],
        "steps_text": None,
        "steps": [
            {"step_number": 1, "text": "Heat the oil."},
            {"step_number": 2, "text": "Fry the onion."},
        ],
    }
    base.update(overrides)
    return base


def test_is_present_rejects_none_empty_and_whitespace() -> None:
    assert is_present("x")
    assert not is_present(None)
    assert not is_present("")
    assert not is_present("   \n\t ")


def test_resolvers_prefer_the_text_field_over_the_row_lists() -> None:
    structured = _structured(ingredients_text="whole blob", steps_text="step blob")
    assert resolve_ingredients_text(structured) == "whole blob"
    assert resolve_steps_text(structured) == "step blob"


def test_resolvers_fall_back_to_the_row_lists_when_text_is_blank() -> None:
    structured = _structured(ingredients_text="  ", steps_text=None)
    assert resolve_ingredients_text(structured) == "2 tbsp olive oil\n1 onion, diced"
    assert resolve_steps_text(structured) == "Heat the oil.\nFry the onion."


def test_resolvers_return_empty_string_for_missing_keys() -> None:
    assert resolve_ingredients_text({}) == ""
    assert resolve_steps_text({}) == ""


def test_join_blocks_skips_blank_blocks() -> None:
    assert join_blocks(["a", "", "  ", "b"]) == "a\n\nb"


def test_compose_body_text_joins_title_ingredients_steps() -> None:
    assert compose_body_text(title="Soup", structured=_structured()) == (
        "Soup\n\n2 tbsp olive oil\n1 onion, diced\n\nHeat the oil.\nFry the onion."
    )


def test_compose_body_text_matches_the_build_chunks_fallback() -> None:
    """The drift guard: one composition, so search and display cannot disagree.

    ``build_chunks`` falls back to composing ``recipe_full`` when an item has no
    ``body_text``; an edit rebuilds ``body_text`` with the same helper. If these
    two ever diverge, the corrected recipe and the indexed recipe say different
    things.
    """
    structured = _structured()
    item = KnowledgeItem(
        id="item_x",
        document_id="doc_x",
        extraction_run_id="run_x",
        source_version=1,
        item_type="recipe",
        title="Soup",
        normalized_title="soup",
        summary=None,
        body_text="",
        source_span_ids=["span_001"],
        structured_data=structured,
        confidence={"overall": 0.9},
        status=KnowledgeItemStatus.READY,
    )

    chunks = {chunk.chunk_type: chunk.text for chunk in build_chunks(item, category="recipes")}

    assert chunks[ChunkType.RECIPE_FULL] == compose_body_text(title="Soup", structured=structured)
