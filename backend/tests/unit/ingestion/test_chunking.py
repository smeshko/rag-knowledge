"""Unit tests for ingestion.pipeline.chunking.build_chunks (Phase 10.1).

Pure tests over detached ``KnowledgeItem`` instances — no DB, no session. Each
test constructs an item with ``_make_item`` and asserts on ``build_chunks``
output: chunk count/order, the empty-source skip rule, the structured-list
fallbacks, status gating, deterministic ``text_hash``, and the metadata block.
"""

from __future__ import annotations

import hashlib
from typing import Any

from rag_recipes.ingestion.pipeline.chunking import build_chunks
from rag_recipes.storage.enums import ChunkParentType, ChunkType, KnowledgeItemStatus
from rag_recipes.storage.models.knowledge_item import KnowledgeItem

CATEGORY = "recipes"


def _make_item(**overrides: Any) -> KnowledgeItem:
    """Build a detached READY KnowledgeItem (never added to a session)."""
    defaults: dict[str, Any] = {
        "id": "item_abc123",
        "document_id": "doc_abc123",
        "item_type": "recipe",
        "title": "Tomato and White Bean Soup",
        "summary": "A hearty weeknight soup.",
        "body_text": "Tomato and White Bean Soup\n\nA hearty weeknight soup.",
        "source_span_ids": ["span_1", "span_2"],
        "structured_data": {
            "ingredients_text": "2 tbsp olive oil\n1 onion",
            "ingredients": [
                {"raw_text": "2 tbsp olive oil"},
                {"raw_text": "1 onion, diced"},
            ],
            "steps_text": "1. Heat the oil.\n2. Add the onion.",
            "steps": [
                {"text": "Heat the oil in a large pot."},
                {"text": "Add the onion and cook until soft."},
            ],
        },
        "status": KnowledgeItemStatus.READY,
    }
    defaults.update(overrides)
    return KnowledgeItem(**defaults)


def _types(chunks: list[Any]) -> list[ChunkType]:
    return [c.chunk_type for c in chunks]


def test_all_present_yields_five_chunks_in_canonical_order() -> None:
    item = _make_item()
    chunks = build_chunks(item, category=CATEGORY)
    assert _types(chunks) == [
        ChunkType.RECIPE_TITLE,
        ChunkType.RECIPE_SUMMARY,
        ChunkType.RECIPE_INGREDIENTS,
        ChunkType.RECIPE_STEPS,
        ChunkType.RECIPE_FULL,
    ]
    by_type = {c.chunk_type: c.text for c in chunks}
    assert by_type[ChunkType.RECIPE_TITLE] == "Tomato and White Bean Soup"
    assert by_type[ChunkType.RECIPE_SUMMARY] == "A hearty weeknight soup."
    assert by_type[ChunkType.RECIPE_INGREDIENTS] == "2 tbsp olive oil\n1 onion"
    assert by_type[ChunkType.RECIPE_STEPS] == "1. Heat the oil.\n2. Add the onion."
    assert by_type[ChunkType.RECIPE_FULL] == item.body_text


def test_empty_summary_drops_only_the_summary_chunk() -> None:
    chunks = build_chunks(_make_item(summary=None), category=CATEGORY)
    assert ChunkType.RECIPE_SUMMARY not in _types(chunks)
    assert ChunkType.RECIPE_TITLE in _types(chunks)
    assert len(chunks) == 4


def test_whitespace_only_summary_is_skipped() -> None:
    chunks = build_chunks(_make_item(summary="   \n  "), category=CATEGORY)
    assert ChunkType.RECIPE_SUMMARY not in _types(chunks)


def test_empty_ingredients_drops_only_the_ingredients_chunk() -> None:
    structured = {"steps_text": "1. Heat the oil.", "steps": [{"text": "Heat."}]}
    chunks = build_chunks(_make_item(structured_data=structured), category=CATEGORY)
    assert ChunkType.RECIPE_INGREDIENTS not in _types(chunks)
    assert ChunkType.RECIPE_STEPS in _types(chunks)


def test_empty_steps_drops_only_the_steps_chunk() -> None:
    structured = {
        "ingredients_text": "2 tbsp olive oil",
        "ingredients": [{"raw_text": "2 tbsp olive oil"}],
    }
    chunks = build_chunks(_make_item(structured_data=structured), category=CATEGORY)
    assert ChunkType.RECIPE_STEPS not in _types(chunks)
    assert ChunkType.RECIPE_INGREDIENTS in _types(chunks)


def test_ingredients_fallback_from_structured_list() -> None:
    structured = {
        "ingredients": [
            {"raw_text": "2 tbsp olive oil"},
            {"raw_text": "1 onion, diced"},
        ],
        "steps_text": "1. Heat the oil.",
        "steps": [{"text": "Heat the oil."}],
    }
    chunks = build_chunks(_make_item(structured_data=structured), category=CATEGORY)
    by_type = {c.chunk_type: c.text for c in chunks}
    assert by_type[ChunkType.RECIPE_INGREDIENTS] == "2 tbsp olive oil\n1 onion, diced"


def test_steps_fallback_from_structured_list() -> None:
    structured = {
        "ingredients_text": "2 tbsp olive oil",
        "ingredients": [{"raw_text": "2 tbsp olive oil"}],
        "steps": [
            {"text": "Heat the oil in a large pot."},
            {"text": "Add the onion and cook until soft."},
        ],
    }
    chunks = build_chunks(_make_item(structured_data=structured), category=CATEGORY)
    by_type = {c.chunk_type: c.text for c in chunks}
    assert by_type[ChunkType.RECIPE_STEPS] == (
        "Heat the oil in a large pot.\nAdd the onion and cook until soft."
    )


def test_recipe_full_falls_back_to_title_ingredients_steps_concatenation() -> None:
    item = _make_item(body_text="   ")
    chunks = build_chunks(item, category=CATEGORY)
    by_type = {c.chunk_type: c.text for c in chunks}
    assert ChunkType.RECIPE_FULL in by_type
    assert by_type[ChunkType.RECIPE_FULL] == (
        "Tomato and White Bean Soup\n\n"
        "2 tbsp olive oil\n1 onion\n\n"
        "1. Heat the oil.\n2. Add the onion."
    )


def test_needs_review_item_yields_no_chunks() -> None:
    item = _make_item(status=KnowledgeItemStatus.NEEDS_REVIEW)
    assert build_chunks(item, category=CATEGORY) == []


def test_superseded_item_yields_no_chunks() -> None:
    item = _make_item(status=KnowledgeItemStatus.SUPERSEDED)
    assert build_chunks(item, category=CATEGORY) == []


def test_text_hash_is_sha256_of_text_and_deterministic() -> None:
    item = _make_item()
    first = build_chunks(item, category=CATEGORY)
    second = build_chunks(item, category=CATEGORY)
    for chunk in first:
        assert chunk.text_hash == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
    # Deterministic across calls: same text → same hash, per chunk type.
    assert {c.chunk_type: c.text_hash for c in first} == {
        c.chunk_type: c.text_hash for c in second
    }


def test_chunk_fields_match_parent_item() -> None:
    item = _make_item()
    for chunk in build_chunks(item, category=CATEGORY):
        assert chunk.document_id == item.document_id
        assert chunk.parent_type == ChunkParentType.KNOWLEDGE_ITEM
        assert chunk.parent_id == item.id
        assert chunk.source_span_ids == ["span_1", "span_2"]
        # A copy, not the same list object (whole-list assignment contract).
        assert chunk.source_span_ids is not item.source_span_ids


def test_metadata_block_is_populated() -> None:
    item = _make_item()
    for chunk in build_chunks(item, category=CATEGORY):
        assert chunk.chunk_metadata == {
            "category": CATEGORY,
            "item_type": "recipe",
            "title": "Tomato and White Bean Soup",
        }
