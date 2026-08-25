"""Unit tests for the pure parts of the menu service.

Two functions carry the menu-specific correctness burden and are testable without a
database: ``_assign_across_courses`` (a dish must not fill two courses) and
``validate_menu_selection`` (the grounding + slot rules the model's output must
satisfy before it is trusted).
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_recipes.answers.context_pack import (
    ContextChunk,
    ContextCitation,
    ContextDocument,
    ContextItem,
    ContextPack,
)
from rag_recipes.menus.service import (
    _assign_across_courses,
    _CourseCandidates,
    validate_menu_selection,
)
from rag_recipes.menus.types import Course, MenuPlan
from rag_recipes.retrieval.types import (
    KnowledgeItemResult,
    MatchedChunkRef,
    ResultDocument,
    ResultItem,
)
from rag_recipes.storage.enums import ChunkType

# --- fixtures -----------------------------------------------------------------


def _item(item_id: str, score: float) -> KnowledgeItemResult:
    return KnowledgeItemResult(
        item=ResultItem(
            knowledge_item_id=item_id,
            item_type="recipe",
            title=f"Recipe {item_id}",
            summary=None,
            status="ready",
        ),
        document=ResultDocument(document_id="doc-1", title="Book", author=""),
        item_score=score,
        matched_chunks=[
            MatchedChunkRef(
                chunk_id=f"chunk-{item_id}", chunk_type=ChunkType.RECIPE_FULL, score=score
            )
        ],
        source_citations=[],
    )


def _pack_item(ctx: str, item_id: str, cite: str) -> ContextItem:
    return ContextItem(
        context_item_id=ctx,
        knowledge_item_id=item_id,
        title=f"Recipe {item_id}",
        summary=None,
        document=ContextDocument(document_id="doc-1", title="Book", author=""),
        matched_chunks=[
            ContextChunk(
                chunk_id=f"chunk-{item_id}", chunk_type="recipe_full", text="...", citation_id=cite
            )
        ],
        citations=[
            ContextCitation(citation_id=cite, source_span_id=f"span-{item_id}", label="page 1")
        ],
    )


PACK = ContextPack(
    query="menu",
    items=[
        _pack_item("ctx_1", "salad-item", "cite_1"),
        _pack_item("ctx_2", "dessert-item", "cite_2"),
    ],
)
PLAN = MenuPlan(
    theme="",
    courses=[Course(slot="salad", query="green salad"), Course(slot="dessert", query="mousse")],
)
SLOT_BY_ITEM = {"salad-item": "salad", "dessert-item": "dessert"}


def _selection(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "menu": {"title": "Dinner", "text": "They balance.", "citations": ["cite_1"]},
        "courses": [
            {
                "slot": "salad",
                "knowledge_item_id": "salad-item",
                "reason": "light",
                "citation_ids": ["cite_1"],
            },
            {
                "slot": "dessert",
                "knowledge_item_id": "dessert-item",
                "reason": "rich",
                "citation_ids": ["cite_2"],
            },
        ],
    }
    base.update(overrides)
    return base


def _validate(selection: dict[str, Any]) -> list[str]:
    return validate_menu_selection(selection, PACK, plan=PLAN, slot_by_item_id=SLOT_BY_ITEM)


# --- cross-course assignment --------------------------------------------------


def test_a_dish_ranking_for_two_courses_is_assigned_to_only_one() -> None:
    """The multi-course version of the duplicate-result bug the layer exists to fix."""
    shared_high = _item("shared", 0.9)
    shared_low = _item("shared", 0.4)
    buckets = [
        _CourseCandidates(Course("salad", "q1"), [shared_low, _item("salad-only", 0.3)]),
        _CourseCandidates(Course("dessert", "q2"), [shared_high, _item("dessert-only", 0.2)]),
    ]

    assigned = _assign_across_courses(buckets, cap=2)
    salad_ids = [i.item.knowledge_item_id for i in assigned[0].items]
    dessert_ids = [i.item.knowledge_item_id for i in assigned[1].items]

    # Dessert scored it higher (0.9 > 0.4), so dessert wins the contested dish.
    assert "shared" in dessert_ids
    assert "shared" not in salad_ids
    assert set(salad_ids) & set(dessert_ids) == set()


def test_a_losing_course_keeps_a_full_candidate_list_from_the_over_fetch() -> None:
    buckets = [
        _CourseCandidates(Course("salad", "q1"), [_item("shared", 0.1), _item("a", 0.05)]),
        _CourseCandidates(Course("dessert", "q2"), [_item("shared", 0.9)]),
    ]
    assigned = _assign_across_courses(buckets, cap=1)

    assert [i.item.knowledge_item_id for i in assigned[0].items] == ["a"]
    assert [i.item.knowledge_item_id for i in assigned[1].items] == ["shared"]


def test_assignment_caps_each_course_and_keeps_score_order() -> None:
    buckets = [
        _CourseCandidates(
            Course("salad", "q1"), [_item("a", 0.1), _item("b", 0.9), _item("c", 0.5)]
        )
    ]
    assigned = _assign_across_courses(buckets, cap=2)

    assert [i.item.knowledge_item_id for i in assigned[0].items] == ["b", "c"]


def test_assignment_is_deterministic_for_tied_scores() -> None:
    buckets = [
        _CourseCandidates(Course("salad", "q1"), [_item("b", 0.5), _item("a", 0.5)]),
        _CourseCandidates(Course("dessert", "q2"), [_item("a", 0.5), _item("b", 0.5)]),
    ]
    first = _assign_across_courses(buckets, cap=1)
    second = _assign_across_courses(buckets, cap=1)

    assert [i.item.knowledge_item_id for i in first[0].items] == [
        i.item.knowledge_item_id for i in second[0].items
    ]
    # Ties break on course order, so the earlier course claims the shared dish.
    assert first[0].items[0].item.knowledge_item_id == "a"


def test_assignment_preserves_course_order_and_empty_courses() -> None:
    buckets = [
        _CourseCandidates(Course("salad", "q1"), []),
        _CourseCandidates(Course("dessert", "q2"), [_item("d", 0.5)]),
    ]
    assigned = _assign_across_courses(buckets, cap=2)

    assert [b.course.slot for b in assigned] == ["salad", "dessert"]
    assert assigned[0].items == []


# --- selection validation -----------------------------------------------------


def test_a_well_formed_selection_validates() -> None:
    assert _validate(_selection()) == []


def test_unknown_citation_is_rejected() -> None:
    selection = _selection()
    selection["courses"][0]["citation_ids"] = ["cite_99"]
    assert any("unknown citation_id" in e for e in _validate(selection))


def test_citation_bound_to_another_item_is_rejected() -> None:
    """Misattribution: citing the dessert's span to justify the salad."""
    selection = _selection()
    selection["courses"][0]["citation_ids"] = ["cite_2"]
    assert any("belongs to a different item" in e for e in _validate(selection))


def test_picking_an_item_from_another_slot_is_rejected() -> None:
    """The id is in the pack, but it is the dessert course's candidate."""
    selection = _selection()
    selection["courses"][0]["knowledge_item_id"] = "dessert-item"
    selection["courses"][0]["citation_ids"] = ["cite_2"]
    errors = _validate(selection)
    assert any("is a candidate for slot" in e for e in errors)


def test_serving_one_dish_in_two_courses_is_rejected() -> None:
    selection = _selection()
    selection["courses"][1]["knowledge_item_id"] = "salad-item"
    selection["courses"][1]["citation_ids"] = ["cite_1"]
    assert any("more than one course" in e for e in _validate(selection))


def test_unplanned_slot_is_rejected() -> None:
    selection = _selection()
    selection["courses"][0]["slot"] = "amuse-bouche"
    assert any("unplanned slot" in e for e in _validate(selection))


def test_filling_the_same_slot_twice_is_rejected() -> None:
    selection = _selection()
    selection["courses"][1]["slot"] = "salad"
    assert any("twice" in e for e in _validate(selection))


def test_a_fillable_slot_left_unfilled_is_rejected() -> None:
    """A silently dropped course is the failure the endpoint exists to prevent."""
    selection = _selection()
    selection["courses"] = selection["courses"][:1]
    assert any("left unfilled" in e for e in _validate(selection))


def test_a_course_without_citations_is_rejected() -> None:
    selection = _selection()
    selection["courses"][0]["citation_ids"] = []
    assert any("no citations" in e for e in _validate(selection))


def test_a_selection_with_no_valid_citation_anywhere_is_rejected() -> None:
    selection = {"menu": {"title": "", "text": "", "citations": []}, "courses": []}
    errors = validate_menu_selection(selection, PACK, plan=PLAN, slot_by_item_id=SLOT_BY_ITEM)
    assert any("no valid citations" in e for e in errors)


@pytest.mark.parametrize(
    "selection",
    [
        {"courses": []},  # no menu block
        {"menu": "not an object", "courses": []},
        {"menu": {"title": "", "text": "", "citations": []}, "courses": "not a list"},
        {"menu": {"title": "", "text": "", "citations": []}, "courses": ["not an object"]},
        # An unhashable nested citation must not reach the membership lookup.
        {"menu": {"title": "", "text": "", "citations": [["nested"]]}, "courses": []},
    ],
)
def test_structurally_malformed_output_is_a_validation_error_not_a_crash(
    selection: dict[str, Any],
) -> None:
    errors = validate_menu_selection(selection, PACK, plan=PLAN, slot_by_item_id=SLOT_BY_ITEM)
    assert errors
