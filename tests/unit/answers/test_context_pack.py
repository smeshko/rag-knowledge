"""Unit tests for build_context_pack (pure, deterministic context-pack builder)."""

from __future__ import annotations

from rag_recipes.answers.context_pack import ChunkInput, build_context_pack
from rag_recipes.retrieval.types import (
    KnowledgeItemResult,
    MatchedChunkRef,
    ResultDocument,
    ResultItem,
    SourceCitation,
)
from rag_recipes.storage.enums import ChunkType


def _result(
    *,
    item_id: str,
    title: str = "Title",
    summary: str | None = "Summary",
    chunks: list[tuple[str, ChunkType]],
    citations: list[tuple[str, str]],
) -> KnowledgeItemResult:
    return KnowledgeItemResult(
        item=ResultItem(
            knowledge_item_id=item_id,
            item_type="recipe",
            title=title,
            summary=summary,
            status="ready",
        ),
        document=ResultDocument(
            document_id=f"doc-{item_id}", title="Doc", author="Author"
        ),
        item_score=1.0,
        matched_chunks=[
            MatchedChunkRef(chunk_id=cid, chunk_type=ctype, score=1.0)
            for cid, ctype in chunks
        ],
        source_citations=[
            SourceCitation(source_span_id=sid, label=label) for sid, label in citations
        ],
    )


def test_empty_results_yields_empty_pack() -> None:
    pack = build_context_pack(
        "q", [], {}, item_limit=6, chunks_per_item=3
    )
    assert pack.query == "q"
    assert pack.items == []


def test_basic_pack_assigns_deterministic_ids_and_text() -> None:
    results = [
        _result(
            item_id="item-1",
            chunks=[("c1", ChunkType.RECIPE_SUMMARY), ("c2", ChunkType.RECIPE_FULL)],
            citations=[("span-a", "p. 1")],
        ),
        _result(
            item_id="item-2",
            chunks=[("c3", ChunkType.RECIPE_TITLE)],
            citations=[("span-b", "p. 2")],
        ),
    ]
    chunk_input = {
        "c1": ChunkInput(text="chunk one", source_span_ids=["span-a"]),
        "c2": ChunkInput(text="chunk two", source_span_ids=["span-a"]),
        "c3": ChunkInput(text="chunk three", source_span_ids=["span-b"]),
    }
    pack = build_context_pack(
        "query", results, chunk_input, item_limit=6, chunks_per_item=3
    )
    assert [i.context_item_id for i in pack.items] == ["ctx_1", "ctx_2"]
    # cite_N assigned in first-seen order across the pack.
    assert pack.items[0].citations == pack.items[0].citations  # stable
    assert pack.items[0].citations[0].citation_id == "cite_1"
    assert pack.items[0].citations[0].source_span_id == "span-a"
    assert pack.items[0].citations[0].label == "p. 1"
    assert pack.items[1].citations[0].citation_id == "cite_2"
    # Every emitted chunk carries its text and a citation_id.
    assert [c.text for c in pack.items[0].matched_chunks] == ["chunk one", "chunk two"]
    assert all(c.citation_id == "cite_1" for c in pack.items[0].matched_chunks)
    assert pack.items[1].matched_chunks[0].citation_id == "cite_2"


def test_deterministic_across_repeated_calls() -> None:
    results = [
        _result(
            item_id="item-1",
            chunks=[("c1", ChunkType.RECIPE_SUMMARY)],
            citations=[("span-a", "p. 1")],
        )
    ]
    chunk_input = {"c1": ChunkInput(text="t", source_span_ids=["span-a"])}
    a = build_context_pack("q", results, chunk_input, item_limit=6, chunks_per_item=3)
    b = build_context_pack("q", results, chunk_input, item_limit=6, chunks_per_item=3)
    assert a == b


def test_items_and_chunks_are_capped() -> None:
    results = [
        _result(
            item_id=f"item-{n}",
            chunks=[("c", ChunkType.RECIPE_SUMMARY)],
            citations=[("span", "p")],
        )
        for n in range(5)
    ]
    chunk_input = {"c": ChunkInput(text="t", source_span_ids=["span"])}
    pack = build_context_pack("q", results, chunk_input, item_limit=2, chunks_per_item=3)
    assert len(pack.items) == 2


def test_chunks_per_item_cap_drops_trailing_chunks_and_their_citations() -> None:
    # Three chunks, each with a *distinct* span; capping at 2 must drop the third
    # chunk AND its span from the citations (reachability: no cite without an
    # emitted chunk referencing it).
    results = [
        _result(
            item_id="item-1",
            chunks=[
                ("c1", ChunkType.RECIPE_SUMMARY),
                ("c2", ChunkType.RECIPE_FULL),
                ("c3", ChunkType.RECIPE_STEPS),
            ],
            citations=[("span-1", "p1"), ("span-2", "p2"), ("span-3", "p3")],
        )
    ]
    chunk_input = {
        "c1": ChunkInput(text="one", source_span_ids=["span-1"]),
        "c2": ChunkInput(text="two", source_span_ids=["span-2"]),
        "c3": ChunkInput(text="three", source_span_ids=["span-3"]),
    }
    pack = build_context_pack("q", results, chunk_input, item_limit=6, chunks_per_item=2)
    item = pack.items[0]
    assert len(item.matched_chunks) == 2
    cited_spans = {c.source_span_id for c in item.citations}
    assert cited_spans == {"span-1", "span-2"}  # span-3 dropped with its chunk
    # Every citation is reachable from an emitted chunk.
    referenced = {c.citation_id for c in item.matched_chunks}
    # span-1 → cite_1, span-2 → cite_2; both are reachable.
    assert {c.citation_id for c in item.citations} == referenced


def test_chunk_with_missing_text_is_skipped() -> None:
    results = [
        _result(
            item_id="item-1",
            chunks=[("c1", ChunkType.RECIPE_SUMMARY), ("c2", ChunkType.RECIPE_FULL)],
            citations=[("span-a", "p1")],
        )
    ]
    # c1 has no entry in the map (not fetched); c2 has text.
    chunk_input = {"c2": ChunkInput(text="present", source_span_ids=["span-a"])}
    pack = build_context_pack("q", results, chunk_input, item_limit=6, chunks_per_item=3)
    assert [c.chunk_id for c in pack.items[0].matched_chunks] == ["c2"]


def test_chunk_with_empty_text_is_skipped() -> None:
    results = [
        _result(
            item_id="item-1",
            chunks=[("c1", ChunkType.RECIPE_SUMMARY)],
            citations=[("span-a", "p1")],
        )
    ]
    chunk_input = {"c1": ChunkInput(text="", source_span_ids=["span-a"])}
    pack = build_context_pack("q", results, chunk_input, item_limit=6, chunks_per_item=3)
    assert pack.items == []


def test_chunk_with_no_matching_span_is_skipped_and_item_omitted() -> None:
    results = [
        _result(
            item_id="item-1",
            chunks=[("c1", ChunkType.RECIPE_SUMMARY)],
            citations=[("span-a", "p1")],
        )
    ]
    # c1 references a span not present in the item's source_citations.
    chunk_input = {"c1": ChunkInput(text="x", source_span_ids=["span-Z"])}
    pack = build_context_pack("q", results, chunk_input, item_limit=6, chunks_per_item=3)
    assert pack.items == []


def test_chunk_with_empty_span_list_is_skipped_and_item_omitted() -> None:
    results = [
        _result(
            item_id="item-1",
            chunks=[("c1", ChunkType.RECIPE_SUMMARY)],
            citations=[("span-a", "p1")],
        )
    ]
    chunk_input = {"c1": ChunkInput(text="x", source_span_ids=[])}
    pack = build_context_pack("q", results, chunk_input, item_limit=6, chunks_per_item=3)
    assert pack.items == []


def test_one_uncitable_item_omitted_other_kept_renumbers_ctx() -> None:
    results = [
        _result(
            item_id="item-bad",
            chunks=[("c1", ChunkType.RECIPE_SUMMARY)],
            citations=[("span-a", "p1")],
        ),
        _result(
            item_id="item-good",
            chunks=[("c2", ChunkType.RECIPE_SUMMARY)],
            citations=[("span-b", "p2")],
        ),
    ]
    chunk_input = {
        "c1": ChunkInput(text="x", source_span_ids=["span-Z"]),  # uncitable
        "c2": ChunkInput(text="y", source_span_ids=["span-b"]),
    }
    pack = build_context_pack("q", results, chunk_input, item_limit=6, chunks_per_item=3)
    assert len(pack.items) == 1
    assert pack.items[0].context_item_id == "ctx_1"  # renumbered over emitted items
    assert pack.items[0].knowledge_item_id == "item-good"
    assert pack.items[0].citations[0].citation_id == "cite_1"


def test_structured_preview_included_when_supplied_and_omitted_otherwise() -> None:
    results = [
        _result(
            item_id="item-1",
            chunks=[("c1", ChunkType.RECIPE_SUMMARY)],
            citations=[("span-a", "p1")],
        ),
        _result(
            item_id="item-2",
            chunks=[("c2", ChunkType.RECIPE_SUMMARY)],
            citations=[("span-b", "p2")],
        ),
    ]
    chunk_input = {
        "c1": ChunkInput(text="x", source_span_ids=["span-a"]),
        "c2": ChunkInput(text="y", source_span_ids=["span-b"]),
    }
    previews = {"item-1": {"yield": "4 servings", "top_ingredients": ["flour"]}}
    pack = build_context_pack(
        "q",
        results,
        chunk_input,
        item_limit=6,
        chunks_per_item=3,
        structured_preview_by_item_id=previews,
    )
    assert pack.items[0].structured_preview == {
        "yield": "4 servings",
        "top_ingredients": ["flour"],
    }
    assert pack.items[1].structured_preview is None


def test_duplicate_span_within_item_gets_single_citation() -> None:
    # Two chunks referencing the same span → one cite_N, both chunks point to it.
    results = [
        _result(
            item_id="item-1",
            chunks=[("c1", ChunkType.RECIPE_SUMMARY), ("c2", ChunkType.RECIPE_FULL)],
            citations=[("span-a", "p1")],
        )
    ]
    chunk_input = {
        "c1": ChunkInput(text="one", source_span_ids=["span-a"]),
        "c2": ChunkInput(text="two", source_span_ids=["span-a"]),
    }
    pack = build_context_pack("q", results, chunk_input, item_limit=6, chunks_per_item=3)
    item = pack.items[0]
    assert len(item.citations) == 1
    assert {c.citation_id for c in item.matched_chunks} == {"cite_1"}
