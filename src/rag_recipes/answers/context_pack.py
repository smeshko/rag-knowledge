"""Context-pack builder for the query-time answer layer (Epic 17, doc 8 § 4).

``build_context_pack`` is a pure, deterministic function: it turns the top
retrieval ``KnowledgeItemResult``s into a ``ContextPack`` whose ``ctx_N`` / ``cite_N``
ids are the *only* ids the answer model is allowed to cite. 17.2's citation
validation checks the model's output against this id space, so id assignment must
be deterministic for fixed inputs.

Two facts about the retrieval result force the richer inputs (review #1):

* ``MatchedChunkRef`` carries only ``chunk_id`` / ``chunk_type`` / ``score`` — no
  text and no per-chunk span ids — so the builder takes ``chunk_input_by_id``
  (text + ``source_span_ids``, fetched by 17.2) to give each emitted chunk its
  text and ``citation_id``.
* ``ResultItem`` has no ``structured_data`` — so ``structured_preview`` is injected
  via ``structured_preview_by_item_id`` (built by 17.2), present ⇒ included,
  absent ⇒ omitted.

Citations are derived **after** chunk capping/skipping, from spans referenced by
emitted chunks only — never from the item's full ``source_citations`` wholesale —
so every exposed ``cite_N`` is backed by context the model actually saw and the
model can't "cite unseen evidence" (review #2, finding 2).

Known limitation (review #2, finding 1): chunk ``source_span_ids`` are item-level
today (``chunking.py`` copies ``item.source_span_ids`` onto every chunk), so the
per-chunk ``citation_id`` reflects item-level provenance, not evidence-locality.
The builder does not assume evidence-local spans; evidence-local chunk spans are a
future chunking enhancement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from rag_recipes.retrieval.types import KnowledgeItemResult

__all__ = [
    "ChunkInput",
    "ContextChunk",
    "ContextCitation",
    "ContextDocument",
    "ContextItem",
    "ContextPack",
    "build_context_pack",
]


@dataclass(frozen=True)
class ChunkInput:
    """Injected per-chunk data the retrieval result doesn't carry (review #1).

    ``text`` is the chunk's fetched text; ``source_span_ids`` are the span ids the
    chunk references (item-level today — see module docstring). 17.2 builds the
    ``chunk_input_by_id`` map from the fetched ``Chunk`` rows.
    """

    text: str
    source_span_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContextCitation:
    """A ``cite_N`` → ``source_span_id`` (+ ``label``) entry, derived from emitted chunks."""

    citation_id: str
    source_span_id: str
    label: str


@dataclass(frozen=True)
class ContextChunk:
    """An emitted chunk: its text and the ``cite_N`` of its first surviving span."""

    chunk_id: str
    chunk_type: str
    text: str
    citation_id: str


@dataclass(frozen=True)
class ContextDocument:
    """The parent document projection carried into the pack."""

    document_id: str
    title: str
    author: str


@dataclass(frozen=True)
class ContextItem:
    """One item in the pack: its ``ctx_N`` id, projection, emitted chunks, and citations."""

    context_item_id: str
    knowledge_item_id: str
    title: str
    summary: str | None
    document: ContextDocument
    matched_chunks: list[ContextChunk]
    citations: list[ContextCitation]
    structured_preview: dict[str, Any] | None = None


@dataclass(frozen=True)
class ContextPack:
    """The query plus the emitted context items — the model's grounding surface."""

    query: str
    items: list[ContextItem]


def build_context_pack(
    query: str,
    results: list[KnowledgeItemResult],
    chunk_input_by_id: Mapping[str, ChunkInput],
    *,
    item_limit: int,
    chunks_per_item: int,
    structured_preview_by_item_id: Mapping[str, dict[str, Any]] | None = None,
) -> ContextPack:
    """Build a citation-ready context pack from retrieval results (pure, deterministic).

    Items are capped to ``min(len(results), item_limit)`` and matched chunks per
    item to ``chunks_per_item`` (capped *before* citation assignment). Within the
    capped chunks, a chunk is skipped when it has no fetched text or when none of
    its ``source_span_ids`` map to a span in the item's ``source_citations``.
    Citations are then derived from the surviving emitted chunks only; an item left
    with no citable emitted chunk is omitted entirely. ``ctx_N`` ids are assigned
    sequentially over the emitted items; ``cite_N`` ids over the spans referenced by
    emitted chunks, in first-seen order across the pack.
    """
    previews = structured_preview_by_item_id or {}
    items: list[ContextItem] = []
    ctx_counter = 0
    cite_counter = 0

    for result in results[:item_limit]:
        # Item-level span → label map; first label wins on a duplicate span id.
        span_label: dict[str, str] = {}
        for citation in result.source_citations:
            span_label.setdefault(citation.source_span_id, citation.label)

        # Cap first, then keep only citable chunks among the top `chunks_per_item`.
        surviving: list[tuple[str, str, str, list[str]]] = []  # chunk_id, type, text, spans
        for chunk_ref in result.matched_chunks[:chunks_per_item]:
            chunk_input = chunk_input_by_id.get(chunk_ref.chunk_id)
            if chunk_input is None or not chunk_input.text:
                continue  # no fetched text — skip, don't error
            valid_spans = [sid for sid in chunk_input.source_span_ids if sid in span_label]
            if not valid_spans:
                continue  # references no citable span — skip
            surviving.append(
                (chunk_ref.chunk_id, chunk_ref.chunk_type.value, chunk_input.text, valid_spans)
            )

        if not surviving:
            continue  # no citable emitted chunk — omit the item

        # Assign cite_N to every span referenced by a surviving chunk, first-seen.
        span_to_cite: dict[str, str] = {}
        citations: list[ContextCitation] = []
        for _chunk_id, _chunk_type, _text, valid_spans in surviving:
            for span_id in valid_spans:
                if span_id in span_to_cite:
                    continue
                cite_counter += 1
                cite_id = f"cite_{cite_counter}"
                span_to_cite[span_id] = cite_id
                citations.append(
                    ContextCitation(
                        citation_id=cite_id,
                        source_span_id=span_id,
                        label=span_label[span_id],
                    )
                )

        chunks = [
            ContextChunk(
                chunk_id=chunk_id,
                chunk_type=chunk_type,
                text=text,
                citation_id=span_to_cite[valid_spans[0]],
            )
            for chunk_id, chunk_type, text, valid_spans in surviving
        ]

        ctx_counter += 1
        items.append(
            ContextItem(
                context_item_id=f"ctx_{ctx_counter}",
                knowledge_item_id=result.item.knowledge_item_id,
                title=result.item.title,
                summary=result.item.summary,
                document=ContextDocument(
                    document_id=result.document.document_id,
                    title=result.document.title,
                    author=result.document.author,
                ),
                matched_chunks=chunks,
                citations=citations,
                structured_preview=previews.get(result.item.knowledge_item_id),
            )
        )

    return ContextPack(query=query, items=items)
