"""Helpers shared by the query-time answer flows (``answers/`` and ``menus/``).

``menus/`` is a second flow *of* the answer layer (DECISIONS.md #15): plan →
retrieve → context pack → grounded LLM → citation validation → safe fallback. The
grounding contract (``build_context_pack``, ``build_response_citations``,
``GROUNDING_RULES``) was imported rather than copied from the start; these are the
remaining pieces both flows need, lifted here so the two can never drift (doc 13
§8 — one place per cross-cutting concern).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.answers.context_pack import ChunkInput, ContextPack
from rag_recipes.api.search_projection import build_structured_preview
from rag_recipes.storage.models.chunk import Chunk

__all__ = [
    "allowed_ids_footer",
    "as_list",
    "fetch_chunk_inputs",
    "previews_from_structured",
]


def as_list(value: Any) -> list[Any]:
    """Coerce a model-supplied field to a list (``[]`` for anything non-list).

    ``parse_error is None`` only guarantees a JSON *object*, not a schema-conforming
    one — the provider does no post-parse JSON-Schema validation. So a wrong-typed
    field (``null``, a string, …) must degrade to a validation failure, never an
    ``AttributeError``/``TypeError`` that would escape the service as a 500.
    """
    return value if isinstance(value, list) else []


async def fetch_chunk_inputs(session: AsyncSession, chunk_ids: list[str]) -> dict[str, ChunkInput]:
    """Batch-fetch ``text`` + ``source_span_ids`` for exactly ``chunk_ids``.

    Callers decide the cap window (which items, how many matched chunks each); only
    those chunks are read.
    """
    if not chunk_ids:
        return {}
    rows = (
        await session.execute(
            select(Chunk.id, Chunk.text, Chunk.source_span_ids).where(Chunk.id.in_(chunk_ids))
        )
    ).all()
    return {
        row.id: ChunkInput(text=row.text, source_span_ids=list(row.source_span_ids or []))
        for row in rows
    }


def previews_from_structured(structured: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``knowledge_item_id → structured_preview`` dict, as the context pack expects.

    Derived through ``build_structured_preview`` so the pack carries the same
    projection the search response does (doc 06 — preview, never canonical data).
    """
    return {
        item_id: build_structured_preview(extra.get("structured_data", {})).model_dump(
            by_alias=True
        )
        for item_id, extra in structured.items()
    }


def allowed_ids_footer(pack: ContextPack) -> str:
    """The explicit allow-lists of ``citation_id``s / ``knowledge_item_id``s.

    Appended to every synthesis input; it is the same id space the citation
    validators check the model's output against. Emitted in pack order, first
    occurrence wins, so the footer is deterministic for a fixed pack.
    """
    allowed_citation_ids: list[str] = []
    allowed_item_ids: list[str] = []
    for item in pack.items:
        if item.knowledge_item_id not in allowed_item_ids:
            allowed_item_ids.append(item.knowledge_item_id)
        for citation in item.citations:
            if citation.citation_id not in allowed_citation_ids:
                allowed_citation_ids.append(citation.citation_id)
    return (
        f"Allowed citation IDs: {', '.join(allowed_citation_ids) or '(none)'}\n"
        f"Allowed knowledge_item_ids: {', '.join(allowed_item_ids) or '(none)'}"
    )
