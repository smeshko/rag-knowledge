"""Chunk creation from ready KnowledgeItems (doc 2 § 5, doc 4 § Chunk Creation).

Pure stage: turn a ``READY`` ``KnowledgeItem`` into the canonical recipe chunks.
``build_chunks`` performs **no I/O** — it reads only the item plus an injected
``category`` (which lives on ``Document``, not ``KnowledgeItem``) and returns
detached ``Chunk`` instances. Persistence is the job of
``persist_chunks_for_ready_items`` (added in Phase 10.1 TASK-002), which owns the
DB session; ``build_chunks`` itself never touches a session, provider, storage,
or ``Settings``.

Per-type source-of-text rules (DECISIONS #2). A chunk type is **skipped** when
its resolved text is empty/whitespace-only (so a ready item yields one to five
chunks, never an empty chunk):

- ``recipe_title``       → ``item.title`` alone (DECISIONS #1). ``title`` is
  NOT NULL and non-empty for ready items, so this chunk effectively always
  survives.
- ``recipe_summary``     → ``item.summary``.
- ``recipe_ingredients`` → ``structured_data["ingredients_text"]`` if non-empty,
  else the ``structured_data["ingredients"]`` list's ``raw_text`` fields joined
  on newlines.
- ``recipe_steps``       → ``structured_data["steps_text"]`` if non-empty, else
  the ``structured_data["steps"]`` list's ``text`` fields joined on newlines.
- ``recipe_full``        → ``item.body_text`` if non-empty, else the
  already-resolved title + ingredients + steps text blocks joined by blank lines.

``text_hash`` reuses the canonical SHA-256-of-text approach (DECISIONS #3),
matching ``pipeline/pdf_text._sha256_text``.
"""

from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.storage.enums import ChunkParentType, ChunkType, KnowledgeItemStatus
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.knowledge_item import KnowledgeItem

__all__ = ["build_chunks", "persist_chunks_for_ready_items"]


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _is_present(text: str | None) -> bool:
    """A source text counts only when it has non-whitespace content."""
    return bool(text and text.strip())


def _resolve_ingredients(structured: dict[str, Any]) -> str:
    """``ingredients_text`` when present, else the ingredient list's ``raw_text``."""
    text = structured.get("ingredients_text")
    if _is_present(text):
        return text  # type: ignore[return-value]
    ingredients = structured.get("ingredients") or []
    return "\n".join(item.get("raw_text", "") for item in ingredients)


def _resolve_steps(structured: dict[str, Any]) -> str:
    """``steps_text`` when present, else the step list's ``text`` fields."""
    text = structured.get("steps_text")
    if _is_present(text):
        return text  # type: ignore[return-value]
    steps = structured.get("steps") or []
    return "\n".join(step.get("text", "") for step in steps)


def _join_blocks(blocks: list[str]) -> str:
    """Concatenate the non-empty text blocks separated by a blank line."""
    return "\n\n".join(block for block in blocks if _is_present(block))


def build_chunks(item: KnowledgeItem, *, category: str) -> list[Chunk]:
    """Build the canonical recipe chunks for a ``READY`` ``KnowledgeItem``.

    Returns up to five ``Chunk`` instances in canonical order
    (``recipe_title, recipe_summary, recipe_ingredients, recipe_steps,
    recipe_full``), skipping any type whose resolved source text is
    empty/whitespace-only. Returns ``[]`` for non-``READY`` items so
    ``NEEDS_REVIEW`` / ``SUPERSEDED`` items produce no chunks. ``category`` is
    injected by the caller (it lives on ``Document``); ``build_chunks`` never
    loads ``item.document``.
    """
    if item.status is not KnowledgeItemStatus.READY:
        return []

    structured = item.structured_data or {}

    title_text = item.title or ""
    summary_text = item.summary or ""
    ingredients_text = _resolve_ingredients(structured)
    steps_text = _resolve_steps(structured)

    body_text = item.body_text or ""
    if not _is_present(body_text):
        body_text = _join_blocks([title_text, ingredients_text, steps_text])

    candidates: list[tuple[ChunkType, str]] = [
        (ChunkType.RECIPE_TITLE, title_text),
        (ChunkType.RECIPE_SUMMARY, summary_text),
        (ChunkType.RECIPE_INGREDIENTS, ingredients_text),
        (ChunkType.RECIPE_STEPS, steps_text),
        (ChunkType.RECIPE_FULL, body_text),
    ]

    chunks: list[Chunk] = []
    for chunk_type, text in candidates:
        if not _is_present(text):
            continue
        chunks.append(
            Chunk(
                document_id=item.document_id,
                parent_type=ChunkParentType.KNOWLEDGE_ITEM,
                parent_id=item.id,
                chunk_type=chunk_type,
                text=text,
                text_hash=_sha256_text(text),
                source_span_ids=list(item.source_span_ids),
                chunk_metadata={
                    "category": category,
                    "item_type": item.item_type,
                    "title": item.title,
                },
            )
        )
    return chunks


async def persist_chunks_for_ready_items(
    session: AsyncSession, *, document_id: str, source_version: int, category: str
) -> int:
    """Build and persist the chunks for every ``READY`` item of a source_version.

    Loads the document's ``READY`` ``KnowledgeItem`` rows *at ``source_version``*,
    builds each item's chunks with ``build_chunks``, adds them all, and flushes
    (surfacing the composite FK / ``@validates`` checks at the call site). Returns
    the total number of chunks written. The caller owns the transaction — this
    never commits, mirroring ``extract_and_persist_spans`` /
    ``persist_knowledge_item``. ``NEEDS_REVIEW`` / ``SUPERSEDED`` items are not
    loaded, so they contribute no chunks.

    Scoping to ``source_version`` (Epic 11.2) keeps a new_source_version run from
    re-chunking the prior version's still-READY items: during that run the prior
    version's items remain active (the supersede + active-version flip happen later,
    at the READY gate) so an unscoped query would duplicate their chunks.
    """
    result = await session.execute(
        select(KnowledgeItem).where(
            KnowledgeItem.document_id == document_id,
            KnowledgeItem.source_version == source_version,
            KnowledgeItem.status == KnowledgeItemStatus.READY,
        )
    )
    chunks: list[Chunk] = []
    for item in result.scalars().all():
        chunks.extend(build_chunks(item, category=category))
    session.add_all(chunks)
    await session.flush()
    return len(chunks)
