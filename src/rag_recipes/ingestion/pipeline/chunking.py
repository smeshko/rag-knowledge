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

The per-type text resolution itself lives in ``pipeline/composition`` (Epic
22.1), shared with the ``body_text`` rebuild an edit performs, so the chunked
text and the stored ``body_text`` cannot drift apart.

**Size cap.** A chunk is an embedding input, and every embedding backend caps a
single input. Nothing here used to bound chunk text, so a long enough recipe
produced a chunk the embedder had to reject -- and because that rejection is a
whole-request failure, one oversized chunk failed its entire document at the
embedding stage (a 10,434-byte gluten-free croissant recipe did exactly that,
with the next-largest chunk in the same book landing one byte under the limit).
``build_chunks`` now splits any over-cap text into consecutive same-type chunks
instead. Splitting rather than truncating keeps every word retrievable; the
alternative silently drops the tail of long recipes and lets chunk text drift
from the ``body_text`` it was composed from. Parts are cut on paragraph, then
line, then character boundaries -- never mid-codepoint -- and carry ``part`` /
``part_count`` metadata. Items under the cap are unaffected and still yield
exactly one chunk per type.
"""

from __future__ import annotations

import hashlib

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.ingestion.pipeline.composition import (
    compose_body_text,
    is_present,
    resolve_ingredients_text,
    resolve_steps_text,
)
from rag_recipes.storage.enums import ChunkParentType, ChunkType, KnowledgeItemStatus
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.knowledge_item import KnowledgeItem

__all__ = ["MAX_CHUNK_BYTES", "build_chunks", "persist_chunks_for_ready_items"]

# Budget for one chunk's text, in UTF-8 bytes. The OpenAI embedding provider
# bounds an input's token count by its UTF-8 byte length (cl100k_base is
# byte-level BPE, so tokens can never exceed bytes) and rejects anything over
# 8192. We cap below that rather than at it: the margin absorbs the provider's
# per-input accounting without this module having to import a provider constant,
# which would couple the pure chunking stage to one backend.
MAX_CHUNK_BYTES = 8000


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _byte_len(s: str) -> int:
    return len(s.encode("utf-8"))


def _hard_split(text: str, max_bytes: int) -> list[str]:
    """Split on character boundaries when a single line still exceeds the cap.

    Accumulates codepoints and measures the encoded length, so a multi-byte
    character is never cut in half.
    """
    parts: list[str] = []
    current = ""
    current_bytes = 0
    for ch in text:
        ch_bytes = _byte_len(ch)
        if current and current_bytes + ch_bytes > max_bytes:
            parts.append(current)
            current = ""
            current_bytes = 0
        current += ch
        current_bytes += ch_bytes
    if current:
        parts.append(current)
    return parts


def _pack(segments: list[str], joiner: str, max_bytes: int) -> list[str]:
    """Greedily join segments into runs that each fit the byte budget.

    A segment that does not fit on its own is yielded alone, for the caller to
    break down further.
    """
    parts: list[str] = []
    current = ""
    for segment in segments:
        candidate = segment if not current else current + joiner + segment
        if current and _byte_len(candidate) > max_bytes:
            parts.append(current)
            current = segment
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _split_text(text: str, max_bytes: int) -> list[str]:
    """Break ``text`` into consecutive pieces that each fit ``max_bytes``.

    Prefers the largest natural boundary that works -- paragraphs, then lines,
    then characters -- so a split lands between steps or ingredients rather than
    mid-sentence wherever the text allows it.
    """
    if _byte_len(text) <= max_bytes:
        return [text]

    out: list[str] = []
    for block in _pack(text.split("\n\n"), "\n\n", max_bytes):
        if _byte_len(block) <= max_bytes:
            out.append(block)
            continue
        for line_run in _pack(block.split("\n"), "\n", max_bytes):
            if _byte_len(line_run) <= max_bytes:
                out.append(line_run)
            else:
                out.extend(_hard_split(line_run, max_bytes))
    return [p for p in out if p.strip()]


def build_chunks(
    item: KnowledgeItem, *, category: str, max_bytes: int = MAX_CHUNK_BYTES
) -> list[Chunk]:
    """Build the canonical recipe chunks for a ``READY`` ``KnowledgeItem``.

    Returns up to five ``Chunk`` instances in canonical order
    (``recipe_title, recipe_summary, recipe_ingredients, recipe_steps,
    recipe_full``), skipping any type whose resolved source text is
    empty/whitespace-only. Returns ``[]`` for non-``READY`` items so
    ``NEEDS_REVIEW`` / ``SUPERSEDED`` items produce no chunks. ``category`` is
    injected by the caller (it lives on ``Document``); ``build_chunks`` never
    loads ``item.document``.

    A type whose text exceeds ``max_bytes`` (UTF-8) yields several consecutive
    chunks of that same type instead of one, so an item can return more than
    five chunks. Ordering is still canonical, with a split type's parts adjacent
    and in reading order.
    """
    if item.status is not KnowledgeItemStatus.READY:
        return []

    structured = item.structured_data or {}

    title_text = item.title or ""
    summary_text = item.summary or ""
    ingredients_text = resolve_ingredients_text(structured)
    steps_text = resolve_steps_text(structured)

    body_text = item.body_text or ""
    if not is_present(body_text):
        body_text = compose_body_text(title=title_text, structured=structured)

    candidates: list[tuple[ChunkType, str]] = [
        (ChunkType.RECIPE_TITLE, title_text),
        (ChunkType.RECIPE_SUMMARY, summary_text),
        (ChunkType.RECIPE_INGREDIENTS, ingredients_text),
        (ChunkType.RECIPE_STEPS, steps_text),
        (ChunkType.RECIPE_FULL, body_text),
    ]

    chunks: list[Chunk] = []
    for chunk_type, text in candidates:
        if not is_present(text):
            continue
        parts = _split_text(text, max_bytes)
        for index, part in enumerate(parts):
            metadata: dict[str, object] = {
                "category": category,
                "item_type": item.item_type,
                "title": item.title,
            }
            # Only stamped on split types, so an under-cap item's metadata is
            # byte-identical to what it was before the cap existed.
            if len(parts) > 1:
                metadata["part"] = index + 1
                metadata["part_count"] = len(parts)
            chunks.append(
                Chunk(
                    document_id=item.document_id,
                    parent_type=ChunkParentType.KNOWLEDGE_ITEM,
                    parent_id=item.id,
                    chunk_type=chunk_type,
                    text=part,
                    text_hash=_sha256_text(part),
                    source_span_ids=list(item.source_span_ids),
                    chunk_metadata=metadata,
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

    Items that already carry chunks are skipped, which makes the call replayable:
    a ``resume`` finalize re-runs this over a READY set that mixes freshly promoted
    winners with items an earlier pass of the same ingest already chunked, and only
    the former need building.
    """
    result = await session.execute(
        select(KnowledgeItem).where(
            KnowledgeItem.document_id == document_id,
            KnowledgeItem.source_version == source_version,
            KnowledgeItem.status == KnowledgeItemStatus.READY,
        )
    )
    items = list(result.scalars().all())
    already_chunked: set[str] = set()
    if items:
        already_chunked = set(
            (
                await session.execute(
                    select(Chunk.parent_id).where(Chunk.parent_id.in_([item.id for item in items]))
                )
            )
            .scalars()
            .all()
        )
    chunks: list[Chunk] = []
    for item in items:
        if item.id in already_chunked:
            continue
        chunks.extend(build_chunks(item, category=category))
    session.add_all(chunks)
    await session.flush()
    return len(chunks)
