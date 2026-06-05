"""Shared SQL predicate builder for the retrieval legs (doc 7 § 4).

Both the keyword leg (12.1) and the vector leg (12.2) run a ``Chunk → KnowledgeItem
→ Document`` query under the same invariant + FilterSet constraints. The predicate
application lives here — a neutral module — so neither leg imports across to the
other (DECISIONS #1). The join itself is built by each leg (the SELECT columns and
the leg-specific match differ); this only adds the shared WHERE predicates.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Select

from rag_recipes.retrieval.types import FilterSet
from rag_recipes.storage.enums import ChunkParentType, KnowledgeItemStatus
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.knowledge_item import KnowledgeItem


def apply_search_predicates(stmt: Select[Any], filters: FilterSet) -> Select[Any]:
    """Add the invariant + request-driven WHERE predicates to a search query.

    Invariant (always-on, never request-driven): the item is ``READY``, its
    ``source_version`` equals the document's ``active_source_version`` (so stale and
    mid-ingestion versions are excluded — a NULL active version matches nothing), and
    the chunk's parent is a ``KnowledgeItem``. Request-driven: ``category`` and
    ``item_type`` are exact matches; ``subcategory`` is an exact match only when
    set (a non-null filter therefore excludes NULL-subcategory documents, doc 7 § 3);
    ``document_ids`` restricts to the allowlist only when non-empty.
    """
    stmt = stmt.where(
        KnowledgeItem.status == KnowledgeItemStatus.READY,
        KnowledgeItem.source_version == Document.active_source_version,
        Chunk.parent_type == ChunkParentType.KNOWLEDGE_ITEM,
        Document.category == filters.category,
        KnowledgeItem.item_type == filters.item_type,
    )
    if filters.subcategory is not None:
        stmt = stmt.where(Document.subcategory == filters.subcategory)
    if filters.document_ids:
        stmt = stmt.where(Document.id.in_(filters.document_ids))
    return stmt
