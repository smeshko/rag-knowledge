"""GET /api/v1/knowledge-items/{item_id} — the canonical item detail (doc 6 § 8).

Returns a KnowledgeItem with its FULL, untruncated ``structured_data`` (every
ingredient and step, plus any unknown keys, passed through verbatim), a small
doc-6 §8 ``display`` block, and item-level source citations. A direct audit-friendly
lookup: it returns the item regardless of status (ready / needs_review / superseded /
extracting) — 404 is reserved for a genuinely-unknown id.

This module stays **read-only** (Epic 21.3, D4): the review surface owns the
writes, including the Epic 22.2 edit ``PATCH``. The response assembly itself
lives in ``api/knowledge_item_view`` so both endpoints answer with one envelope.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.dependencies import get_session
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.knowledge_item_view import build_knowledge_item_response
from rag_recipes.api.schemas.knowledge_items import KnowledgeItemResponse
from rag_recipes.storage.models.knowledge_item import KnowledgeItem

router = APIRouter(tags=["knowledge-items"])


@router.get("/knowledge-items/{item_id}", response_model=KnowledgeItemResponse)
async def get_knowledge_item(
    item_id: str,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Any:
    item = await session.get(KnowledgeItem, item_id)
    if item is None:
        raise ApiError(
            status_code=404,
            code=ErrorCode.KNOWLEDGE_ITEM_NOT_FOUND,
            message=f"Knowledge item {item_id!r} not found.",
            details={"item_id": item_id},
        )

    return await build_knowledge_item_response(session, item)
