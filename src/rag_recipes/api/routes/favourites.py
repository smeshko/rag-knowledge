"""The reader's favourites: one star per recipe, and the shelf of starred ones.

- ``PUT /api/v1/knowledge-items/{item_id}/favourite`` — star a recipe.
- ``DELETE /api/v1/knowledge-items/{item_id}/favourite`` — unstar it.
- ``GET /api/v1/favourites`` — every starred recipe, newest star first.

All three live here rather than in ``knowledge_items.py`` even though two of
them hang off ``/knowledge-items/{id}``: the listing is a cross-book surface
with no addressed item at all, and splitting the three would put the write and
the read of one small table in two modules. ``knowledge_items.py`` owns the
extraction row's lifecycle (read, edit, delete); this owns an annotation about
it, which is exactly the boundary the storage model draws too.

**Both writes are idempotent** and neither guards on document status. Starring
is not a lifecycle transition — it writes no ``knowledge_items`` row (see the
model docstring), touches nothing ingestion reads, and is safe mid-reprocess in
the same way that *reading* a book's contents is. The 409 the write verbs in
``knowledge_items.py`` raise would be answering a question nobody asked.

**Any status can be starred**, ``needs_review`` and ``superseded`` included,
and the listing hides nothing. The per-book listing's rule (hide ``superseded``
and ``rejected`` unfiltered) is wrong here for one reason: those rows are
hidden there because nobody asked for them, whereas here somebody explicitly
did. A star that silently stopped showing its recipe would look like data loss.

Consequence worth knowing: reprocessing a book supersedes its items and
extracts NEW ones, so a favourite survives as a star on the superseded row
rather than following the recipe into the new generation. The list keeps
showing it (with ``status: "superseded"``, which the UI can mark), instead of
the row quietly vanishing. Carrying stars across generations needs identity
matching between generations, which does not exist today.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Response
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.dependencies import get_session
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.knowledge_item_list import build_summaries
from rag_recipes.api.routes._params import parse_int
from rag_recipes.api.schemas.favourites import Favourite, FavouriteResponse
from rag_recipes.api.schemas.review import KnowledgeItemListResponse
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.knowledge_item_favourite import KnowledgeItemFavourite

router = APIRouter(tags=["favourites"])

logger = logging.getLogger(__name__)

_LIST_LIMIT_DEFAULT = 50
_LIST_LIMIT_MAX = 200
_LIST_OFFSET_DEFAULT = 0


async def _require_item(session: AsyncSession, item_id: str) -> None:
    """404 on an unknown recipe — the ``GET /knowledge-items/{id}`` rule.

    A missing row is the ONLY 404 either write raises: starring an
    already-starred recipe, or unstarring one that is not starred, are both
    successful no-ops (see the module docstring on idempotence).
    """
    exists = (
        await session.execute(select(KnowledgeItem.id).where(KnowledgeItem.id == item_id))
    ).scalar_one_or_none()
    if exists is None:
        raise ApiError(
            status_code=404,
            code=ErrorCode.KNOWLEDGE_ITEM_NOT_FOUND,
            message=f"Knowledge item {item_id!r} not found.",
            details={"item_id": item_id},
        )


@router.put(
    "/knowledge-items/{item_id}/favourite",
    response_model=FavouriteResponse,
)
async def add_favourite(
    item_id: str,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Any:
    """Star a recipe. 200 with the star's timestamp, whether or not it is new.

    ``ON CONFLICT DO NOTHING`` plus a read-back, rather than an upsert that
    restamps ``created_at``: the second PUT of a double-click must not reorder
    the favourites list. The read-back is also what makes the timestamp in the
    response the *server's* — a concurrent PUT that won the insert produced it.

    PUT rather than POST because that is what this is: an idempotent write of a
    named sub-resource whose whole content is "it exists".
    """
    await _require_item(session, item_id)

    await session.execute(
        pg_insert(KnowledgeItemFavourite)
        .values(knowledge_item_id=item_id)
        .on_conflict_do_nothing(index_elements=["knowledge_item_id"])
    )
    # Read back INSIDE the transaction: on the conflict path the insert
    # returned nothing, and on the insert path `created_at` is a server default
    # the statement never sent us.
    favourited_at = (
        await session.execute(
            select(KnowledgeItemFavourite.created_at).where(
                KnowledgeItemFavourite.knowledge_item_id == item_id
            )
        )
    ).scalar_one()
    await session.commit()

    return FavouriteResponse(
        favourite=Favourite(knowledge_item_id=item_id, favourited_at=favourited_at)
    )


@router.delete("/knowledge-items/{item_id}/favourite", status_code=204)
async def remove_favourite(
    item_id: str,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Response:
    """Unstar a recipe. 204 whether or not it was starred.

    Deliberately NOT a 404 on an unstarred recipe: the caller asked for a state
    ("this is not favourited") that holds when the handler returns, and a UI
    retrying a dropped request would otherwise be told its own success failed.
    An unknown *item* is still a 404 — that is a caller mistake, not a repeat.

    Nothing is logged: unlike the item delete this destroys no content, and the
    row can be recreated with one click.
    """
    await _require_item(session, item_id)

    favourite = await session.get(KnowledgeItemFavourite, item_id)
    if favourite is not None:
        await session.delete(favourite)
        await session.commit()
    return Response(status_code=204)


@router.get("/favourites", response_model=KnowledgeItemListResponse)
async def list_favourites(
    limit: str | None = None,
    offset: str | None = None,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Any:
    """Every starred recipe, newest star first.

    The same ``limit``/``offset``, the same bounds and the same no-total,
    no-cursor contract as the shelf and the review queue: clients walk pages
    until one comes back short.

    Ordered by the STAR's ``created_at``, not the item's — "what I saved most
    recently" is the question this surface answers, and the two orders diverge
    the moment an old recipe is starred. ``knowledge_item_id`` breaks ties so
    the page walk cannot repeat or skip a row when two stars share a timestamp.
    """
    limit_int = parse_int(
        limit,
        field="limit",
        default=_LIST_LIMIT_DEFAULT,
        minimum=1,
        maximum=_LIST_LIMIT_MAX,
    )
    offset_int = parse_int(
        offset,
        field="offset",
        default=_LIST_OFFSET_DEFAULT,
        minimum=0,
    )

    stmt = (
        select(KnowledgeItem, Document.id, Document.title)
        .join(
            KnowledgeItemFavourite,
            KnowledgeItemFavourite.knowledge_item_id == KnowledgeItem.id,
        )
        .join(Document, KnowledgeItem.document_id == Document.id)
        .order_by(
            KnowledgeItemFavourite.created_at.desc(),
            KnowledgeItemFavourite.knowledge_item_id.desc(),
        )
        .limit(limit_int)
        .offset(offset_int)
    )
    rows = [tuple(row) for row in (await session.execute(stmt)).all()]

    return KnowledgeItemListResponse(knowledge_items=await build_summaries(session, rows))
