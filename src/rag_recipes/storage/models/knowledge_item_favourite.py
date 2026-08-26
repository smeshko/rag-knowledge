"""KnowledgeItemFavourite ORM model — the reader's own star on a recipe.

A SEPARATE TABLE rather than a ``knowledge_items.favourited_at`` column, and
the reason is ``ingestion.cron.sweep_stuck_indexing_items``: that sweeper
reclaims items on ``status == INDEXING AND updated_at < threshold``, trusting
that nothing but a lifecycle transition touches ``updated_at``. A column would
put a reader's star on the same row, so favouriting an item that happens to be
mid-``indexing`` would push its ``updated_at`` forward and delay the very sweep
that exists to un-stick it. Toggling repeatedly would delay it repeatedly.

Keeping the star off the extraction row also says the true thing about what it
is: an annotation *about* a recipe, owned by the reader, not part of the
recipe's extraction lifecycle. Nothing in ingestion, retrieval or review reads
this table.

Single-user app (ARCHITECTURE.md), so the item id IS the primary key — which is
what makes ``PUT`` idempotent for free (``ON CONFLICT DO NOTHING``). If this
ever grows accounts, the key becomes ``(user_id, knowledge_item_id)`` and the
listing gains a ``WHERE user_id = …``; nothing else here changes.

Deletion needs no code: the FK is ``ON DELETE CASCADE``, so both
``DELETE /knowledge-items/{id}`` and ``DELETE /documents/{id}`` drop the star
with the row they already delete.
"""

from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from rag_recipes.storage.base import Base


class KnowledgeItemFavourite(Base):
    __tablename__ = "knowledge_item_favourites"
    __table_args__ = (
        # Backs the GET /favourites ordering. The listing sorts by
        # (created_at DESC, knowledge_item_id DESC); this single-column index
        # serves the leading key and the PK breaks the (rare) timestamp ties.
        sa.Index(
            "ix_knowledge_item_favourites_created_at",
            sa.text("created_at DESC"),
        ),
    )

    knowledge_item_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("knowledge_items.id", ondelete="CASCADE"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
