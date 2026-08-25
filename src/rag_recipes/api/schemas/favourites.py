"""Schemas for the favourites surface.

The listing reuses ``KnowledgeItemListResponse`` — a favourited recipe is the
same row the shelf and the review queue render, so it is projected by the same
``knowledge_item_list.build_summaries`` and cannot drift from them. Only the
toggle's own tiny acknowledgement lives here.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class Favourite(BaseModel):
    knowledge_item_id: str
    #: When the star was FIRST set. A repeated ``PUT`` returns this unchanged
    #: rather than restamping it, so the listing order is stable under a
    #: double-click.
    favourited_at: datetime


class FavouriteResponse(BaseModel):
    favourite: Favourite
