"""ORM models. Importing this package registers all tables on Base.metadata."""

from __future__ import annotations

from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.source_asset import SourceAsset

__all__ = ["Document", "SourceAsset"]
