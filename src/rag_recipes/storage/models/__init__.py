"""ORM models. Importing this package registers all tables on Base.metadata."""

from __future__ import annotations

from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan

__all__ = ["Document", "SourceAsset", "SourceSpan"]
