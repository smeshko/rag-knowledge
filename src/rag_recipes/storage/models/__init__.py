"""ORM models. Importing this package registers all tables on Base.metadata."""

from __future__ import annotations

from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.chunk_embedding import ChunkEmbedding
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan

__all__ = [
    "Chunk",
    "ChunkEmbedding",
    "Document",
    "ExtractionRun",
    "KnowledgeItem",
    "SourceAsset",
    "SourceSpan",
]
