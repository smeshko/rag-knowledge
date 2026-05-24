"""ChunkEmbedding ORM model — doc 2 § 6."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from rag_recipes.storage.base import Base, Vector
from rag_recipes.storage.ids import new_id

if TYPE_CHECKING:
    from rag_recipes.storage.models.chunk import Chunk


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    ID_PREFIX: ClassVar[str] = "embedding"

    id: Mapped[str] = mapped_column(
        sa.Text,
        primary_key=True,
        default=lambda: new_id(ChunkEmbedding.ID_PREFIX),
    )
    chunk_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("chunks.id"),
        nullable=False,
    )
    embedding_provider: Mapped[str] = mapped_column(sa.Text, nullable=False)
    embedding_model: Mapped[str] = mapped_column(sa.Text, nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    embedding_vector: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    chunk: Mapped[Chunk] = relationship(
        "Chunk",
        back_populates="embeddings",
    )
