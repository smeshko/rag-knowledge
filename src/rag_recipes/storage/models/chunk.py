"""Chunk ORM model — doc 2 § 5."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from sqlalchemy.sql import func

from rag_recipes.storage.base import Base
from rag_recipes.storage.enums import ChunkParentType, ChunkType
from rag_recipes.storage.ids import new_id

if TYPE_CHECKING:
    from rag_recipes.storage.models.chunk_embedding import ChunkEmbedding
    from rag_recipes.storage.models.knowledge_item import KnowledgeItem


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        # Composite FK enforces document_id agreement with the parent knowledge item
        # even on raw-ID writes that bypass the @validates relationship check.
        sa.ForeignKeyConstraint(
            ["parent_id", "document_id"],
            ["knowledge_items.id", "knowledge_items.document_id"],
            name="fk_chunks_parent_document",
        ),
    )

    ID_PREFIX: ClassVar[str] = "chunk"

    id: Mapped[str] = mapped_column(
        sa.Text,
        primary_key=True,
        default=lambda: new_id(Chunk.ID_PREFIX),
    )
    document_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("documents.id"),
        nullable=False,
    )
    parent_type: Mapped[ChunkParentType] = mapped_column(
        sa.Enum(
            ChunkParentType,
            name="chunk_parent_type_enum",
            native_enum=True,
            values_callable=lambda enum: [member.value for member in enum],
            validate_strings=True,
        ),
        nullable=False,
    )
    parent_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("knowledge_items.id"),
        nullable=False,
    )
    chunk_type: Mapped[ChunkType] = mapped_column(
        sa.Enum(
            ChunkType,
            name="chunk_type_enum",
            native_enum=True,
            values_callable=lambda enum: [member.value for member in enum],
            validate_strings=True,
        ),
        nullable=False,
    )
    text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(sa.Text, nullable=False)
    source_span_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    # `metadata` collides with DeclarativeBase.metadata; map the column under chunk_metadata.
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    knowledge_item: Mapped[KnowledgeItem] = relationship(
        "KnowledgeItem",
        back_populates="chunks",
        foreign_keys="Chunk.parent_id",
    )
    embeddings: Mapped[list[ChunkEmbedding]] = relationship(
        "ChunkEmbedding",
        back_populates="chunk",
    )

    @validates("document_id", "knowledge_item")
    def _validate_document_id_matches_parent(self, key: str, value: Any) -> Any:
        if key == "document_id":
            item = getattr(self, "knowledge_item", None)
            if item is not None and item.document_id != value:
                raise ValueError(
                    f"Chunk.document_id {value!r} does not match "
                    f"KnowledgeItem.document_id {item.document_id!r}"
                )
        else:  # key == "knowledge_item"
            current = getattr(self, "document_id", None)
            if value is not None and current is not None and value.document_id != current:
                raise ValueError(
                    f"KnowledgeItem.document_id {value.document_id!r} does not match "
                    f"Chunk.document_id {current!r}"
                )
        return value

    @validates("parent_type")
    def _coerce_parent_type(self, key: str, value: Any) -> Any:
        return None if value is None else ChunkParentType(value)

    @validates("chunk_type")
    def _coerce_chunk_type(self, key: str, value: Any) -> Any:
        return None if value is None else ChunkType(value)
