"""KnowledgeItem ORM model — doc 2 § 4."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from sqlalchemy.sql import func

from rag_recipes.storage.base import Base
from rag_recipes.storage.enums import KnowledgeItemStatus
from rag_recipes.storage.ids import new_id

if TYPE_CHECKING:
    from rag_recipes.storage.models.chunk import Chunk
    from rag_recipes.storage.models.document import Document
    from rag_recipes.storage.models.extraction_run import ExtractionRun


class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"

    ID_PREFIX: ClassVar[str] = "item"

    id: Mapped[str] = mapped_column(
        sa.Text,
        primary_key=True,
        default=lambda: new_id(KnowledgeItem.ID_PREFIX),
    )
    document_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("documents.id"),
        nullable=False,
    )
    extraction_run_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("extraction_runs.id"),
        nullable=False,
    )
    source_version: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    item_type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    title: Mapped[str] = mapped_column(sa.Text, nullable=False)
    normalized_title: Mapped[str] = mapped_column(sa.Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    body_text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    source_span_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    structured_data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )
    confidence: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[KnowledgeItemStatus] = mapped_column(
        sa.Enum(
            KnowledgeItemStatus,
            name="knowledge_item_status_enum",
            native_enum=True,
            values_callable=lambda enum: [member.value for member in enum],
            validate_strings=True,
        ),
        nullable=False,
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

    document: Mapped[Document] = relationship(
        "Document",
        back_populates="knowledge_items",
    )
    extraction_run: Mapped[ExtractionRun] = relationship(
        "ExtractionRun",
        back_populates="knowledge_items",
    )
    chunks: Mapped[list[Chunk]] = relationship(
        "Chunk",
        back_populates="knowledge_item",
    )

    @validates("status")
    def _coerce_status(self, key: str, value: Any) -> Any:
        return None if value is None else KnowledgeItemStatus(value)
