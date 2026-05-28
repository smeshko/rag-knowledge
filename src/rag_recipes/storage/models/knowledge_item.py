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
    __table_args__ = (
        # Backs the composite FK from chunks(parent_id, document_id).
        sa.UniqueConstraint("id", "document_id", name="uq_knowledge_items_id_document"),
        # Composite FK enforces document_id AND source_version agreement with the
        # extraction run even on raw-ID writes that bypass the @validates check.
        # source_version is copied from the accepted run (doc 2 § core data model).
        sa.ForeignKeyConstraint(
            ["extraction_run_id", "document_id", "source_version"],
            [
                "extraction_runs.id",
                "extraction_runs.document_id",
                "extraction_runs.source_version",
            ],
            name="fk_knowledge_items_extraction_run_document_version",
        ),
        sa.Index("ix_knowledge_items_document_id", "document_id"),
        sa.Index("ix_knowledge_items_status", "status"),
    )

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
        foreign_keys="KnowledgeItem.extraction_run_id",
    )
    chunks: Mapped[list[Chunk]] = relationship(
        "Chunk",
        back_populates="knowledge_item",
        foreign_keys="Chunk.parent_id",
    )

    @validates("document_id", "extraction_run")
    def _validate_document_id_matches_run(self, key: str, value: Any) -> Any:
        if key == "document_id":
            run = getattr(self, "extraction_run", None)
            if run is not None and run.document_id != value:
                raise ValueError(
                    f"KnowledgeItem.document_id {value!r} does not match "
                    f"ExtractionRun.document_id {run.document_id!r}"
                )
        else:  # key == "extraction_run"
            current = getattr(self, "document_id", None)
            if value is not None and current is not None and value.document_id != current:
                raise ValueError(
                    f"ExtractionRun.document_id {value.document_id!r} does not match "
                    f"KnowledgeItem.document_id {current!r}"
                )
        return value

    @validates("status")
    def _coerce_status(self, key: str, value: Any) -> Any:
        return None if value is None else KnowledgeItemStatus(value)
