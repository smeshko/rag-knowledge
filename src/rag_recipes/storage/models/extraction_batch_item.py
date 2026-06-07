"""ExtractionBatchItem ORM model — per-window batch registration (Epic 19.2).

One row per extraction window deferred into a batch. Registered (``PENDING``) by
``process_document(batch_mode=True)`` *before* any batch exists, then claimed
(``SUBMITTING`` → ``SUBMITTED``) by the cron submitter. The row carries everything
submission needs — ``request_input`` (rendered prompt) **and** ``request_schema``
(the sanitized ``recipe.v1`` JSON captured at registration) — so submission is a
pure, drift-proof transform with no ``SourceSpan`` reload and no schema rebuild
(DECISIONS #5). The ``id`` (``ebitem_<ULID>``) doubles as the Anthropic
``custom_id`` and so as the result-routing key in 19.3.

The partial-unique index on ``(document_id, source_version, input_hash)`` for
non-terminal statuses is the load-bearing idempotency guard: it backstops the
read-before-insert check in registration against concurrent / re-delivered
``process_document`` runs (DECISIONS #4).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from sqlalchemy.sql import func

from rag_recipes.storage.base import Base
from rag_recipes.storage.enums import ExtractionBatchItemStatus
from rag_recipes.storage.ids import new_id

if TYPE_CHECKING:
    from rag_recipes.storage.models.document import Document
    from rag_recipes.storage.models.extraction_batch import ExtractionBatch


class ExtractionBatchItem(Base):
    __tablename__ = "extraction_batch_items"
    __table_args__ = (
        sa.Index("ix_extraction_batch_items_status", "status"),
        sa.Index(
            "ix_extraction_batch_items_document_source_version",
            "document_id",
            "source_version",
        ),
        # Idempotency backstop: at most one non-terminal item per
        # (document, source_version, input_hash). Registration relies on this
        # under concurrency (DECISIONS #4); a terminal item does not block a
        # fresh re-registration of the same window.
        sa.Index(
            "uq_extraction_batch_items_doc_version_hash_active",
            "document_id",
            "source_version",
            "input_hash",
            unique=True,
            postgresql_where=sa.text(
                "status IN ('pending', 'submitting', 'submitted')"
            ),
        ),
    )

    ID_PREFIX: ClassVar[str] = "ebitem"

    id: Mapped[str] = mapped_column(
        sa.Text,
        primary_key=True,
        default=lambda: new_id(ExtractionBatchItem.ID_PREFIX),
    )
    document_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("documents.id"),
        nullable=False,
    )
    source_version: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    input_hash: Mapped[str] = mapped_column(sa.Text, nullable=False)
    input_source_span_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    # The rendered prompt and the sanitized recipe.v1 schema, captured at
    # registration so submission needs no rebuild (DECISIONS #5).
    request_input: Mapped[str] = mapped_column(sa.Text, nullable=False)
    request_schema: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    prompt_version: Mapped[str] = mapped_column(sa.Text, nullable=False)
    schema_version: Mapped[str] = mapped_column(sa.Text, nullable=False)
    status: Mapped[ExtractionBatchItemStatus] = mapped_column(
        sa.Enum(
            ExtractionBatchItemStatus,
            name="extraction_batch_item_status_enum",
            native_enum=True,
            values_callable=lambda enum: [member.value for member in enum],
        ),
        nullable=False,
    )
    # NULL until the item is claimed into a SUBMITTING batch (DECISIONS #4, #7).
    batch_id: Mapped[str | None] = mapped_column(
        sa.Text,
        sa.ForeignKey("extraction_batches.id"),
        nullable=True,
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

    batch: Mapped[ExtractionBatch | None] = relationship(
        "ExtractionBatch",
        back_populates="items",
    )
    document: Mapped[Document] = relationship("Document")

    @validates("status")
    def _coerce_status(self, key: str, value: Any) -> Any:
        return None if value is None else ExtractionBatchItemStatus(value)
