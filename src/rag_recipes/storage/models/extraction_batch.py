"""ExtractionBatch ORM model — Anthropic Message Batches tracking (Epic 19.2).

One row per submitted (or being-submitted) Anthropic batch. Created at submission
time and linked to the ``ExtractionBatchItem`` windows it carries. ``processing_status``
mirrors the provider lifecycle locally; ``SUBMITTING`` is the durable-claim state the
cron submitter commits *before* the provider call so a crash can never re-submit /
double-charge (DECISIONS #7). Mirrors the ``ExtractionRun`` audit conventions.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from sqlalchemy.sql import func

from rag_recipes.storage.base import Base
from rag_recipes.storage.enums import ExtractionBatchStatus
from rag_recipes.storage.ids import new_id

if TYPE_CHECKING:
    from rag_recipes.storage.models.extraction_batch_item import ExtractionBatchItem


class ExtractionBatch(Base):
    __tablename__ = "extraction_batches"

    ID_PREFIX: ClassVar[str] = "ebatch"

    id: Mapped[str] = mapped_column(
        sa.Text,
        primary_key=True,
        default=lambda: new_id(ExtractionBatch.ID_PREFIX),
    )
    provider: Mapped[str] = mapped_column(sa.Text, nullable=False)
    # The Anthropic ``msgbatch_…`` id, set once the provider accepts the batch
    # (NULL while the row is in the durable-claim SUBMITTING state).
    provider_batch_id: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    model: Mapped[str] = mapped_column(sa.Text, nullable=False)
    processing_status: Mapped[ExtractionBatchStatus] = mapped_column(
        sa.Enum(
            ExtractionBatchStatus,
            name="extraction_batch_status_enum",
            native_enum=True,
            values_callable=lambda enum: [member.value for member in enum],
        ),
        nullable=False,
    )
    request_count: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )

    items: Mapped[list[ExtractionBatchItem]] = relationship(
        "ExtractionBatchItem",
        back_populates="batch",
    )

    @validates("processing_status")
    def _coerce_status(self, key: str, value: Any) -> Any:
        return None if value is None else ExtractionBatchStatus(value)
