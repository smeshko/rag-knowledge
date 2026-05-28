"""IngestionFailure ORM model — append-only forensic log for failed jobs."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from rag_recipes.storage.base import Base
from rag_recipes.storage.enums import DocumentStatus
from rag_recipes.storage.ids import new_id

if TYPE_CHECKING:
    from rag_recipes.storage.models.document import Document


class IngestionFailure(Base):
    __tablename__ = "ingestion_failures"

    ID_PREFIX: ClassVar[str] = "fail"

    id: Mapped[str] = mapped_column(
        sa.Text,
        primary_key=True,
        default=lambda: new_id(IngestionFailure.ID_PREFIX),
    )
    document_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    failed_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )
    last_status: Mapped[DocumentStatus] = mapped_column(
        sa.Enum(
            DocumentStatus,
            name="document_status_enum",
            create_type=False,
            values_callable=lambda enum: [member.value for member in enum],
        ),
        nullable=False,
    )
    reason: Mapped[str] = mapped_column(sa.Text, nullable=False)
    error_message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )

    document: Mapped[Document] = relationship("Document", back_populates="failures")
