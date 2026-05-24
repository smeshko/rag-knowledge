"""ExtractionRun ORM model — doc 2 § 7."""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from rag_recipes.storage.base import Base
from rag_recipes.storage.enums import ExtractionRunStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document


class ExtractionRun(Base):
    __tablename__ = "extraction_runs"

    ID_PREFIX: ClassVar[str] = "run"

    id: Mapped[str] = mapped_column(
        sa.Text,
        primary_key=True,
        default=lambda: new_id(ExtractionRun.ID_PREFIX),
    )
    document_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("documents.id"),
        nullable=False,
    )
    source_version: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    provider: Mapped[str] = mapped_column(sa.Text, nullable=False)
    model: Mapped[str] = mapped_column(sa.Text, nullable=False)
    prompt_version: Mapped[str] = mapped_column(sa.Text, nullable=False)
    schema_version: Mapped[str] = mapped_column(sa.Text, nullable=False)
    input_source_span_ids: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )
    input_hash: Mapped[str] = mapped_column(sa.Text, nullable=False)
    status: Mapped[ExtractionRunStatus] = mapped_column(
        sa.Enum(
            ExtractionRunStatus,
            name="extraction_run_status_enum",
            native_enum=True,
            values_callable=lambda enum: [member.value for member in enum],
        ),
        nullable=False,
    )
    output_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )

    document: Mapped[Document] = relationship(
        "Document",
        back_populates="extraction_runs",
    )
