"""SourceSpan ORM model — doc 2 § 3."""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from sqlalchemy.sql import func

from rag_recipes.storage.base import Base
from rag_recipes.storage.enums import SourceType
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document


class SourceSpan(Base):
    __tablename__ = "source_spans"
    __table_args__ = (
        sa.UniqueConstraint(
            "document_id",
            "source_version",
            "locator_hash",
            name="uq_source_spans_document_version_locator",
        ),
    )

    ID_PREFIX: ClassVar[str] = "span"

    id: Mapped[str] = mapped_column(
        sa.Text,
        primary_key=True,
        default=lambda: new_id(SourceSpan.ID_PREFIX),
    )
    document_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("documents.id"),
        nullable=False,
    )
    source_version: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    source_type: Mapped[SourceType] = mapped_column(
        sa.Enum(
            SourceType,
            name="source_type_enum",
            create_type=False,
            values_callable=lambda enum: [member.value for member in enum],
        ),
        nullable=False,
    )
    locator: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    locator_hash: Mapped[str] = mapped_column(sa.Text, nullable=False)
    text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(sa.Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    document: Mapped[Document] = relationship(
        "Document",
        back_populates="source_spans",
    )

    @validates("source_type", "document")
    def _validate_source_type_matches_document(self, key: str, value: Any) -> Any:
        if key == "source_type":
            doc = getattr(self, "document", None)
            if doc is not None and doc.source_type != value:
                raise ValueError(
                    f"SourceSpan.source_type={value!r} does not match "
                    f"Document.source_type={doc.source_type!r}"
                )
        else:  # key == "document"
            current_type = getattr(self, "source_type", None)
            if value is not None and current_type is not None and value.source_type != current_type:
                raise ValueError(
                    f"Document.source_type={value.source_type!r} does not match "
                    f"SourceSpan.source_type={current_type!r}"
                )
        return value
