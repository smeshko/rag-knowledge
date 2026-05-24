"""Document ORM model — doc 2 § 2."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from sqlalchemy.sql import func

from rag_recipes.storage.base import Base
from rag_recipes.storage.enums import DocumentStatus, SourceType
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.source_asset import SourceAsset

if TYPE_CHECKING:
    from rag_recipes.storage.models.extraction_run import ExtractionRun
    from rag_recipes.storage.models.source_span import SourceSpan


class Document(Base):
    __tablename__ = "documents"

    ID_PREFIX: ClassVar[str] = "doc"

    id: Mapped[str] = mapped_column(
        sa.Text,
        primary_key=True,
        default=lambda: new_id(Document.ID_PREFIX),
    )
    asset_id: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("source_assets.id"),
        nullable=False,
        unique=True,
    )
    category: Mapped[str] = mapped_column(sa.Text, nullable=False)
    subcategory: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    title: Mapped[str] = mapped_column(sa.Text, nullable=False)
    author: Mapped[str] = mapped_column(sa.Text, nullable=False)
    source_type: Mapped[SourceType] = mapped_column(
        sa.Enum(SourceType, name="source_type_enum", create_type=False),
        nullable=False,
    )
    language: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    active_source_version: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    status: Mapped[DocumentStatus] = mapped_column(
        sa.Enum(DocumentStatus, name="document_status_enum", native_enum=True),
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

    asset: Mapped[SourceAsset] = relationship(
        "SourceAsset",
        back_populates="document",
    )
    source_spans: Mapped[list[SourceSpan]] = relationship(
        "SourceSpan",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    extraction_runs: Mapped[list[ExtractionRun]] = relationship(
        "ExtractionRun",
        back_populates="document",
    )

    @validates("source_type", "asset")
    def _validate_source_type_matches_asset(self, key: str, value: Any) -> Any:
        if key == "source_type":
            asset = getattr(self, "asset", None)
            if asset is not None and asset.source_type != value:
                raise ValueError(
                    f"Document.source_type={value!r} does not match "
                    f"SourceAsset.source_type={asset.source_type!r}"
                )
        else:  # key == "asset"
            current_type = getattr(self, "source_type", None)
            if value is not None and current_type is not None and value.source_type != current_type:
                raise ValueError(
                    f"SourceAsset.source_type={value.source_type!r} does not match "
                    f"Document.source_type={current_type!r}"
                )
        return value
