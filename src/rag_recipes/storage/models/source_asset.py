"""SourceAsset ORM model — doc 2 § 1."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from sqlalchemy.sql import func

from rag_recipes.storage.base import Base
from rag_recipes.storage.enums import SourceType, UploadStatus
from rag_recipes.storage.ids import new_id

if TYPE_CHECKING:
    from rag_recipes.storage.models.document import Document


class SourceAsset(Base):
    __tablename__ = "source_assets"

    ID_PREFIX: ClassVar[str] = "asset"

    id: Mapped[str] = mapped_column(
        sa.Text,
        primary_key=True,
        default=lambda: new_id(SourceAsset.ID_PREFIX),
    )
    source_type: Mapped[SourceType] = mapped_column(
        sa.Enum(
            SourceType,
            name="source_type_enum",
            native_enum=True,
            values_callable=lambda enum: [member.value for member in enum],
        ),
        nullable=False,
    )
    original_filename: Mapped[str] = mapped_column(sa.Text, nullable=False)
    storage_provider: Mapped[str] = mapped_column(sa.Text, nullable=False)
    storage_key: Mapped[str] = mapped_column(sa.Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(sa.Text, nullable=False, unique=True)
    upload_status: Mapped[UploadStatus] = mapped_column(
        sa.Enum(
            UploadStatus,
            name="upload_status_enum",
            native_enum=True,
            values_callable=lambda enum: [member.value for member in enum],
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

    document: Mapped[Document | None] = relationship(
        "Document",
        uselist=False,
        back_populates="asset",
    )

    @validates("source_type")
    def _coerce_source_type(self, key: str, value: Any) -> Any:
        return None if value is None else SourceType(value)

    @validates("upload_status")
    def _coerce_upload_status(self, key: str, value: Any) -> Any:
        return None if value is None else UploadStatus(value)
