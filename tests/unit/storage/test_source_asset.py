"""Unit tests for rag_recipes.storage.models.source_asset."""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from rag_recipes.storage.enums import SourceType, UploadStatus
from rag_recipes.storage.models import SourceAsset


def test_tablename() -> None:
    assert SourceAsset.__tablename__ == "source_assets"


def test_table_columns_match_doc_2() -> None:
    expected = [
        "id",
        "source_type",
        "original_filename",
        "storage_provider",
        "storage_key",
        "content_hash",
        "upload_status",
        "created_at",
        "updated_at",
    ]
    assert list(SourceAsset.__table__.columns.keys()) == expected


def test_default_id_uses_asset_prefix() -> None:
    default = SourceAsset.__table__.columns["id"].default
    assert default is not None
    value = default.arg(None)
    assert value.startswith("asset_")


def test_content_hash_is_unique() -> None:
    assert SourceAsset.__table__.columns["content_hash"].unique is True


def test_source_type_uses_native_enum() -> None:
    col_type = SourceAsset.__table__.columns["source_type"].type
    assert isinstance(col_type, sa.Enum)
    assert col_type.name == "source_type_enum"
    assert col_type.native_enum is True
    assert col_type.enum_class is SourceType


def test_upload_status_uses_native_enum() -> None:
    col_type = SourceAsset.__table__.columns["upload_status"].type
    assert isinstance(col_type, sa.Enum)
    assert col_type.name == "upload_status_enum"
    assert col_type.native_enum is True
    assert col_type.enum_class is UploadStatus


def test_invalid_upload_status_raises() -> None:
    with pytest.raises(ValueError):
        SourceAsset(
            source_type=SourceType.PDF,
            original_filename="x.pdf",
            storage_provider="s3",
            storage_key="k",
            content_hash="h",
            upload_status="invalid_value",  # type: ignore[arg-type]
        )


def test_invalid_source_type_raises() -> None:
    with pytest.raises(ValueError):
        SourceAsset(source_type="invalid_value")  # type: ignore[arg-type]


def test_timestamps_have_server_defaults() -> None:
    created_at = SourceAsset.__table__.columns["created_at"]
    updated_at = SourceAsset.__table__.columns["updated_at"]
    assert created_at.server_default is not None
    assert updated_at.server_default is not None
    assert updated_at.onupdate is not None
