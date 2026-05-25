"""Pydantic payload types for the file-storage provider (doc 11 § 1)."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["StoredObject"]


class StoredObject(BaseModel):
    """Metadata for an object persisted by a FileStorageProvider.

    ``storage_provider`` / ``storage_key`` mirror the ``SourceAsset`` columns of
    the same name so callers can persist the result directly.
    """

    storage_provider: str
    storage_key: str
    content_type: str
    size_bytes: int
