"""FileStorageProvider interface (doc 11 § 1)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_recipes.providers.file_storage.types import StoredObject

__all__ = ["FileStorageProvider"]


class FileStorageProvider(ABC):
    """Abstract object store for source PDFs and derived artifacts.

    Async for call-site uniformity across the async ingestion pipeline; a
    blocking real implementation (Epic 4) wraps its I/O in a threadpool.
    Technical failures raise ``FileStorageError`` (``providers.errors``).
    """

    @abstractmethod
    async def put_object(self, key: str, data: bytes, content_type: str) -> StoredObject:
        """Store ``data`` under ``key`` and return its metadata."""

    @abstractmethod
    async def get_object(self, key: str) -> bytes:
        """Return the bytes stored under ``key``."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Return whether an object is stored under ``key``."""

    @abstractmethod
    async def delete_object(self, key: str) -> None:
        """Delete the object stored under ``key``."""
