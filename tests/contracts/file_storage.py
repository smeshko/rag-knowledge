"""Reusable contract suite for FileStorageProvider implementations (doc 12 § 2).

Subclasses bind a concrete provider by overriding the ``provider`` fixture.
Named ``…Contract`` (not ``Test…``) so pytest does not collect the abstract
base directly — only the ``Test*`` subclasses that supply ``provider`` run.
"""

from __future__ import annotations

import pytest

from rag_recipes.providers.errors import FileStorageError
from rag_recipes.providers.file_storage.base import FileStorageProvider

__all__ = ["FileStorageContract"]


class FileStorageContract:
    """Interface guarantees every FileStorageProvider must satisfy."""

    @pytest.fixture
    def provider(self) -> FileStorageProvider:
        raise NotImplementedError("subclasses must override the `provider` fixture")

    async def test_put_then_get_roundtrip(self, provider: FileStorageProvider) -> None:
        await provider.put_object("k", b"hello", "text/plain")
        assert await provider.get_object("k") == b"hello"

    async def test_exists_true_after_put_false_before(
        self, provider: FileStorageProvider
    ) -> None:
        assert await provider.exists("k") is False
        await provider.put_object("k", b"data", "application/octet-stream")
        assert await provider.exists("k") is True

    async def test_delete_removes_object(self, provider: FileStorageProvider) -> None:
        await provider.put_object("k", b"data", "application/octet-stream")
        await provider.delete_object("k")
        assert await provider.exists("k") is False

    async def test_delete_missing_is_idempotent(self, provider: FileStorageProvider) -> None:
        await provider.delete_object("never-stored")

    async def test_duplicate_key_overwrites(self, provider: FileStorageProvider) -> None:
        await provider.put_object("k", b"first", "text/plain")
        await provider.put_object("k", b"second", "text/plain")
        assert await provider.get_object("k") == b"second"

    async def test_get_missing_raises(self, provider: FileStorageProvider) -> None:
        with pytest.raises(FileStorageError):
            await provider.get_object("missing")

    async def test_put_returns_stored_object(self, provider: FileStorageProvider) -> None:
        stored = await provider.put_object("k", b"payload", "application/pdf")
        assert stored.storage_key == "k"
        assert stored.content_type == "application/pdf"
        assert stored.size_bytes == len(b"payload")
