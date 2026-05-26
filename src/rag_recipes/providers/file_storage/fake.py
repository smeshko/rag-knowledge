"""In-memory FakeFileStorageProvider for tests and local development (doc 13 § 9).

Production code, not test scaffolding: every later epic develops against this
Fake so no test makes a paid or networked storage call.
"""

from __future__ import annotations

from rag_recipes.providers.errors import FileStorageError
from rag_recipes.providers.file_storage.base import FileStorageProvider
from rag_recipes.providers.file_storage.types import StoredObject

__all__ = ["FakeFileStorageProvider"]


class FakeFileStorageProvider(FileStorageProvider):
    """File storage backed by an in-memory dict keyed on the storage key."""

    def __init__(self) -> None:
        self._objects: dict[str, tuple[bytes, str]] = {}

    async def put_object(self, key: str, data: bytes, content_type: str) -> StoredObject:
        self._objects[key] = (data, content_type)
        return StoredObject(
            storage_provider="fake",
            storage_key=key,
            content_type=content_type,
            size_bytes=len(data),
        )

    async def get_object(self, key: str) -> bytes:
        try:
            return self._objects[key][0]
        except KeyError as exc:
            raise FileStorageError(f"no object stored under key {key!r}") from exc

    async def exists(self, key: str) -> bool:
        return key in self._objects

    async def delete_object(self, key: str) -> None:
        self._objects.pop(key, None)
