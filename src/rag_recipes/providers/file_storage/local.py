"""LocalFileStorage: filesystem-backed FileStorageProvider (doc 11 § 1).

Reads and writes object bytes under a configurable root directory. Blocking
filesystem I/O is wrapped in ``asyncio.to_thread`` so the async ingestion
pipeline never stalls the event loop. The caller resolves ``local_storage_root``
and passes it as ``root_path``; this class does not read configuration itself.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rag_recipes.providers.errors import FileStorageError
from rag_recipes.providers.file_storage.base import FileStorageProvider
from rag_recipes.providers.file_storage.types import StoredObject

__all__ = ["LocalFileStorage"]


class LocalFileStorage(FileStorageProvider):
    """Object store backed by files under ``root_path``."""

    def __init__(self, root_path: Path) -> None:
        self._root_path = root_path

    def _resolve(self, key: str) -> Path:
        return self._root_path / key

    async def put_object(self, key: str, data: bytes, content_type: str) -> StoredObject:
        def _write() -> None:
            path = self._resolve(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        await asyncio.to_thread(_write)
        return StoredObject(
            storage_provider="local",
            storage_key=key,
            content_type=content_type,
            size_bytes=len(data),
        )

    async def get_object(self, key: str) -> bytes:
        def _read() -> bytes:
            return self._resolve(key).read_bytes()

        try:
            return await asyncio.to_thread(_read)
        except OSError as exc:
            raise FileStorageError(f"no object stored under key {key!r}") from exc

    async def exists(self, key: str) -> bool:
        return await asyncio.to_thread(self._resolve(key).is_file)

    async def delete_object(self, key: str) -> None:
        await asyncio.to_thread(self._resolve(key).unlink, missing_ok=True)
