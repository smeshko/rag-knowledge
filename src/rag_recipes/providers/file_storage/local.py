"""LocalFileStorage: filesystem-backed FileStorageProvider (doc 11 § 1).

Reads and writes object bytes under a configurable root directory. Blocking
filesystem I/O is wrapped in ``asyncio.to_thread`` so the async ingestion
pipeline never stalls the event loop. The caller resolves ``local_storage_root``
and passes it as ``root_path``; this class does not read configuration itself.

Keys are validated, not normalised: ``.``/``..``/empty segments are rejected so
distinct keys never alias onto the same file. Writes are atomic (temp file in
the destination directory + ``os.replace``) so a concurrent reader never sees a
partial value; crash durability is out of scope (see plan DECISIONS.md § 3).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
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
        """Validate ``key`` and return the absolute path it maps to under root.

        Rejects empty, absolute, empty-segment, and any ``.``/``..`` segment
        keys, then confirms the joined path stays under ``root_path``.
        """
        if not key or key.startswith("/"):
            raise FileStorageError(f"invalid storage key {key!r}")
        segments = key.split("/")
        if any(seg in ("", ".", "..") for seg in segments):
            raise FileStorageError(f"invalid storage key {key!r}")

        root = self._root_path.resolve()
        path = (root / key).resolve()
        if path != root and root not in path.parents:
            raise FileStorageError(f"storage key {key!r} escapes the storage root")
        return path

    async def put_object(self, key: str, data: bytes, content_type: str) -> StoredObject:
        path = self._resolve(key)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_name, path)
            except BaseException:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp_name)
                raise

        try:
            await asyncio.to_thread(_write)
        except OSError as exc:
            raise FileStorageError(f"failed to write object under key {key!r}") from exc

        return StoredObject(
            storage_provider="local",
            storage_key=key,
            content_type=content_type,
            size_bytes=len(data),
        )

    async def get_object(self, key: str) -> bytes:
        path = self._resolve(key)

        def _read() -> bytes:
            if not path.is_file():
                raise FileStorageError(f"no object stored under key {key!r}")
            return path.read_bytes()

        try:
            return await asyncio.to_thread(_read)
        except OSError as exc:
            raise FileStorageError(f"no object stored under key {key!r}") from exc

    async def exists(self, key: str) -> bool:
        path = self._resolve(key)
        return await asyncio.to_thread(path.is_file)

    async def delete_object(self, key: str) -> None:
        path = self._resolve(key)

        def _delete() -> None:
            if path.is_file():
                path.unlink()

        await asyncio.to_thread(_delete)
