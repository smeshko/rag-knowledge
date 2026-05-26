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
import stat
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
        """Validate ``key`` and return the lexical path it maps to under root.

        Rejects empty, absolute, empty-segment, and any ``.``/``..`` segment
        keys, then assembles the path lexically from ``root_path``. The path is
        *not* run through ``Path.resolve()``: resolving would follow symlinks
        and let one key alias onto another object's bytes. Because no segment is
        ``..``/``.``/empty, the lexical join is guaranteed to stay under
        ``root_path``; symlink components are rejected at I/O time in
        ``_verify_no_symlink``.
        """
        if not key or key.startswith("/"):
            raise FileStorageError(f"invalid storage key {key!r}")
        segments = key.split("/")
        if any(seg in ("", ".", "..") for seg in segments):
            raise FileStorageError(f"invalid storage key {key!r}")

        return self._root_path.joinpath(*segments)

    def _verify_no_symlink(self, key: str, path: Path) -> None:
        """Reject if any component between ``root_path`` and ``path`` is a symlink.

        Resolving keys lexically keeps distinct keys from aliasing, but
        ``open``/``stat``/``unlink`` still follow a symlink planted inside the
        root by an external actor (restore, manual ops). Walking the components
        with ``is_symlink`` (an ``lstat``, which does not follow) and rejecting
        any link closes that aliasing/escape path. Missing components are not
        symlinks, so this is safe to call before ``put_object`` creates parents.
        Run inside the I/O worker to keep the check-to-use window minimal.
        """
        current = path
        while current != self._root_path:
            if current == current.parent:
                raise FileStorageError(f"storage key {key!r} escapes the storage root")
            if current.is_symlink():
                raise FileStorageError(f"storage key {key!r} resolves through a symlink")
            current = current.parent

    async def put_object(self, key: str, data: bytes, content_type: str) -> StoredObject:
        path = self._resolve(key)

        def _write() -> None:
            self._verify_no_symlink(key, path)
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
            self._verify_no_symlink(key, path)
            if not path.is_file():
                raise FileStorageError(f"no object stored under key {key!r}")
            return path.read_bytes()

        try:
            return await asyncio.to_thread(_read)
        except OSError as exc:
            raise FileStorageError(f"no object stored under key {key!r}") from exc

    async def exists(self, key: str) -> bool:
        path = self._resolve(key)

        def _exists() -> bool:
            self._verify_no_symlink(key, path)
            try:
                return stat.S_ISREG(os.stat(path).st_mode)
            except (FileNotFoundError, NotADirectoryError):
                return False
            except OSError as exc:
                raise FileStorageError(f"failed to stat object under key {key!r}") from exc

        return await asyncio.to_thread(_exists)

    async def delete_object(self, key: str) -> None:
        path = self._resolve(key)

        def _delete() -> None:
            self._verify_no_symlink(key, path)
            try:
                if not stat.S_ISREG(os.stat(path).st_mode):
                    return
                os.unlink(path)
            except (FileNotFoundError, NotADirectoryError):
                return
            except OSError as exc:
                raise FileStorageError(f"failed to delete object under key {key!r}") from exc

        await asyncio.to_thread(_delete)
