"""LocalFileStorage: contract binding plus filesystem-specific guarantees."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from rag_recipes.providers.errors import FileStorageError
from rag_recipes.providers.file_storage.base import FileStorageProvider
from rag_recipes.providers.file_storage.local import LocalFileStorage
from tests.contracts.file_storage import FileStorageContract


class TestLocalFileStorage(FileStorageContract):
    @pytest.fixture
    def provider(self, tmp_path: Path) -> FileStorageProvider:
        return LocalFileStorage(tmp_path)


@pytest.fixture
def storage(tmp_path: Path) -> LocalFileStorage:
    return LocalFileStorage(tmp_path)


@pytest.mark.parametrize("key", ["../escape", "a/../../escape"])
async def test_rejects_parent_traversal(storage: LocalFileStorage, key: str) -> None:
    with pytest.raises(FileStorageError):
        await storage.put_object(key, b"x", "text/plain")


async def test_rejects_absolute_key(storage: LocalFileStorage) -> None:
    with pytest.raises(FileStorageError):
        await storage.put_object("/etc/passwd", b"x", "text/plain")


async def test_rejects_key_escaping_root(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path / "root")
    # A key that, once joined, climbs out of the configured root.
    with pytest.raises(FileStorageError):
        await storage.put_object("../sibling/file", b"x", "text/plain")


@pytest.mark.parametrize("key", ["", ".", "..", "a//b", "./a", "a/./b", "a/."])
async def test_rejects_empty_and_dot_keys(storage: LocalFileStorage, key: str) -> None:
    with pytest.raises(FileStorageError):
        await storage.put_object(key, b"x", "text/plain")


async def test_symlink_key_cannot_alias_another_object(storage: LocalFileStorage) -> None:
    await storage.put_object("k", b"secret", "text/plain")
    root = storage._root_path
    (root / "link").symlink_to(root / "k")

    with pytest.raises(FileStorageError):
        await storage.get_object("link")
    with pytest.raises(FileStorageError):
        await storage.put_object("link", b"overwrite", "text/plain")
    with pytest.raises(FileStorageError):
        await storage.delete_object("link")
    with pytest.raises(FileStorageError):
        await storage.exists("link")

    # The aliased object is untouched.
    assert await storage.get_object("k") == b"secret"


async def test_symlinked_dir_component_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    storage = LocalFileStorage(root)
    (root / "sub").symlink_to(outside)

    with pytest.raises(FileStorageError):
        await storage.put_object("sub/file", b"x", "text/plain")
    with pytest.raises(FileStorageError):
        await storage.get_object("sub/file")
    assert not (outside / "file").exists()


async def test_lstat_failure_surfaces_as_file_storage_error(
    storage: LocalFileStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_self: Path) -> bool:
        raise PermissionError("injected lstat failure")

    monkeypatch.setattr(Path, "is_symlink", boom)

    with pytest.raises(FileStorageError):
        await storage.exists("k")
    with pytest.raises(FileStorageError):
        await storage.delete_object("k")
    with pytest.raises(FileStorageError):
        await storage.get_object("k")
    with pytest.raises(FileStorageError):
        await storage.put_object("k", b"x", "text/plain")


async def test_prefix_is_not_an_object(storage: LocalFileStorage) -> None:
    await storage.put_object("a/b/c.pdf", b"payload", "application/pdf")

    assert await storage.exists("a") is False
    assert await storage.exists("a/b") is False

    with pytest.raises(FileStorageError):
        await storage.get_object("a")

    await storage.delete_object("a")
    assert await storage.get_object("a/b/c.pdf") == b"payload"


async def test_concurrent_put_no_torn_read(storage: LocalFileStorage) -> None:
    values = [bytes([i]) * 4096 for i in range(1, 33)]

    async def reader() -> None:
        for _ in range(200):
            try:
                data = await storage.get_object("k")
            except FileStorageError:
                continue
            assert len(data) == 4096
            assert data == bytes([data[0]]) * 4096

    writers = [storage.put_object("k", v, "application/octet-stream") for v in values]
    await asyncio.gather(*writers, reader(), reader())

    final = await storage.get_object("k")
    assert final == bytes([final[0]]) * 4096


async def test_failed_put_leaves_no_temp_and_no_partial(
    storage: LocalFileStorage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    await storage.put_object("k", b"original", "text/plain")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(FileStorageError):
        await storage.put_object("k", b"new-value", "text/plain")

    assert await storage.get_object("k") == b"original"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["k"]


async def test_cancelled_put_is_atomic_after_quiescence(storage: LocalFileStorage) -> None:
    await storage.put_object("k", b"original", "text/plain")

    task = asyncio.create_task(storage.put_object("k", b"new-value", "text/plain"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Let any in-flight worker thread go quiescent before asserting.
    await asyncio.sleep(0.05)

    final = await storage.get_object("k")
    assert final in (b"original", b"new-value")
    root = storage._root_path
    assert sorted(p.name for p in root.iterdir()) == ["k"]


async def test_many_objects_no_handle_leak(storage: LocalFileStorage) -> None:
    count = 500
    for i in range(count):
        await storage.put_object(f"obj-{i}", f"value-{i}".encode(), "text/plain")
    for i in range(count):
        assert await storage.get_object(f"obj-{i}") == f"value-{i}".encode()
