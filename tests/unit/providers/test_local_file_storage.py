"""Bind LocalFileStorage to the shared FileStorage contract suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_recipes.providers.file_storage.base import FileStorageProvider
from rag_recipes.providers.file_storage.local import LocalFileStorage
from tests.contracts.file_storage import FileStorageContract


class TestLocalFileStorage(FileStorageContract):
    @pytest.fixture
    def provider(self, tmp_path: Path) -> FileStorageProvider:
        return LocalFileStorage(tmp_path)
