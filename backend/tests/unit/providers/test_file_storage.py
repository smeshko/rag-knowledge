"""Bind FakeFileStorageProvider to the shared FileStorage contract suite."""

from __future__ import annotations

import pytest

from rag_recipes.providers.file_storage.base import FileStorageProvider
from rag_recipes.providers.file_storage.fake import FakeFileStorageProvider
from tests.contracts.file_storage import FileStorageContract


class TestFakeFileStorage(FileStorageContract):
    @pytest.fixture
    def provider(self) -> FileStorageProvider:
        return FakeFileStorageProvider()
