"""FastAPI dependency factories for settings, DB sessions, Redis, and storage."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.config import Settings
from rag_recipes.config import get_settings as _get_settings
from rag_recipes.providers.file_storage.base import FileStorageProvider
from rag_recipes.providers.file_storage.local import LocalFileStorage


def get_settings() -> Settings:
    return _get_settings()


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        yield session


def get_redis(request: Request) -> Redis:
    redis_client: Redis = request.app.state.redis
    return redis_client


def get_file_storage(
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> FileStorageProvider:
    return LocalFileStorage(Path(settings.local_storage_root))
