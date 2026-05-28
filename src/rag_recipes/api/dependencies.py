"""FastAPI dependency factories for settings, DB sessions, Redis, and storage."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.config import Settings
from rag_recipes.config import get_settings as _get_settings
from rag_recipes.providers.file_storage.base import FileStorageProvider
from rag_recipes.providers.file_storage.local import LocalFileStorage

_bearer_scheme = HTTPBearer(auto_error=False)


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


def require_api_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> None:
    expected = settings.personal_api_token
    unauthorized = ApiError(
        status_code=401,
        code=ErrorCode.UNAUTHORIZED,
        message="Authentication required.",
    )
    if not expected:
        raise unauthorized
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise unauthorized
    if not secrets.compare_digest(credentials.credentials, expected):
        raise unauthorized
