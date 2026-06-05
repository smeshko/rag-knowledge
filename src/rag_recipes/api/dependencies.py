"""FastAPI dependency factories for settings, DB sessions, Redis, and storage."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from pathlib import Path

from arq.connections import ArqRedis
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.config import Settings
from rag_recipes.config import get_settings as _get_settings
from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.providers.embeddings.openai import OpenAIEmbeddingProvider
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


def get_arq_redis(request: Request) -> ArqRedis:
    arq_redis: ArqRedis = request.app.state.arq_redis
    return arq_redis


def get_file_storage(
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> FileStorageProvider:
    return LocalFileStorage(Path(settings.local_storage_root))


def get_embedding_provider(
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> EmbeddingProvider:
    """Build the production embedding provider for the search endpoint's vector leg.

    Tests override this with a ``FakeEmbeddingProvider``. The provider is built from
    settings (no Langfuse session at request time — search is synchronous and not a
    traced ingestion job).
    """
    return OpenAIEmbeddingProvider(
        settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
    )


def require_debug_enabled(
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> None:
    """Gate the dev-only debug endpoints on ``Settings.debug_endpoints_enabled``.

    When disabled, raise a plain ``HTTPException(404)`` so the rendered body is
    FastAPI's default ``{"detail": "Not Found"}`` — byte-for-byte identical to a
    genuinely-unknown route for an **authenticated GET** (raising the project
    ``ApiError`` envelope would leak a distinguishing fingerprint). When enabled,
    pass through.

    As a per-request gate (the project relies on runtime toggling, including the
    test harness), this fires after Starlette's route match and the app-level token
    check, so a wrong HTTP method still yields ``405 + Allow`` and a missing token
    still yields ``401`` — exactly as for every other real ``/api/v1`` route, so the
    debug routes carry no debug-specific tell and serve no data either way (review #1).
    """
    if not settings.debug_endpoints_enabled:
        raise HTTPException(status_code=404)


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
