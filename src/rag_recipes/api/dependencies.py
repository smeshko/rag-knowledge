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
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.openai import OpenAILLMProvider
from rag_recipes.providers.reranker.base import RerankerProvider
from rag_recipes.providers.reranker.openai import OpenAIRerankerProvider

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


def get_llm_provider(
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> LLMProvider:
    """Build the production LLM provider for the query-time answer endpoint.

    Tests override this with a ``FakeLLMProvider``. The default model resolves to
    ``answer_llm_model`` when set, else ``llm_model`` (a class-level default can't
    reference a sibling field, so the fallback lives here at the boundary). Like
    ``get_embedding_provider``, no ``ProviderObservability`` is injected — request-
    time providers are deliberately untraced (answers are synchronous, not a traced
    ingestion job), so the answer layer inherits the provider's retry + parse-error
    handling but not Langfuse tracing (request-time answer tracing is deferred).
    """
    return OpenAILLMProvider(
        settings.openai_api_key,
        default_model=settings.answer_llm_model or settings.llm_model,
        max_rate_limit_retries=settings.llm_max_rate_limit_retries,
        request_timeout=settings.llm_request_timeout_seconds,
    )


def get_reranker_provider(
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> RerankerProvider | None:
    """Build the reranker for the search endpoint, or ``None`` when reranking is off.

    Returns ``None`` (and constructs nothing) when ``reranking_enabled`` is false, so a
    disabled search pays zero added cost. When enabled, dispatches on
    ``rerank_provider`` — only ``"openai"`` is supported in Epic 18.2; any other value
    raises rather than silently routing chunk text to an unintended vendor (a
    ``Settings`` validator already rejects this at load, so this is a defensive
    backstop). Tests override it with ``FakeRerankerProvider``.
    """
    if not settings.reranking_enabled:
        return None
    if settings.rerank_provider == "openai":
        return OpenAIRerankerProvider(
            settings.openai_api_key,
            model=settings.rerank_model,
            request_timeout=settings.rerank_request_timeout_seconds,
        )
    raise ValueError(
        f"unsupported rerank_provider {settings.rerank_provider!r}; only 'openai' "
        "is supported"
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
