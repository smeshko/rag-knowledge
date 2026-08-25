"""FastAPI application entrypoint.

The lifespan constructs the async engine, session factory, and Redis
client and attaches them to ``app.state`` — none of these open
connections at construction time, so the app boots without Postgres or
Redis running. The required env vars are validated by
``get_settings()`` when the lifespan starts, not at module import.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis.asyncio import from_url as redis_from_url

from rag_recipes.api.dependencies import require_api_token
from rag_recipes.api.errors import ApiError, ErrorCode, error_body
from rag_recipes.api.routes import (
    answers,
    debug,
    documents,
    health,
    knowledge_items,
    menus,
    review,
    search,
)
from rag_recipes.config import get_settings
from rag_recipes.ingestion.queue import create_arq_pool
from rag_recipes.storage.session import build_engine, build_session_factory


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    redis_client = redis_from_url(settings.redis_url)  # type: ignore[no-untyped-call]
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.redis = redis_client
    arq_redis = None
    try:
        # `create_arq_pool` may raise if Redis is unreachable; the outer
        # `finally` closes `redis_client` and disposes the engine regardless.
        arq_redis = await create_arq_pool(settings)
        app.state.arq_redis = arq_redis
        yield
    finally:
        if arq_redis is not None:
            await arq_redis.aclose()
        await redis_client.aclose()
        await engine.dispose()


async def _handle_api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return JSONResponse(status_code=exc.status_code, content=exc.to_body())


async def _handle_request_validation_error(
    request: Request, exc: Exception
) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return JSONResponse(
        status_code=422,
        content=error_body(
            code=ErrorCode.INVALID_REQUEST,
            message="Request validation failed.",
            # jsonable_encoder (FastAPI's own default-handler idiom): a
            # validator that raises ValueError puts the exception object itself
            # in the error's `ctx`, which json.dumps cannot serialize — without
            # this, a 422 turns into a 500.
            details={"errors": jsonable_encoder(exc.errors())},
        ),
    )


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=error_body(
            code=ErrorCode.INTERNAL_ERROR,
            message="Unexpected server error.",
        ),
    )


app = FastAPI(
    title="rag-recipes",
    lifespan=lifespan,
    dependencies=[Depends(require_api_token)],
    # D4 (Epic 21.1): SearchResponse/AnswerResponse carry a wrap serializer
    # (absent-when-None `debug`), which collapses their *serialization* JSON
    # schema. Documenting responses in validation mode instead keeps their full
    # `properties` in /openapi.json. No-op for today's schema set (no model is
    # used as both request and response, so no -Input/-Output split exists).
    separate_input_output_schemas=False,
)
app.add_exception_handler(ApiError, _handle_api_error)
app.add_exception_handler(RequestValidationError, _handle_request_validation_error)
app.add_exception_handler(Exception, _handle_unexpected_error)
app.include_router(health.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(answers.router, prefix="/api/v1")
app.include_router(menus.router, prefix="/api/v1")
app.include_router(knowledge_items.router, prefix="/api/v1")
app.include_router(review.router, prefix="/api/v1")
# Dev-only debug endpoints: hidden from the production OpenAPI; the router's
# require_debug_enabled gate 404s when debug_endpoints_enabled is false.
app.include_router(debug.router, prefix="/api/v1", include_in_schema=False)
