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

from fastapi import FastAPI
from redis.asyncio import from_url as redis_from_url

from rag_recipes.api.routes import documents, health
from rag_recipes.config import get_settings
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
    try:
        yield
    finally:
        await redis_client.aclose()
        await engine.dispose()


app = FastAPI(title="rag-recipes", lifespan=lifespan)
app.include_router(health.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
