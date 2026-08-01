"""Health-check route under /api/v1/health."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from rag_recipes.api.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> Any:
    return HealthResponse(status="ok")
