"""Response schema for GET /api/v1/health (Epic 21.1)."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
