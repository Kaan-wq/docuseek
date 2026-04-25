"""
docuseek/api/schemas.py
------------------------
Pydantic request/response schemas for the FastAPI endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question.")


class SourceInfo(BaseModel):
    title: str
    url: str
    source: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    latency_ms: float
    query_variants: list[str] | None = None


class HealthLiveResponse(BaseModel):
    status: str  # always "ok"


class HealthReadyResponse(BaseModel):
    status: str  # "ok" or "degraded"
    qdrant: bool
    experiment: str


class ErrorResponse(BaseModel):
    detail: str
