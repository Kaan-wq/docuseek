"""
docuseek/api/schemas.py
------------------------
Pydantic request/response schemas for the FastAPI endpoints.
"""

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


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    langfuse: bool
