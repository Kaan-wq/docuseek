"""
docuseek/api/routes/health.py
------------------------------
Liveness and readiness endpoints.

GET /health/live  — always 200 if the process is up. No dependencies checked.
GET /health/ready — pings Qdrant and verifies app.state is populated.
                    Returns 503 if the pipeline is not ready to serve traffic.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from docuseek.api.schemas import ErrorResponse, HealthLiveResponse, HealthReadyResponse
from docuseek.config import settings

router = APIRouter()
logger = structlog.get_logger()


@router.get(
    "/live",
    response_model=HealthLiveResponse,
    summary="Liveness probe",
)
def live() -> HealthLiveResponse:
    """Return 200 if the process is running. No dependency checks."""
    return HealthLiveResponse(status="ok")


@router.get(
    "/ready",
    response_model=HealthReadyResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Readiness probe",
)
def ready(request: Request) -> HealthReadyResponse | JSONResponse:
    """Return 200 if all components are loaded and Qdrant is reachable.

    Returns 503 if the pipeline is not yet ready to serve traffic — either
    because startup is still in progress or Qdrant is unreachable.
    """
    # Verify app.state was populated by the lifespan
    if not hasattr(request.app.state, "retriever"):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Pipeline not initialised yet."},
        )

    qdrant_ok = _ping_qdrant()
    experiment = request.app.state.config.name

    if not qdrant_ok:
        logger.warning("readiness_check_failed", qdrant=False)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Qdrant is unreachable."},
        )

    return HealthReadyResponse(status="ok", qdrant=qdrant_ok, experiment=experiment)


def _ping_qdrant() -> bool:
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )
        client.get_collections()
        return True  # noqa: TRY300
    except Exception:
        return False
