"""GET /api/v1/health – Service health check."""

import logging
from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    triton_ready: bool
    gallery_ready: bool
    gallery_size: int


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if Triton and gallery index are ready.",
)
async def health(request: Request):
    triton_ready   = request.app.state.triton_client.is_ready()
    retrieval      = request.app.state.retrieval_service
    gallery_ready  = retrieval.is_ready()
    gallery_size   = retrieval.gallery_size()

    overall = "ok" if (triton_ready and gallery_ready) else "degraded"

    return HealthResponse(
        status=overall,
        triton_ready=triton_ready,
        gallery_ready=gallery_ready,
        gallery_size=gallery_size,
    )
