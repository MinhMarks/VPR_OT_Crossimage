"""
POST /api/v1/retrieve – Query image → Top-K matching places.

Example:
    curl -X POST http://localhost:8080/api/v1/retrieve \
         -H "X-API-Key: dev-secret-key" \
         -F "image=@query.jpg" \
         -F "top_k=5"
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import BaseModel, Field

from api.config import Settings
from api.middleware.auth import require_api_key
from api.services.preprocess import preprocess_image
from api.services.triton_client import TritonVPRClient
from api.services.retrieval import RetrievalService, RetrievalResult
from api.services.metrics import (
    TRITON_INFERENCE_LATENCY,
    FAISS_RETRIEVAL_LATENCY,
    PREPROCESS_LATENCY,
    REQUESTS_TOTAL,
    ERRORS_TOTAL,
    DESCRIPTOR_DIM,
    GALLERY_SIZE,
)

logger = logging.getLogger(__name__)
settings = Settings()
router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Response schemas
# ──────────────────────────────────────────────────────────────────────────────

class PlaceMatch(BaseModel):
    rank: int           = Field(..., description="Rank (1 = best match)")
    gallery_id: int     = Field(..., description="Gallery image index")
    score: float        = Field(..., description="Similarity score [0, 1]")
    distance: float     = Field(..., description="L2 distance (lower = better)")
    image_path: str     = Field(..., description="Path to matched gallery image")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata (GPS, place name, etc.)")


class RetrieveResponse(BaseModel):
    query_descriptor_dim: int           = Field(..., description="Descriptor dimensionality")
    gallery_size: int                   = Field(..., description="Total gallery images")
    top_k: int                          = Field(..., description="Number of results returned")
    matches: List[PlaceMatch]           = Field(..., description="Ranked matching results")


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Query place recognition",
    description=(
        "Upload a query image and receive the Top-K matching places "
        "from the gallery database, ranked by similarity."
    ),
)
async def retrieve(
    request: Request,
    images: List[UploadFile] = File(...,  description="Query images (at least 1, up to N)"),
    top_k: int         = Form(default=5, ge=1, le=20, description="Number of results"),
    _: str             = Depends(require_api_key),
):
    # ── Read upload ──────────────────────────────────────────────────────────
    image_bytes_list = []
    for image in images:
        b = await image.read()
        image_bytes_list.append(b)

    logger.info(
        f"Retrieve request | num_images={len(images)} top_k={top_k}"
    )

    try:
        # ── Preprocess ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        
        # Preprocess all uploaded images
        import numpy as np
        processed_images = [preprocess_image(b) for b in image_bytes_list]
        
        # Duplicate last image until we have at least 4 images (Cross-Attention requirement)
        while len(processed_images) < 4:
            processed_images.append(processed_images[-1])
            
        # Concatenate along batch dimension -> [N, 3, H, W]
        image_array = np.concatenate(processed_images, axis=0)

        PREPROCESS_LATENCY.observe(time.perf_counter() - t0)

        # ── Triton inference ─────────────────────────────────────────────────
        triton: TritonVPRClient = request.app.state.triton_client
        t1 = time.perf_counter()
        # image_array is [N, 3, H, W] -> Triton returns descriptor [N, D]
        descriptor = triton.get_descriptor(image_array)
        TRITON_INFERENCE_LATENCY.labels(
            model_name="vpr_encoder",
            model_version="1",
        ).observe(time.perf_counter() - t1)

        # Update descriptor dim gauge (idempotent after first call)
        DESCRIPTOR_DIM.set(float(descriptor.shape[1]))

        # ── Milvus retrieval ──────────────────────────────────────────────────
        # Use the first descriptor for retrieval
        query_desc = descriptor[0]
        
        retrieval: RetrievalService = request.app.state.retrieval_service
        t2 = time.perf_counter()
        results: List[RetrievalResult] = retrieval.search(query_desc, top_k=top_k)
        FAISS_RETRIEVAL_LATENCY.labels(top_k=str(top_k)).observe(time.perf_counter() - t2)

        # Update gallery size gauge
        GALLERY_SIZE.set(float(retrieval.gallery_size()))

        REQUESTS_TOTAL.labels(endpoint="/retrieve", status="success").inc()

    except Exception as exc:
        REQUESTS_TOTAL.labels(endpoint="/retrieve", status="error").inc()
        ERRORS_TOTAL.labels(error_type=type(exc).__name__).inc()
        raise

    matches = [
        PlaceMatch(
            rank=r.rank,
            gallery_id=r.gallery_id,
            score=round(r.score, 4),
            distance=round(r.distance, 4),
            image_path=r.image_path,
            metadata=r.metadata,
        )
        for r in results
    ]

    return RetrieveResponse(
        query_descriptor_dim=int(descriptor.shape[1]),
        gallery_size=retrieval.gallery_size(),
        top_k=len(matches),
        matches=matches,
    )
