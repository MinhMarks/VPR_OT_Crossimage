"""
POST /api/v1/index – Add a new image to the gallery index.

Example:
    curl -X POST http://localhost:8080/api/v1/index \
         -H "X-API-Key: dev-secret-key" \
         -F "image=@gallery_img.jpg" \
         -F "place_name=Ben Thanh Market" \
         -F "lat=10.7769" \
         -F "lon=106.7009"
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import BaseModel, Field

from api.middleware.auth import require_api_key
from api.services.preprocess import preprocess_image
from api.services.triton_client import TritonVPRClient
from api.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)
router = APIRouter()


class IndexResponse(BaseModel):
    gallery_id: int  = Field(..., description="Assigned gallery ID")
    gallery_size: int = Field(..., description="Total gallery size after indexing")
    message: str


@router.post(
    "/index",
    response_model=IndexResponse,
    summary="Add image to gallery",
    description="Embed a gallery image and add it to the FAISS index.",
)
async def add_to_index(
    request: Request,
    image: UploadFile      = File(..., description="Gallery image to index"),
    place_name: str        = Form(default="", description="Human-readable place name"),
    lat: Optional[float]   = Form(default=None, description="Latitude (GPS)"),
    lon: Optional[float]   = Form(default=None, description="Longitude (GPS)"),
    image_path: str        = Form(default="", description="Original file path / URL"),
    _: str                 = Depends(require_api_key),
):
    image_bytes = await image.read()
    logger.info(f"Index request | image={image.filename}")

    # Preprocess
    image_array = preprocess_image(image_bytes)

    # WORKAROUND: Nhân bản ảnh lên 4 lần để đáp ứng cấu trúc (Batch=4)
    # của kiến trúc Cross-Attention DINOv2-SALAD.
    import numpy as np
    image_array = np.repeat(image_array, 4, axis=0)

    # Embed
    triton: TritonVPRClient = request.app.state.triton_client
    descriptor = triton.get_descriptor(image_array)

    # Do 4 ảnh giống hệt nhau, ta chỉ lấy descriptor của ảnh đầu tiên
    descriptor = descriptor[0]

    # Add to Milvus
    metadata = {
        "place_name": place_name,
        "lat": lat,
        "lon": lon,
        "image_path": image_path or image.filename,
    }
    retrieval: RetrievalService = request.app.state.retrieval_service
    gallery_id = retrieval.add_descriptor(descriptor, metadata)

    # Auto-save index to disk
    retrieval.save()

    return IndexResponse(
        gallery_id=gallery_id,
        gallery_size=retrieval.gallery_size(),
        message=f"Image indexed successfully as gallery_id={gallery_id}",
    )
