"""
POST /api/v1/index – Add a new image to the gallery index.

Example:
    curl -X POST http://localhost:8080/api/v1/index \\
         -H "X-API-Key: dev-secret-key" \\
         -F "image=@gallery_img.jpg" \\
         -F "place_name=Ben Thanh Market" \\
         -F "lat=10.7769" \\
         -F "lon=106.7009"
"""

import base64
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import BaseModel, Field

from api.config import Settings
from api.middleware.auth import require_api_key
from api.services.preprocess import preprocess_image
from api.services.triton_client import TritonVPRClient
from api.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)
router = APIRouter()
settings = Settings()


class IndexResponse(BaseModel):
    gallery_id: int  = Field(..., description="Assigned gallery ID")
    gallery_size: int = Field(..., description="Total gallery size after indexing")
    image_url: str   = Field(default="", description="Public URL of the indexed image")
    message: str


async def _upload_to_imgbb(image_bytes: bytes, filename: str, api_key: str) -> str:
    """Upload ảnh lên ImgBB và trả về URL công khai. Trả về "" nếu lỗi."""
    if not api_key:
        return ""
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.imgbb.com/1/upload",
                data={"key": api_key, "image": b64, "name": filename},
            )
        data = resp.json()
        if data.get("success"):
            return data["data"]["url"]
        logger.warning(f"ImgBB upload failed: {data}")
        return ""
    except Exception as e:
        logger.warning(f"ImgBB upload error: {e}")
        return ""


@router.post(
    "/index",
    response_model=IndexResponse,
    summary="Add image to gallery",
    description="Embed a gallery image and add it to the Milvus index. Image is automatically uploaded to ImgBB for public URL.",
)
async def add_to_index(
    request: Request,
    image: UploadFile      = File(..., description="Gallery image to index"),
    place_name: str        = Form(default="", description="Human-readable place name"),
    lat: Optional[float]   = Form(default=None, description="Latitude (GPS)"),
    lon: Optional[float]   = Form(default=None, description="Longitude (GPS)"),
    image_path: str        = Form(default="", description="Original file path / URL (overridden by ImgBB)"),
    _: str                 = Depends(require_api_key),
):
    image_bytes = await image.read()
    logger.info(f"Index request | image={image.filename}")

    # Upload lên ImgBB để lấy URL công khai (cho Frontend hiển thị)
    image_url = await _upload_to_imgbb(
        image_bytes,
        filename=image.filename or "gallery_image.jpg",
        api_key=settings.imgbb_api_key,
    )
    if image_url:
        logger.info(f"ImgBB upload success: {image_url}")
    else:
        logger.info("ImgBB not configured or failed — image_url will be empty")

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

    # Add to Milvus — lưu image_url vào metadata để retrieve trả về sau
    metadata = {
        "place_name": place_name,
        "lat": lat,
        "lon": lon,
        "image_path": image_url or image_path or image.filename,
        "image_url": image_url,  # Public URL để Frontend hiển thị ảnh
    }
    retrieval: RetrievalService = request.app.state.retrieval_service
    gallery_id = retrieval.add_descriptor(descriptor, metadata)

    # Auto-save index to disk
    retrieval.save()

    return IndexResponse(
        gallery_id=gallery_id,
        gallery_size=retrieval.gallery_size(),
        image_url=image_url,
        message=f"Image indexed successfully as gallery_id={gallery_id}",
    )
