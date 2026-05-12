"""
VPR FastAPI Gateway – main application entry point.

Endpoints:
    POST /api/v1/retrieve   – Query image → Top-K matching places
    POST /api/v1/index      – Add new images to gallery index
    GET  /api/v1/health     – Service health check
    GET  /metrics           – Prometheus metrics scrape endpoint
    GET  /docs              – Swagger UI (auto-generated)
    GET  /redoc             – ReDoc UI

Run locally:
    uvicorn api.main:app --reload --port 8080

Environment variables:
    TRITON_HOST           – Triton server hostname (default: localhost)
    TRITON_HTTP_PORT      – Triton HTTP port (default: 8000)
    GALLERY_INDEX_PATH    – Path to FAISS index file
    GALLERY_META_PATH     – Path to gallery metadata JSON
    API_KEY               – Bearer token for authentication
    LOG_LEVEL             – Logging level (default: info)
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.routers import retrieve, index, health
from api.services.retrieval import RetrievalService
from api.services.triton_client import TritonVPRClient
from api.services.local_inference import LocalVPRClient
from api.services.metrics import GALLERY_SIZE, TRITON_READY
from api.config import Settings

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Rate limiter
# ──────────────────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan – startup / shutdown
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared services on startup, clean up on shutdown."""
    logger.info("Starting VPR API ...")

    # 1. Inference Client (Local or Triton)
    if settings.use_local_inference:
        logger.info("Using LocalVPRClient (Local Inference) ...")
        triton_client = LocalVPRClient(
            model_path="deployment/triton_models/vpr_encoder/1/model.pt"
        )
    else:
        logger.info(f"Connecting to Triton Cloud at {settings.triton_host}:{settings.triton_http_port} (SSL={settings.triton_use_ssl}) ...")
        triton_client = TritonVPRClient(
            host=settings.triton_host,
            http_port=settings.triton_http_port,
            model_name=settings.triton_model_name,
            model_version=settings.triton_model_version,
            use_ssl=settings.triton_use_ssl
        )
    app.state.triton_client = triton_client

    # Initialize Milvus retrieval service
    retrieval_service = RetrievalService(
        host=settings.milvus_host,
        port=settings.milvus_port,
        collection_name=settings.milvus_collection_name,
        token=settings.milvus_token,
    )
    app.state.retrieval_service = retrieval_service

    # Seed Prometheus gauges with initial values
    TRITON_READY.set(1.0 if triton_client.is_ready() else 0.0)
    GALLERY_SIZE.set(float(retrieval_service.gallery_size()))
    logger.info(
        f"Prometheus gauges seeded — "
        f"triton_ready={triton_client.is_ready()} "
        f"gallery_size={retrieval_service.gallery_size()}"
    )

    logger.info("VPR API ready ✓")
    yield

    # Shutdown
    logger.info("Shutting down VPR API ...")


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VPR API",
    description=(
        "Visual Place Recognition API — upload a query image and receive "
        "the Top-K matching places with their geo-coordinates."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Rate limit handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Request timing middleware
# ──────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────────────────────────────────────

app.include_router(health.router,   prefix="/api/v1", tags=["Health"])
app.include_router(retrieve.router, prefix="/api/v1", tags=["Retrieval"])
app.include_router(index.router,    prefix="/api/v1", tags=["Indexing"])

# ──────────────────────────────────────────────────────────────────────────────
# Prometheus – expose /metrics endpoint
# instrumentator auto-collects: request count, latency, in-flight requests
# Custom VPR metrics are in api/services/metrics.py
# ──────────────────────────────────────────────────────────────────────────────

Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics", "/docs", "/redoc", "/openapi.json"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# ──────────────────────────────────────────────────────────────────────────────
# Root
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "VPR API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
