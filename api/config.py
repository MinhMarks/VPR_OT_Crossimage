"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Triton
    triton_host: str      = Field("localhost", description="Triton server hostname")
    triton_http_port: int = Field(8000,        description="Triton HTTP port")
    triton_grpc_port: int = Field(8001,        description="Triton gRPC port")
    triton_model_name: str    = Field("vpr_encoder", description="Triton model name")
    triton_model_version: str = Field("1",           description="Triton model version")
    triton_use_ssl: bool = Field(False, description="Use HTTPS/SSL for Triton connection")
    use_local_inference: bool = Field(True, description="Use LocalVPRClient instead of Triton")

    # Milvus
    milvus_host: str = Field(
        "localhost",
        description="Milvus server host",
    )
    milvus_port: int = Field(
        19530,
        description="Milvus server port",
    )
    milvus_collection_name: str = Field(
        "gsv_cities",
        description="Milvus collection name for the gallery",
    )
    milvus_token: str = Field(
        "",
        description="Token for Zilliz Cloud authentication",
    )

    # Auth
    api_key: str = Field(
        "2959537a8c2d7c9e6b4d1a7c5f2e8d0b",
        description="API key for Bearer authentication",
    )

    # App
    log_level: str = Field("info", description="Logging level")
    top_k_default: int = Field(5, description="Default number of results")
    top_k_max: int     = Field(20, description="Maximum allowed top-k")
