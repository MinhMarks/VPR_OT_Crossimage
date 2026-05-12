"""
Triton Inference Client for VPR encoder.

Sends preprocessed image tensors to the Triton server and returns
the place descriptor as a numpy array.
"""

import logging
from typing import Optional

import numpy as np
import tritonclient.http as httpclient
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class TritonVPRClient:
    """
    HTTP client wrapper for the vpr_encoder model hosted on Triton.

    Args:
        host:        Triton server hostname (default: localhost)
        http_port:   Triton HTTP port (default: 8000)
        model_name:  Triton model name (default: vpr_encoder)
        model_version: Triton model version string (default: "1")
    """

    def __init__(
        self,
        host: str = "localhost",
        http_port: int = 8000,
        model_name: str = "vpr_encoder",
        model_version: str = "1",
        use_ssl: bool = False,
    ):
        self.url           = f"{host}:{http_port}"
        self.model_name    = model_name
        self.model_version = model_version
        self.use_ssl       = use_ssl

        try:
            self.client = httpclient.InferenceServerClient(
                url=self.url,
                verbose=False,
                concurrency=4,     # allow concurrent requests
                ssl=self.use_ssl,
            )
            logger.info(f"TritonVPRClient connected to {self.url} (SSL={self.use_ssl})")
        except Exception as e:
            logger.error(f"Failed to connect to Triton @ {self.url}: {e}")
            self.client = None

    def is_ready(self) -> bool:
        """Check if Triton server and model are ready."""
        try:
            return (
                self.client is not None
                and self.client.is_server_ready()
                and self.client.is_model_ready(self.model_name, self.model_version)
            )
        except Exception:
            return False

    def get_descriptor(self, image_array: np.ndarray) -> np.ndarray:
        """
        Run inference on Triton and return the place descriptor.

        Args:
            image_array: float32 numpy array of shape [1, 3, H, W]

        Returns:
            descriptor: float32 numpy array of shape [D] (L2-normalized)

        Raises:
            HTTPException 503 if Triton is unavailable.
            HTTPException 500 if inference fails.
        """
        if not self.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Triton inference server is not available. Check deployment.",
            )

        try:
            infer_input = httpclient.InferInput(
                name="input__0",
                shape=list(image_array.shape),
                datatype="FP32",
            )
            infer_input.set_data_from_numpy(image_array)

            infer_output = httpclient.InferRequestedOutput("output__0")

            response = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=[infer_input],
                outputs=[infer_output],
            )

            descriptor = response.as_numpy("output__0")[0]   # [D]
            return descriptor.astype(np.float32)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Triton inference error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference failed: {str(e)}",
            )
