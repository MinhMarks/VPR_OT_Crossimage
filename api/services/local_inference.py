import os
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LocalVPRClient:
    """
    A drop-in replacement for TritonVPRClient that runs inference locally using PyTorch.
    Useful for local development or when system RAM is too low to run Triton Server.
    """
    def __init__(self, model_path: str = "deployment/triton_models/vpr_encoder/1/model.pt"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.error(f"Local model not found at {self.model_path}")
            return
        
        logger.info(f"Loading local TorchScript model from {self.model_path} onto {self.device}...")
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            
            # Determine if model is FP16 or FP32 based on its first parameter
            first_param = next(self.model.parameters())
            self.dtype = first_param.dtype
            logger.info(f"Model loaded successfully. Dtype: {self.dtype}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def get_descriptor(self, image_array: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of images.
        image_array: numpy array of shape [N, 3, H, W], float32
        Returns: numpy array of shape [N, D], float32
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot perform inference.")

        # Convert numpy array to torch tensor
        tensor = torch.from_numpy(image_array).to(self.device)
        
        # Cast to model's dtype (e.g. FP16 if model was exported with --half)
        tensor = tensor.to(self.dtype)

        with torch.no_grad():
            output = self.model(tensor)
        
        # Always return FP32 numpy array for downstream processing (e.g. Milvus)
        return output.detach().cpu().to(torch.float32).numpy()
