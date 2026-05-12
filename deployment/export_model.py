"""
Export VPR Model to TorchScript for Triton Inference Server.

Usage:
    python deployment/export_model.py \
        --ckpt_path pretrainedWeight/your_model.ckpt \
        --output_dir deployment/triton_models/vpr_encoder/1 \
        --device cpu

Notes:
    - DINOv2 + SALAD is exported as a single TorchScript traced model.
    - Input:  image tensor  [B, 3, H, W], float32, normalized ImageNet stats
    - Output: descriptor    [B, D],        float32, L2-normalized
"""

import argparse
import sys
import os

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vpr_model import VPRModel


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

BACKBONE_ARCH = "dinov2_vitb14"
BACKBONE_CONFIG = {
    "num_trainable_blocks": 4,
    "return_token": True,
    "norm_layer": True,
}
AGG_ARCH = "SALAD"
AGG_CONFIG = {
    "num_channels": 768,
    "num_clusters": 64,
    "cluster_dim": 128,
    "token_dim": 256,
}

# Default image size used during inference
DEFAULT_IMAGE_SIZE = (224, 224)


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper: strips training-only logic, keeps eval forward path
# ──────────────────────────────────────────────────────────────────────────────

class VPRInferenceWrapper(torch.nn.Module):
    """
    Thin wrapper around VPRModel for TorchScript export.
    Calls backbone → aggregator(features) to allow cross-image context.
    """

    def __init__(self, vpr_model: VPRModel):
        super().__init__()
        self.backbone = vpr_model.backbone
        self.aggregator = vpr_model.aggregator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] float32 – normalized image tensor
        Returns:
            descriptor: [B, D] float32 – L2-normalized place descriptor
        """
        features = self.backbone(x)          # backbone output (tuple or tensor)
        descriptor = self.aggregator(features)  # cross-image path
        return descriptor


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_vpr_model(ckpt_path: str, device: torch.device) -> VPRModel:
    model = VPRModel(
        backbone_arch=BACKBONE_ARCH,
        backbone_config=BACKBONE_CONFIG,
        agg_arch=AGG_ARCH,
        agg_config=AGG_CONFIG,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)
    print(f"[✓] Loaded checkpoint: {ckpt_path}")
    return model


def make_dummy_input(device: torch.device, image_size=DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Create a dummy batch for tracing (batch=4, required by Cross-Attention)."""
    return torch.randn(4, 3, *image_size, device=device)


def export_torchscript(wrapper: VPRInferenceWrapper,
                        dummy_input: torch.Tensor,
                        output_path: str):
    """Trace and save TorchScript model."""
    # Tracing
    print("[→] Tracing model with TorchScript...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_input)

    # Quick sanity check
    out = traced(dummy_input)
    print(f"[✓] Trace OK — output shape: {out.shape}, dtype: {out.dtype}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced.save(output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[✓] Saved TorchScript model → {output_path}  ({size_mb:.1f} MB)")


def verify_output(wrapper: VPRInferenceWrapper, device: torch.device, half: bool = False):
    """Run a forward pass and print descriptor stats."""
    transform = T.Compose([
        T.Resize(DEFAULT_IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Use a synthetic image if no real one is available
    dummy_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    tensor = transform(dummy_img).unsqueeze(0).repeat(4, 1, 1, 1).to(device)
    if half:
        tensor = tensor.half()

    with torch.no_grad():
        desc = wrapper(tensor)

    norm = torch.linalg.norm(desc, dim=-1)
    print(f"[✓] Descriptor dim  : {desc.shape[1]}")
    print(f"[✓] L2 norm (≈ 1.0) : {norm[0].item():.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export VPR model to TorchScript for Triton",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True,
        help="Path to .ckpt checkpoint file",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="deployment/triton_models/vpr_encoder/1",
        help="Directory to save model.pt (Triton version directory)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device for tracing (use 'cpu' for portability)",
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2,
        default=list(DEFAULT_IMAGE_SIZE),
        metavar=("H", "W"),
        help="Image size used for tracing (must match inference input)",
    )
    parser.add_argument("--half", action="store_true", help="Export model in FP16 (Half Precision)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    image_size = tuple(args.image_size)
    output_path = os.path.join(args.output_dir, "model.pt")

    print("=" * 60)
    print("  VPR Model Export — TorchScript for Triton")
    print("=" * 60)
    print(f"  Checkpoint : {args.ckpt_path}")
    print(f"  Device     : {device}")
    print(f"  Image size : {image_size}")
    print(f"  Output     : {output_path}")
    print("=" * 60)

    # 1. Load model
    vpr_model = load_vpr_model(args.ckpt_path, device)

    # 2. Wrap for inference
    wrapper = VPRInferenceWrapper(vpr_model)
    wrapper.eval()

    # 3. Handle Half Precision
    if args.half:
        print("[!] Converting model and inputs to FP16...")
        wrapper = wrapper.half()

    # 4. Export
    dummy = make_dummy_input(device, image_size)
    if args.half:
        dummy = dummy.half()
        
    export_torchscript(wrapper, dummy, output_path)

    # 5. Verification
    verify_output(wrapper, device, half=args.half)

    print("\n[✓] Export complete!")
    print(f"    Next step: run `docker compose up` inside deployment/")


if __name__ == "__main__":
    main()
