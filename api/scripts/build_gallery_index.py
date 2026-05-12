"""
Build Gallery FAISS Index from a directory of images.

Usage:
    python api/scripts/build_gallery_index.py \
        --images_dir datasets/gallery/ \
        --ckpt_path  pretrainedWeight/your_model.ckpt \
        --output_dir api/data/ \
        --device     cpu

Output:
    api/data/gallery_index.faiss  – FAISS inner-product index
    api/data/gallery_meta.json    – list of metadata dicts

Supported image extensions: .jpg .jpeg .png .bmp .webp
"""

import argparse
import json
import os
import sys
import glob

import numpy as np
import faiss
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from vpr_model import VPRModel

# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = (322, 322)
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = VPRModel(
        backbone_arch="dinov2_vitb14",
        backbone_config={"num_trainable_blocks": 4, "return_token": True, "norm_layer": True},
        agg_arch="SALAD",
        agg_config={"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256},
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval().to(device)
    return model


def get_transform():
    return T.Compose([
        T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.no_grad()
def embed_image(model, img_path: str, transform, device: torch.device) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    desc = model(tensor).cpu().numpy()[0]   # [D]
    # L2 normalize
    desc = desc / (np.linalg.norm(desc) + 1e-8)
    return desc.astype(np.float32)


def build_index(images_dir: str, ckpt_path: str, output_dir: str, device_str: str):
    device = torch.device(device_str)
    print(f"[→] Loading model from {ckpt_path} ...")
    model = load_model(ckpt_path, device)
    transform = get_transform()

    # Collect image paths
    all_images = []
    for ext in SUPPORTED_EXT:
        all_images.extend(glob.glob(os.path.join(images_dir, "**", f"*{ext}"), recursive=True))
        all_images.extend(glob.glob(os.path.join(images_dir, "**", f"*{ext.upper()}"), recursive=True))
    all_images = sorted(set(all_images))

    if not all_images:
        print(f"[✗] No images found in {images_dir}")
        sys.exit(1)

    print(f"[✓] Found {len(all_images)} images — embedding ...")

    descriptors = []
    metadata    = []
    failed      = 0

    for idx, img_path in enumerate(tqdm(all_images, desc="Embedding")):
        try:
            desc = embed_image(model, img_path, transform, device)
            descriptors.append(desc)
            metadata.append({
                "id": idx - failed,
                "image_path": os.path.relpath(img_path, start=images_dir),
            })
        except Exception as e:
            print(f"  [WARN] Skipping {img_path}: {e}")
            failed += 1

    if not descriptors:
        print("[✗] No descriptors computed — aborting.")
        sys.exit(1)

    # Build FAISS index
    print(f"\n[→] Building FAISS index ...")
    desc_matrix = np.stack(descriptors)    # [N, D]
    faiss.normalize_L2(desc_matrix)        # ensure unit norm
    dim = desc_matrix.shape[1]

    index = faiss.IndexFlatIP(dim)         # Inner Product (cosine after normalization)
    index.add(desc_matrix)
    print(f"[✓] Index: {index.ntotal} vectors, dim={dim}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "gallery_index.faiss")
    meta_path  = os.path.join(output_dir, "gallery_meta.json")

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] Saved FAISS index → {index_path}")
    print(f"[✓] Saved metadata   → {meta_path}")
    if failed:
        print(f"[!] {failed} images failed / skipped")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--images_dir",  required=True, help="Root directory of gallery images")
    p.add_argument("--ckpt_path",   required=True, help="Model checkpoint .ckpt")
    p.add_argument("--output_dir",  default="api/data", help="Output directory for index files")
    p.add_argument("--device",      default="cpu",  choices=["cpu", "cuda"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(args.images_dir, args.ckpt_path, args.output_dir, args.device)
