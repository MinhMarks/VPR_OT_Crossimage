"""
Build the Milvus gallery index from a dataset directory (e.g., GSV-Cities).

Since the model requires Cross-Attention across 4 images, this script
will automatically duplicate a single gallery image 4 times before passing
it to Triton, then save the 1st descriptor to Milvus.

Usage:
    # Ensure Triton and Milvus are running via docker-compose up
    python api/scripts/build_milvus_index.py \
        --dataset_dir datasets/gsv_cities \
        --milvus_host localhost
"""

import argparse
import glob
import logging
import os
import sys
import time

import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from api.services.preprocess import preprocess_image
from api.services.retrieval import RetrievalService
from api.services.triton_client import TritonVPRClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build Milvus Index for VPR Gallery")
    parser.add_argument(
        "--dataset_dir", type=str, required=True,
        help="Path to directory containing gallery images (e.g., GSV-Cities)"
    )
    parser.add_argument(
        "--triton_url", type=str, default="localhost:8000",
        help="Triton server HTTP URL"
    )
    parser.add_argument(
        "--milvus_host", type=str, default="localhost",
        help="Milvus server host"
    )
    parser.add_argument(
        "--milvus_port", type=int, default=19530,
        help="Milvus server port"
    )
    parser.add_argument(
        "--collection", type=str, default="gsv_cities",
        help="Milvus collection name"
    )
    args = parser.parse_args()

    # 1. Connect to Triton
    logger.info(f"Connecting to Triton @ {args.triton_url}")
    triton = TritonVPRClient(url=args.triton_url)
    if not triton.is_ready():
        logger.error("Triton server is not ready.")
        sys.exit(1)

    # 2. Connect to Milvus
    logger.info(f"Connecting to Milvus @ {args.milvus_host}:{args.milvus_port}")
    retrieval = RetrievalService(
        host=args.milvus_host,
        port=args.milvus_port,
        collection_name=args.collection
    )
    # Ping or just let the first insert create the collection

    # 3. Find images
    valid_exts = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(args.dataset_dir, "**", ext), recursive=True))
    
    if not image_paths:
        logger.error(f"No images found in {args.dataset_dir}")
        sys.exit(1)

    logger.info(f"Found {len(image_paths)} images. Building index...")

    # 4. Extract and Insert
    t0 = time.time()
    success = 0

    for i, path in enumerate(image_paths):
        try:
            with open(path, "rb") as f:
                img_bytes = f.read()

            # Preprocess to [1, 3, H, W]
            arr = preprocess_image(img_bytes)

            # Duplicate to [4, 3, H, W] for Cross-Attention
            arr_4 = np.repeat(arr, 4, axis=0)

            # Infer -> returns [4, D]
            descriptor_batch = triton.get_descriptor(arr_4)

            # Take the first descriptor
            desc = descriptor_batch[0]

            # Insert to Milvus
            meta = {
                "image_path": path,
                "filename": os.path.basename(path),
                # Add any GSV-Cities specific parsing here (like lat/lon from path)
            }
            inserted_id = retrieval.add_descriptor(desc, meta)
            if inserted_id != -1:
                success += 1

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(image_paths)} ...")

        except Exception as e:
            logger.warning(f"Failed to process {path}: {e}")

    duration = time.time() - t0
    logger.info(f"Done! Successfully indexed {success}/{len(image_paths)} images in {duration:.2f}s.")
    logger.info(f"Total collection size in Milvus: {retrieval.gallery_size()}")

if __name__ == "__main__":
    main()
