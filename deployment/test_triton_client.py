"""
Test VPR Triton client – sends an image to Triton and prints the descriptor.

Usage:
    # Start Triton first (docker compose up triton), then:
    python deployment/test_triton_client.py --image path/to/query.jpg

Dependencies:
    pip install tritonclient[http] Pillow numpy
"""

import argparse
import sys
import time

import numpy as np
import tritonclient.http as httpclient
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Config – must match export_model.py and config.pbtxt
# ──────────────────────────────────────────────────────────────────────────────

TRITON_URL      = "localhost:8000"
MODEL_NAME      = "vpr_encoder"
MODEL_VERSION   = "1"
IMAGE_SIZE      = (224, 224)            # (H, W)
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]


def preprocess(image_path: str) -> np.ndarray:
    """Load, resize, normalize image → float32 numpy [1, 3, H, W]."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0                # [H, W, 3]

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    arr  = (arr - mean) / std                                      # normalize

    arr  = arr.transpose(2, 0, 1)                                  # [3, H, W]
    arr  = np.expand_dims(arr, axis=0)                             # [1, 3, H, W]
    return arr


def infer(client: httpclient.InferenceServerClient,
          image_array: np.ndarray) -> np.ndarray:
    """Send image array to Triton and return descriptor."""
    infer_input = httpclient.InferInput(
        name="input__0",
        shape=image_array.shape,
        datatype="FP32",
    )
    infer_input.set_data_from_numpy(image_array)

    infer_output = httpclient.InferRequestedOutput("output__0")

    response = client.infer(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        inputs=[infer_input],
        outputs=[infer_output],
    )
    return response.as_numpy("output__0")


def check_server(client: httpclient.InferenceServerClient):
    """Verify server and model are ready."""
    if not client.is_server_ready():
        print("[✗] Triton server is not ready. Is it running?")
        sys.exit(1)
    if not client.is_model_ready(MODEL_NAME, MODEL_VERSION):
        print(f"[✗] Model '{MODEL_NAME}' v{MODEL_VERSION} is not loaded.")
        print("    Check that model.pt exists in triton_models/vpr_encoder/1/")
        sys.exit(1)
    print(f"[✓] Triton ready — model '{MODEL_NAME}' v{MODEL_VERSION} loaded")


def main():
    parser = argparse.ArgumentParser(description="Test VPR Triton inference")
    parser.add_argument("--image",      type=str, required=True,    help="Path to query image")
    parser.add_argument("--triton_url", type=str, default=TRITON_URL, help="Triton HTTP endpoint")
    parser.add_argument("--runs",       type=int, default=5,         help="Number of runs for latency avg")
    args = parser.parse_args()

    # Connect
    print(f"\nConnecting to Triton @ {args.triton_url} ...")
    client = httpclient.InferenceServerClient(url=args.triton_url, verbose=False)
    check_server(client)

    # Preprocess
    print(f"\nPreprocessing: {args.image}")
    image_array = preprocess(args.image)
    print(f"  Input shape : {image_array.shape}  dtype: {image_array.dtype}")

    # Warm up
    print("\nWarm-up run ...")
    _ = infer(client, image_array)

    # Benchmark
    latencies = []
    for i in range(args.runs):
        t0  = time.perf_counter()
        out = infer(client, image_array)
        latencies.append((time.perf_counter() - t0) * 1000)

    descriptor = out[0]   # [D]

    print(f"\n{'='*50}")
    print(f"  Descriptor dim    : {descriptor.shape[0]}")
    print(f"  L2 norm (≈ 1.0)  : {np.linalg.norm(descriptor):.4f}")
    print(f"  Min / Max         : {descriptor.min():.4f} / {descriptor.max():.4f}")
    print(f"  Avg latency       : {sum(latencies)/len(latencies):.2f} ms  ({args.runs} runs)")
    print(f"  Min latency       : {min(latencies):.2f} ms")
    print(f"{'='*50}")
    print("\n[✓] Triton inference test PASSED!")


if __name__ == "__main__":
    main()
