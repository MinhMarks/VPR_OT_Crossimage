"""
Script nhúng (embed) dataset MSLS vào Zilliz Cloud.

Luồng xử lý:
    dataset/ {city} / {location_id} / [4 ảnh]
        → Preprocess 4 ảnh → batch [4, 3, 224, 224]
        → Triton HF Space (Cross-Image Attention)
        → Output [4, 8448] (4 descriptor riêng biệt)
        → Lưu cả 4 vào Zilliz với metadata {city, location_id, lat, lon, filename}

Gọi trực tiếp Triton trên HF + Zilliz, bỏ qua Render để đạt tốc độ tối đa.

Usage:
    pip install tritonclient[http] pymilvus pillow numpy pandas tqdm

    python scripts/embed_msls_to_zilliz.py \\
        --dataset  datasets/msls_cross_embedding_v2 \\
        --metadata datasets/msls_cross_embedding_v2/metadata.csv \\
        --triton-host minhmarks-splace.hf.space \\
        --triton-port 443 \\
        --milvus-host in03-adf6fed6c6b9e2b.serverless.aws-eu-central-1.cloud.zilliz.com \\
        --milvus-token YOUR_ZILLIZ_TOKEN \\
        --collection gsv_cities \\
        --cities paris,tokyo \\          # (tùy chọn) chỉ xử lý 1 số thành phố
        --dry-run                        # (tùy chọn) chỉ in ra, không insert
"""

import argparse
import base64
import io
import logging
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─── ImgBB Upload ────────────────────────────────────────────────────────────
def upload_to_imgbb(image_path: Path, api_key: str) -> str:
    """
    Upload 1 ảnh lên ImgBB và trả về URL công khai.
    Trả về chuỗi rỗng nếu không có api_key hoặc bị lỗi.
    """
    if not api_key:
        return ""
    try:
        b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        resp = httpx.post(
            "https://api.imgbb.com/1/upload",
            data={"key": api_key, "image": b64, "name": image_path.stem},
            timeout=20.0,
        )
        data = resp.json()
        if data.get("success"):
            return data["data"]["url"]
        logger.warning(f"ImgBB failed for {image_path.name}: {data.get('error')}")
    except Exception as e:
        logger.warning(f"ImgBB error {image_path.name}: {e}")
    return ""


# ─── Image Preprocessing (same as api/services/preprocess.py) ────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE    = (224, 224)  # Must match Triton model export


def preprocess_image_file(path: Path) -> np.ndarray:
    """Đọc 1 ảnh từ file và trả về [1, 3, 224, 224] float32."""
    img = Image.open(path).convert("RGB")
    img = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)           # [3, H, W]
    return np.expand_dims(arr, axis=0)    # [1, 3, H, W]


def build_batch(image_paths: list[Path]) -> np.ndarray:
    """Stack N ảnh thành batch [N, 3, H, W]."""
    frames = [preprocess_image_file(p) for p in image_paths]
    return np.concatenate(frames, axis=0).astype(np.float32)  # [N, 3, H, W]


# ─── Triton Client ────────────────────────────────────────────────────────────
def get_triton_client(host: str, port: int, use_ssl: bool):
    import tritonclient.http as httpclient
    url = f"{host}:{port}"
    client = httpclient.InferenceServerClient(url=url, verbose=False, ssl=use_ssl)
    logger.info(f"Triton connected: {url} (SSL={use_ssl})")
    return client


def embed_batch(triton_client, batch: np.ndarray, model_name: str = "vpr_encoder") -> np.ndarray:
    """
    Gửi batch [N, 3, H, W] → Triton → nhận [N, D].
    Cross-Image Attention hoạt động khi N=4 (4 ảnh cùng địa điểm).
    """
    import tritonclient.http as httpclient

    infer_input = httpclient.InferInput("input__0", list(batch.shape), "FP32")
    infer_input.set_data_from_numpy(batch)
    infer_output = httpclient.InferRequestedOutput("output__0")

    response = triton_client.infer(
        model_name=model_name,
        model_version="1",
        inputs=[infer_input],
        outputs=[infer_output],
    )
    descriptors = response.as_numpy("output__0")  # [N, D]
    return descriptors.astype(np.float32)


# ─── Zilliz / Milvus Client ───────────────────────────────────────────────────
def get_milvus_client(host: str, port: int, token: str):
    from pymilvus import MilvusClient, DataType

    # Tự động thêm https:// nếu port là 443
    if host.startswith("http"):
        uri = host
    elif port == 443:
        uri = f"https://{host}"
    else:
        uri = f"http://{host}:{port}"

    client = MilvusClient(uri=uri, token=token)
    logger.info(f"Zilliz connected: {uri}")
    return client


def ensure_collection(client, collection_name: str, dim: int):
    """Tạo collection nếu chưa có, giữ nguyên nếu đã có."""
    from pymilvus import MilvusClient, DataType

    if client.has_collection(collection_name=collection_name):
        stats = client.get_collection_stats(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' đã tồn tại — {stats.get('row_count', 0)} vectors")
        return

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id",         DataType.INT64,        is_primary=True)
    schema.add_field("embedding",  DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("image_path", DataType.VARCHAR,      max_length=1024)
    schema.add_field("metadata",   DataType.JSON)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="IP",
        index_type="HNSW",
        params={"M": 8, "efConstruction": 64},
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info(f"✅ Đã tạo collection '{collection_name}' (dim={dim})")


def insert_descriptors(client, collection_name: str,
                       descriptors: np.ndarray,
                       image_paths: list[Path],
                       metadatas: list[dict]) -> int:
    """Insert N descriptors vào Zilliz, trả về số vector đã insert."""
    data = []
    for i, (vec, path, meta) in enumerate(zip(descriptors, image_paths, metadatas)):
        # L2 normalize trước khi lưu (cho IP search = cosine)
        norm = np.linalg.norm(vec)
        vec_norm = (vec / (norm + 1e-10)).tolist()
        data.append({
            "embedding":  vec_norm,
            "image_path": str(path.name),
            "metadata":   meta,
        })

    res = client.insert(collection_name=collection_name, data=data)
    return len(data)


# ─── Main Pipeline ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Embed MSLS dataset → Zilliz")
    parser.add_argument("--dataset",      default="datasets/msls_cross_embedding_v2")
    parser.add_argument("--metadata",     default="datasets/msls_cross_embedding_v2/metadata.csv")
    parser.add_argument("--triton-host",  default="minhmarks-splace.hf.space")
    parser.add_argument("--triton-port",  type=int, default=443)
    parser.add_argument("--milvus-host",  required=True, help="Zilliz host (không cần https://)")
    parser.add_argument("--milvus-port",  type=int, default=443)
    parser.add_argument("--milvus-token", required=True, help="Zilliz API Token")
    parser.add_argument("--collection",   default="gsv_cities")
    parser.add_argument("--cities",       default="", help="Danh sách thành phố cách nhau bởi dấu phẩy (để trống = tất cả)")
    parser.add_argument("--max-locations",type=int, default=0, help="Giới hạn số location mỗi thành phố (0=tất cả)")
    parser.add_argument("--imgbb-key",    default="", help="ImgBB API key để upload ảnh (lấy miễn phí tại imgbb.com)")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--delay",        type=float, default=0.5, help="Giây chờ giữa mỗi batch Triton")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        logger.error(f"Dataset không tồn tại: {dataset_root}")
        sys.exit(1)

    # Đọc metadata
    logger.info("📂 Đọc metadata.csv...")
    meta_df = pd.read_csv(args.metadata)
    # Tạo dict: key → row (để tra nhanh GPS)
    key_to_meta = {row["key"]: row for _, row in meta_df.iterrows()}

    # Lọc thành phố
    city_filter = [c.strip() for c in args.cities.split(",") if c.strip()]
    city_dirs = sorted([d for d in dataset_root.iterdir()
                        if d.is_dir() and (not city_filter or d.name in city_filter)])

    if not city_dirs:
        logger.error(f"Không tìm thấy thư mục thành phố nào! (filter={city_filter})")
        sys.exit(1)

    logger.info(f"🌍 Tìm thấy {len(city_dirs)} thành phố: {[d.name for d in city_dirs]}")

    # Kết nối services
    if not args.dry_run:
        triton = get_triton_client(args.triton_host, args.triton_port, use_ssl=(args.triton_port == 443))
        milvus = get_milvus_client(args.milvus_host, args.milvus_port, args.milvus_token)
    else:
        triton = milvus = None
        logger.info("🔍 DRY-RUN mode — không gửi request thật")

    dim_confirmed = None
    total_inserted = 0
    total_skipped  = 0

    # ── Vòng lặp chính ────────────────────────────────────────────────────────
    for city_dir in city_dirs:
        city = city_dir.name
        location_dirs = sorted([d for d in city_dir.iterdir() if d.is_dir()])
        if args.max_locations > 0:
            location_dirs = location_dirs[:args.max_locations]

        logger.info(f"\n{'='*60}")
        logger.info(f"🏙️  Thành phố: {city.upper()} — {len(location_dirs)} locations")
        logger.info(f"{'='*60}")

        for loc_dir in location_dirs:
            location_id = loc_dir.name  # VD: "48.8515_2.2234"

            # Lấy danh sách ảnh trong location
            img_files = sorted([f for f in loc_dir.iterdir()
                                 if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])

            if len(img_files) < 4:
                logger.warning(f"  ⚠️  {location_id}: chỉ có {len(img_files)} ảnh (cần ≥4) → bỏ qua")
                total_skipped += 1
                continue

            # Lấy GPS từ metadata (dùng ảnh đầu tiên trong location)
            lat, lon = None, None
            try:
                lat_s, lon_s = location_id.split("_")
                lat, lon = float(lat_s), float(lon_s)
            except Exception:
                pass

            if args.dry_run:
                logger.info(f"  📍 {location_id} | {len(img_files)} ảnh | GPS=({lat},{lon})")
                total_inserted += len(img_files)
                continue

            # ── Preprocess → batch [4, 3, 224, 224] ──────────────────────────
            try:
                batch = build_batch(img_files[:4])  # Luôn dùng đúng 4 ảnh
            except Exception as e:
                logger.warning(f"  ❌ Preprocess lỗi {location_id}: {e}")
                total_skipped += 1
                continue

            # ── Gửi sang Triton (Cross-Image Attention với batch=4) ───────────
            try:
                descriptors = embed_batch(triton, batch)  # → [4, D]
                if dim_confirmed is None:
                    dim_confirmed = descriptors.shape[1]
                    ensure_collection(milvus, args.collection, dim_confirmed)
                    logger.info(f"✅ Descriptor dim: {dim_confirmed}")
            except Exception as e:
                logger.warning(f"  ❌ Triton lỗi {location_id}: {e}")
                total_skipped += 1
                time.sleep(2)  # Chờ server hồi phục
                continue

            # ── Upload ảnh lên ImgBB (nếu có key) + Build metadata ──────────
            metadatas = []
            uploaded_count = 0
            for img_path in img_files[:4]:
                img_key = img_path.stem
                row = key_to_meta.get(img_key, {})

                # Upload lên ImgBB để lấy URL công khai cho Frontend
                image_url = upload_to_imgbb(img_path, args.imgbb_key)
                if image_url:
                    uploaded_count += 1

                metadatas.append({
                    "city":        city,
                    "location_id": location_id,
                    "place_name":  f"{city.capitalize()} ({location_id})",
                    "lat":         lat,
                    "lon":         lon,
                    "filename":    img_path.name,
                    "image_url":   image_url,   # URL public để Frontend hiển thi
                    "image_path":  img_path.name,
                    "captured_at": str(row.get("captured_at", "")),
                })

            if args.imgbb_key:
                logger.info(f"    ImgBB: {uploaded_count}/4 anh duoc upload")

            # ── Insert vào Zilliz ─────────────────────────────────────────────
            try:
                n = insert_descriptors(milvus, args.collection,
                                       descriptors, img_files[:4], metadatas)
                total_inserted += n
                logger.info(f"  ✅ {location_id} | +{n} vectors | total={total_inserted}")
            except Exception as e:
                logger.warning(f"  ❌ Zilliz insert lỗi {location_id}: {e}")
                total_skipped += 1

            time.sleep(args.delay)  # Tránh rate-limit HF Space

        # Flush sau mỗi thành phố
        if not args.dry_run and milvus and dim_confirmed:
            try:
                milvus.flush(collection_name=args.collection)
                stats = milvus.get_collection_stats(collection_name=args.collection)
                logger.info(f"💾 Flush xong — Gallery size: {stats.get('row_count', '?')}")
            except Exception as e:
                logger.warning(f"Flush warning: {e}")

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Hoàn thành!")
    logger.info(f"   Inserted : {total_inserted} vectors")
    logger.info(f"   Skipped  : {total_skipped} locations")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
