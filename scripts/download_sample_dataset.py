"""
Tải một dataset thu nhỏ (Sample Gallery) để test VPR hệ thống.

Vì GSV-Cities nguyên bản trên Kaggle là một cục ZIP 25GB không thể tải từng phần, 
script này sẽ tự động tải 20 bức ảnh đường phố từ Unsplash Source (hoặc các ảnh mẫu)
và cấu trúc chúng giống hệt GSV-Cities để bạn làm base truy vấn trên Milvus.

Yêu cầu:
    pip install requests
"""

import os
import time
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Danh sách URL ảnh mẫu tĩnh (có thể thay bằng ảnh khác)
# Sử dụng các địa danh nổi tiếng làm các "Place" khác nhau
SAMPLE_PLACES = {
    "place_001_paris_eiffel": [
        "https://images.unsplash.com/photo-1511739001486-6bfe10ce785f?w=400",
        "https://images.unsplash.com/photo-1543305113-82b47b489d6e?w=400",
        "https://images.unsplash.com/photo-1499856871958-5b9627545d1a?w=400"
    ],
    "place_002_newyork_times_square": [
        "https://images.unsplash.com/photo-1500916434205-0c77489c6211?w=400",
        "https://images.unsplash.com/photo-1522083111817-573a6ce51b8a?w=400",
        "https://images.unsplash.com/photo-1534430480872-3498386e7856?w=400"
    ],
    "place_003_tokyo_shibuya": [
        "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=400",
        "https://images.unsplash.com/photo-1503899036084-c55cdd92da26?w=400",
        "https://images.unsplash.com/photo-1554797589-7241f4b55099?w=400"
    ],
    "place_004_london_bigben": [
        "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=400",
        "https://images.unsplash.com/photo-1529655683823-dc29638abf34?w=400",
        "https://images.unsplash.com/photo-1520939817895-060bdaf4fe1b?w=400"
    ]
}

def download_image(url: str, save_path: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        logger.error(f"Lỗi khi tải {url}: {e}")
        return False

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "../../datasets/sample_gallery")
    os.makedirs(base_dir, exist_ok=True)
    
    logger.info(f"Đang tạo Sample Dataset tại: {os.path.abspath(base_dir)}")
    
    total_downloaded = 0
    for place_name, urls in SAMPLE_PLACES.items():
        place_dir = os.path.join(base_dir, place_name)
        os.makedirs(place_dir, exist_ok=True)
        
        for idx, url in enumerate(urls):
            filename = f"img_{idx:03d}.jpg"
            save_path = os.path.join(place_dir, filename)
            
            if not os.path.exists(save_path):
                logger.info(f"Tải xuống: {place_name}/{filename}...")
                if download_image(url, save_path):
                    total_downloaded += 1
                time.sleep(0.5) # Tránh bị block
            else:
                logger.info(f"Đã tồn tại: {place_name}/{filename}")
                
    logger.info(f"Hoàn tất! Đã tải {total_downloaded} ảnh mới vào {base_dir}")
    logger.info("Bạn có thể chạy lệnh sau để đưa vào Milvus:")
    logger.info("python api/scripts/build_milvus_index.py --dataset_dir datasets/sample_gallery")

if __name__ == "__main__":
    main()
