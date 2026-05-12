"""
Milvus-based gallery retrieval service.

The gallery is stored in a Milvus collection named `gsv_cities`.
Each entity has:
  - id (int64)
  - embedding (float_vector)
  - image_path (varchar)
  - metadata (json)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    rank: int
    gallery_id: int
    score: float            # cosine similarity (higher = better)
    distance: float         # L2 distance (lower = better)
    image_path: str
    metadata: Dict[str, Any]


class RetrievalService:
    """
    Milvus nearest-neighbor retrieval over the gallery embedding index.
    """

    def __init__(self, host: str = "localhost", port: int = 19530, collection_name: str = "gsv_cities", token: str = ""):
        # Nếu host bắt đầu bằng https (Zilliz), dùng trực tiếp URI đó
        if host.startswith("http"):
            self.uri = host
        else:
            self.uri = f"http://{host}:{port}"
            
        self.collection_name = collection_name
        self.token = token
        self.client = None
        self._connect()

    # ──────────────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────────────

    def _connect(self):
        try:
            # Hỗ trợ cả Local Milvus (no token) và Zilliz Cloud (with token)
            if self.token:
                self.client = MilvusClient(uri=self.uri, token=self.token)
                logger.info(f"Connected to Zilliz Cloud at {self.uri}")
            else:
                self.client = MilvusClient(uri=self.uri)
                logger.info(f"Connected to Milvus at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.client = None

    def is_ready(self) -> bool:
        if not self.client:
            return False
        try:
            return self.client.has_collection(collection_name=self.collection_name)
        except Exception:
            return False

    def gallery_size(self) -> int:
        if not self.is_ready():
            return 0
        try:
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    def search(
        self,
        query_descriptor: np.ndarray,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Find Top-K gallery images most similar to query_descriptor.

        Args:
            query_descriptor: float32 array of shape [D] – L2-normalized
            top_k:            number of results to return

        Returns:
            List of RetrievalResult sorted by similarity (best first)
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Collection '{self.collection_name}' is not ready in Milvus. "
                "Run build_milvus_index.py first."
            )

        # Ensure correct shape and type
        vec = query_descriptor.astype(np.float32)
        if len(vec.shape) == 1:
            vec = vec.reshape(1, -1)

        # Normalize for IP distance
        vec_norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        vec = vec / (vec_norm + 1e-10)

        # Milvus search
        try:
            res = self.client.search(
                collection_name=self.collection_name,
                data=vec.tolist(),
                limit=top_k,
                output_fields=["image_path", "metadata"],
                search_params={"metric_type": "IP"} # Inner Product for Cosine Similarity
            )
        except Exception as e:
            logger.error(f"Milvus search error: {e}")
            return []

        results = []
        if not res or len(res) == 0:
            return results
        
        # res[0] contains top_k matches for the 1st query vector
        for rank, hit in enumerate(res[0]):
            score = float(hit.get("distance", 0.0))
            distance = 2.0 * (1.0 - score) # Convert Cosine to approx L2 distance
            
            entity = hit.get("entity", {})
            metadata = entity.get("metadata", {})
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}

            results.append(
                RetrievalResult(
                    rank=rank + 1,
                    gallery_id=int(hit.get("id")),
                    score=score,
                    distance=distance,
                    image_path=entity.get("image_path", ""),
                    metadata=metadata,
                )
            )

        return results

    def _create_collection_if_not_exists(self, dim: int):
        if self.client.has_collection(collection_name=self.collection_name):
            return

        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="IP",
            index_type="HNSW",
            params={"M": 8, "efConstruction": 64}
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"Created Milvus collection: {self.collection_name} with dim {dim}")

    def add_descriptor(
        self,
        descriptor: np.ndarray,
        metadata: Dict[str, Any],
    ) -> int:
        """
        Add a single descriptor + metadata to the gallery index.
        Returns the inserted primary key.
        """
        if not self.client:
            self._connect()
        
        vec = descriptor.astype(np.float32)
        if len(vec.shape) > 1:
            vec = vec.flatten()
            
        dim = vec.shape[0]
        self._create_collection_if_not_exists(dim)

        vec_norm = np.linalg.norm(vec)
        vec = vec / (vec_norm + 1e-10)

        image_path = metadata.get("image_path", "")

        data = [{
            "embedding": vec.tolist(),
            "image_path": image_path,
            "metadata": metadata
        }]

        res = self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
        # return first id
        if res and "ids" in res:
            return res["ids"][0]
        return -1

    def save(self):
        """No-op for Milvus, data is persisted automatically."""
        pass
