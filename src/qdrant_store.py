"""
Qdrant Vector Store for Legal RAG

Features:
- Dense vectors (DEk21 embedding)
- Sparse vectors (BM25-style)
- Hybrid search with RRF (Reciprocal Rank Fusion)
- Cross-Encoder Reranking
- Payload indexing for legal metadata
"""

import os
import logging
import re
import time
from typing import Any, List, Dict, Optional, Tuple, Sequence
from dataclasses import dataclass
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector,
    Filter, FieldCondition, MatchValue, MatchAny,
)

logger = logging.getLogger(__name__)

# Configuration
QDRANT_PATH = os.getenv("QDRANT_PATH", "./data/qdrant_db")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_USE_SERVER = os.getenv("QDRANT_USE_SERVER", "true").lower() == "true"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "legal_documents")
DENSE_DIM = int(os.getenv("QDRANT_DENSE_DIM", "768"))
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "120"))

# Qdrant Cloud settings
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_USE_CLOUD = os.getenv("QDRANT_USE_CLOUD", "false").lower() == "true"

# Reranker settings
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "20"))  # Fetch more for reranking

# Hybrid search weights
DEFAULT_DENSE_WEIGHT = 0.6
DEFAULT_SPARSE_WEIGHT = 0.4


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _runtime_qdrant_config() -> Dict[str, Any]:
    return {
        "path": os.getenv("QDRANT_PATH", QDRANT_PATH),
        "host": os.getenv("QDRANT_HOST", QDRANT_HOST),
        "port": int(os.getenv("QDRANT_PORT", str(QDRANT_PORT))),
        "use_server": _env_bool("QDRANT_USE_SERVER", QDRANT_USE_SERVER),
        "collection_name": os.getenv("QDRANT_COLLECTION", COLLECTION_NAME),
        "dense_dim": int(os.getenv("QDRANT_DENSE_DIM", str(DENSE_DIM))),
        "timeout": int(os.getenv("QDRANT_TIMEOUT", str(QDRANT_TIMEOUT))),
        "use_cloud": _env_bool("QDRANT_USE_CLOUD", QDRANT_USE_CLOUD),
        "cloud_url": os.getenv("QDRANT_CLOUD_URL", QDRANT_CLOUD_URL),
        "api_key": os.getenv("QDRANT_API_KEY", QDRANT_API_KEY),
        "payload_profile": os.getenv("QDRANT_PAYLOAD_PROFILE", "full"),
    }


def _qdrant_runtime_signature() -> Tuple[Any, ...]:
    cfg = _runtime_qdrant_config()
    return (
        cfg["path"],
        cfg["host"],
        int(cfg["port"]),
        bool(cfg["use_server"]),
        bool(cfg["use_cloud"]),
        str(cfg["cloud_url"]),
        str(cfg["collection_name"]),
        int(cfg["dense_dim"]),
        int(cfg["timeout"]),
        str(cfg["payload_profile"]).strip().lower(),
    )

LEAN_QDRANT_PAYLOAD_FIELDS: Tuple[str, ...] = (
    "chunk_id",
    "content",
    "doc_id",
    "so_ky_hieu",
    "van_ban_id",
    "ten_van_ban",
    "loai_van_ban",
    "doc_type",
    "doc_type_rank",
    "co_quan_ban_hanh",
    "co_quan",
    "chuc_danh",
    "nguoi_ky",
    "law_family",
    "ngay_ban_hanh",
    "ngay_co_hieu_luc",
    "ngay_het_hieu_luc",
    "is_active",
    "status_label",
    "chuong",
    "ten_chuong",
    "dieu_so",
    "tieu_de_dieu",
    "khoan_so",
    "diem_so",
    "legal_path",
    "chunk_type",
    "segment_level",
    "fallback_reason",
    "provision_id",
    "parent_provision_id",
    "citation_base",
    "citation_full",
    "chunk_index",
    "total_chunks",
    "char_start",
    "char_end",
    "token_len_est",
    "token_len_model",
    "overflow_index",
    "overflow_total",
    "window_char_start",
    "window_char_end",
    "clause_spans",
    "encoder_profile",
)


def _payload_value_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def build_qdrant_payload(chunk_or_payload: Any, profile: str = "full") -> Dict[str, Any]:
    if hasattr(chunk_or_payload, "to_dict"):
        payload = dict(chunk_or_payload.to_dict())
    else:
        payload = dict(chunk_or_payload or {})

    normalized_profile = str(profile or "full").strip().lower()
    if normalized_profile in {"full", "raw"}:
        return payload
    if normalized_profile not in {"lean", "serving"}:
        raise ValueError(f"Unsupported Qdrant payload profile: {profile}")

    doc_id = str(payload.get("doc_id") or payload.get("van_ban_id") or "").strip()
    if doc_id:
        payload["doc_id"] = doc_id
        if not str(payload.get("van_ban_id") or "").strip():
            payload["van_ban_id"] = doc_id
        if not str(payload.get("so_ky_hieu") or "").strip():
            payload["so_ky_hieu"] = doc_id

    legal_path = str(payload.get("legal_path") or payload.get("path") or "").strip()
    if legal_path:
        payload["legal_path"] = legal_path

    if not str(payload.get("co_quan") or "").strip() and str(payload.get("co_quan_ban_hanh") or "").strip():
        payload["co_quan"] = payload.get("co_quan_ban_hanh")
    if not str(payload.get("co_quan_ban_hanh") or "").strip() and str(payload.get("co_quan") or "").strip():
        payload["co_quan_ban_hanh"] = payload.get("co_quan")

    lean_payload: Dict[str, Any] = {}
    for field_name in LEAN_QDRANT_PAYLOAD_FIELDS:
        value = payload.get(field_name)
        if not _payload_value_is_present(value):
            continue
        if isinstance(value, str):
            value = value.strip()
        lean_payload[field_name] = value

    return lean_payload


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(model_name or "").lower()).strip("_")
    return slug[:80] or "unknown_model"


def collection_name_for_model(base_name: str, model_name: str, dense_dim: int) -> str:
    return f"{base_name}__{model_slug(model_name)}__d{int(dense_dim)}"


@dataclass
class SearchResult:
    """Search result with metadata"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict
    
    @property
    def van_ban_id(self) -> str:
        return self.metadata.get("van_ban_id", "")
    
    @property
    def dieu_so(self) -> str:
        return self.metadata.get("dieu_so", "")
    
    @property
    def ten_van_ban(self) -> str:
        return self.metadata.get("ten_van_ban", "")

    @property
    def doc_id(self) -> str:
        return self.metadata.get("doc_id", self.metadata.get("van_ban_id", ""))


class BM25Encoder:
    """Simple BM25-style sparse encoder for Vietnamese legal text"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab = {}
        self.idf = {}
        self.doc_freq = {}
        self.avg_doc_len = 0
        self.doc_count = 0
        self.total_doc_len = 0
        
        # Try to load pyvi for Vietnamese tokenization
        try:
            from pyvi import ViTokenizer
            self.tokenizer = ViTokenizer.tokenize
            logger.info("[BM25] Using pyvi Vietnamese tokenizer")
        except ImportError:
            self.tokenizer = lambda x: x
            logger.warning("[BM25] pyvi not found, using simple tokenizer")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Vietnamese text"""
        text = self.tokenizer(text.lower())
        # Split by spaces and underscores (pyvi joins words with _)
        tokens = []
        for word in text.split():
            # Keep compound words from pyvi
            tokens.append(word.replace("_", " "))
        return tokens
    
    def _update_stats(self, documents: List[str]):
        for doc in documents:
            tokens = self.tokenize(doc)
            self.doc_count += 1
            self.total_doc_len += len(tokens)

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def _recompute_idf(self):
        import math

        self.avg_doc_len = self.total_doc_len / self.doc_count if self.doc_count > 0 else 1
        self.idf = {}
        for token, df in self.doc_freq.items():
            self.idf[token] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

    def fit(self, documents: List[str]):
        """Build vocabulary and IDF from documents (full reset)."""
        self.vocab = {}
        self.idf = {}
        self.doc_freq = {}
        self.avg_doc_len = 0
        self.doc_count = 0
        self.total_doc_len = 0
        self.partial_fit(documents)

    def partial_fit(self, documents: List[str]):
        """Incrementally update vocabulary and IDF."""
        if not documents:
            return
        self._update_stats(documents)
        self._recompute_idf()
        logger.info(f"[BM25] Vocabulary size: {len(self.vocab)} | docs={self.doc_count}")
    
    def encode(self, text: str) -> Tuple[List[int], List[float]]:
        """Encode text to sparse vector (indices, values)"""
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        
        # Count term frequency
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        
        indices = []
        values = []
        
        for token, freq in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                idf = self.idf.get(token, 0)
                
                # BM25 score
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score = idf * (numerator / denominator)
                
                indices.append(idx)
                values.append(float(score))
        
        return indices, values
    
    def save(self, path: str):
        """Save encoder to file"""
        import json
        data = {
            "vocab": self.vocab,
            "idf": self.idf,
            "doc_freq": self.doc_freq,
            "avg_doc_len": self.avg_doc_len,
            "doc_count": self.doc_count,
            "total_doc_len": self.total_doc_len,
            "k1": self.k1,
            "b": self.b
        }
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(f"[BM25] Saved encoder to {path}")
    
    def load(self, path: str):
        """Load encoder from file"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.doc_freq = data.get("doc_freq", {})
        self.avg_doc_len = data["avg_doc_len"]
        self.doc_count = data["doc_count"]
        self.total_doc_len = data.get("total_doc_len", int(self.avg_doc_len * self.doc_count))
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)
        logger.info(f"[BM25] Loaded encoder with {len(self.vocab)} vocab")


class QdrantLegalStore:
    """
    Qdrant Vector Store for Vietnamese Legal Documents
    
    Features:
    - Hybrid search (dense + sparse)
    - Metadata filtering
    - Payload indexing
    """
    
    def __init__(
        self,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
        dense_dim: Optional[int] = None,
        force_unlock: bool = True,
        use_server: Optional[bool] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[int] = None,
        use_cloud: Optional[bool] = None,
        cloud_url: Optional[str] = None,
        api_key: Optional[str] = None,
        bm25_path: Optional[str] = None,
        payload_profile: Optional[str] = None,
    ):
        runtime_cfg = _runtime_qdrant_config()
        self.path = str(path or runtime_cfg["path"])
        self.collection_name = str(collection_name or runtime_cfg["collection_name"])
        self.dense_dim = int(dense_dim if dense_dim is not None else runtime_cfg["dense_dim"])
        self.use_server = bool(runtime_cfg["use_server"] if use_server is None else use_server)
        self.use_cloud = bool(runtime_cfg["use_cloud"] if use_cloud is None else use_cloud)
        self.timeout = int(runtime_cfg["timeout"] if timeout is None else timeout)
        self.cloud_url = str(runtime_cfg["cloud_url"] if cloud_url is None else cloud_url)
        self.api_key = str(runtime_cfg["api_key"] if api_key is None else api_key)
        self.payload_profile = str(payload_profile or runtime_cfg["payload_profile"]).strip().lower()
        host = str(host or runtime_cfg["host"])
        port = int(runtime_cfg["port"] if port is None else port)
        self.runtime_signature = _qdrant_runtime_signature()

        # Initialize Qdrant client
        if self.use_cloud and self.cloud_url:
            # Qdrant Cloud mode
            self.client = QdrantClient(url=self.cloud_url, api_key=self.api_key, timeout=self.timeout)
            logger.info(f"[QDRANT] Connected to Cloud: {self.cloud_url[:50]}...")
        elif self.use_server:
            # Server mode - connect to Qdrant server (Docker)
            self.client = QdrantClient(host=host, port=port, timeout=self.timeout)
            logger.info(f"[QDRANT] Connected to server at {host}:{port}")
        else:
            # Local mode - SQLite storage
            os.makedirs(self.path, exist_ok=True)
            if force_unlock:
                self._remove_lock_file()
            self.client = QdrantClient(path=self.path, timeout=self.timeout)
            logger.info(f"[QDRANT] Initialized local storage at {self.path}")

        # BM25 encoder for sparse vectors
        self.bm25_encoder = BM25Encoder()
        self.bm25_path = bm25_path or os.path.join(self.path, "bm25_encoder.json")

        # Load BM25 if exists
        if os.path.exists(self.bm25_path):
            self.bm25_encoder.load(self.bm25_path)

        logger.info(
            "[QDRANT] Ready (cloud=%s, server=%s, payload_profile=%s)",
            self.use_cloud,
            self.use_server,
            self.payload_profile,
        )

    def set_bm25_path(self, path: str, load_if_exists: bool = True):
        """Set BM25 artifact path explicitly (used for collection-scoped sparse consistency)."""
        self.bm25_path = str(path)
        if load_if_exists and os.path.exists(self.bm25_path):
            self.bm25_encoder.load(self.bm25_path)
    
    def _remove_lock_file(self):
        """Remove Qdrant lock file to allow reconnection"""
        import glob
        lock_patterns = [
            os.path.join(self.path, ".lock"),
            os.path.join(self.path, "*.lock"),
            os.path.join(self.path, "storage.lock"),
        ]
        
        for pattern in lock_patterns:
            for lock_file in glob.glob(pattern):
                try:
                    os.remove(lock_file)
                    logger.info(f"[QDRANT] Removed lock file: {lock_file}")
                except Exception as e:
                    logger.warning(f"[QDRANT] Could not remove lock: {e}")
    
    def create_collection(self, force_recreate: bool = False):
        """Create collection with dense + sparse vectors"""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if exists:
            if force_recreate:
                self.client.delete_collection(self.collection_name)
                logger.info(f"[QDRANT] Deleted existing collection: {self.collection_name}")
            else:
                logger.info(f"[QDRANT] Collection exists: {self.collection_name}")
                return
        
        # Create collection with hybrid vectors
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.dense_dim,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            },
            on_disk_payload=True,
        )
        
        # Create payload indexes for filtering
        self._create_payload_indexes()
        
        logger.info(f"[QDRANT] Created collection: {self.collection_name}")
    
    def _create_payload_indexes(self):
        """Create indexes for efficient filtering"""
        indexed_fields = [
            ("doc_id", "keyword"),
            ("so_ky_hieu", "keyword"),
            ("van_ban_id", "keyword"),
            ("loai_van_ban", "keyword"),
            ("doc_type", "keyword"),
            ("doc_type_rank", "integer"),
            ("chunk_type", "keyword"),
            ("segment_level", "keyword"),
            ("status_label", "keyword"),
            ("law_family", "keyword"),
            ("co_quan", "keyword"),
            ("co_quan_ban_hanh", "keyword"),
            ("nguoi_ky", "keyword"),
            ("dieu_so", "keyword"),
            ("khoan_so", "keyword"),
            ("diem_so", "keyword"),
            ("chuong", "keyword"),
            ("legal_path", "keyword"),
            ("is_active", "bool"),
            ("ngay_ban_hanh", "keyword"),
            ("ngay_co_hieu_luc", "keyword"),
            ("ngay_het_hieu_luc", "keyword"),
            ("token_len_est", "integer"),
            ("token_len_model", "integer"),
            ("overflow_index", "integer"),
            ("overflow_total", "integer"),
            ("encoder_profile", "keyword"),
        ]
        
        for field_name, field_type in indexed_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.info(f"[QDRANT] Created index for: {field_name}")
            except Exception as e:
                logger.debug(f"[QDRANT] Index may exist: {field_name}")

    def _build_filter(self, filter_conditions: Optional[Dict[str, object]]) -> Optional[Filter]:
        if not filter_conditions:
            return None

        conditions = []
        for field, value in filter_conditions.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                values = [v for v in value if v is not None]
                if not values:
                    continue
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchAny(any=list(values)),
                    )
                )
            else:
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value),
                    )
                )

        if not conditions:
            return None
        return Filter(must=conditions)
    
    def add_documents(
        self,
        chunks: List,  # List[LegalChunk]
        embeddings: List[List[float]],
        batch_size: int = 100,
        fit_bm25: bool = True,
        upsert_batch_ladder: Optional[List[int]] = None,
        max_retries_per_batch: int = 3,
        max_consecutive_failures: int = 3,
    ):
        """Add documents with dense and sparse vectors"""
        if not chunks:
            logger.warning("[QDRANT] No chunks to add")
            return {
                "indexed_chunks": 0,
                "retries": 0,
                "timeouts": 0,
                "errors": 0,
                "consecutive_failures": 0,
                "upsert_calls": 0,
            }
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}")

        if fit_bm25:
            # Fit BM25 encoder on the provided document set.
            contents = []
            for chunk in chunks:
                content = getattr(chunk, "content", None) or getattr(chunk, "text", "")
                contents.append(content)
            self.bm25_encoder.fit(contents)
            self.bm25_encoder.save(self.bm25_path)
        elif not self.bm25_encoder.vocab:
            raise RuntimeError("BM25 encoder is empty; fit it before calling add_documents(..., fit_bm25=False)")
        
        # Build points once, then upsert with optional adaptive batch strategy.
        points: List[PointStruct] = []
        for chunk, dense_vec in zip(chunks, embeddings):
            content = getattr(chunk, "content", None) or getattr(chunk, "text", "")
            sparse_indices, sparse_values = self.bm25_encoder.encode(content)
            point_id = int(hashlib.md5(chunk.chunk_id.encode()).hexdigest()[:16], 16)
            payload = build_qdrant_payload(chunk, profile=self.payload_profile)
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        )
                    },
                    payload=payload,
                )
            )

        metrics = self._upsert_points_adaptive(
            points=points,
            default_batch_size=batch_size,
            upsert_batch_ladder=upsert_batch_ladder,
            max_retries_per_batch=max_retries_per_batch,
            max_consecutive_failures=max_consecutive_failures,
        )
        return metrics

    def _upsert_points_adaptive(
        self,
        points: List[PointStruct],
        default_batch_size: int = 100,
        upsert_batch_ladder: Optional[List[int]] = None,
        max_retries_per_batch: int = 3,
        max_consecutive_failures: int = 3,
    ) -> Dict[str, int]:
        if not points:
            return {
                "indexed_chunks": 0,
                "retries": 0,
                "timeouts": 0,
                "errors": 0,
                "consecutive_failures": 0,
            }

        ladder = [int(x) for x in (upsert_batch_ladder or []) if int(x) > 0]
        if not ladder:
            ladder = [int(default_batch_size)]
        ladder = sorted(list(dict.fromkeys(ladder)), reverse=True)
        if int(default_batch_size) > 0 and int(default_batch_size) not in ladder:
            ladder.append(int(default_batch_size))
            ladder = sorted(list(dict.fromkeys(ladder)), reverse=True)

        current_level = 0
        cursor = 0
        total = len(points)
        retries = 0
        timeouts = 0
        errors = 0
        consecutive_failures = 0
        upsert_calls = 0

        while cursor < total:
            batch_size = ladder[current_level]
            batch_points = points[cursor:cursor + batch_size]
            attempt = 0
            success = False
            last_err: Optional[Exception] = None

            while attempt <= max_retries_per_batch:
                try:
                    upsert_calls += 1
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch_points,
                        wait=True,
                    )
                    success = True
                    break
                except Exception as exc:
                    last_err = exc
                    retries += 1
                    err_text = str(exc).lower()
                    if "timeout" in err_text:
                        timeouts += 1
                    else:
                        errors += 1
                    attempt += 1
                    if attempt <= max_retries_per_batch:
                        time.sleep(min(2.0 ** attempt, 8.0))

            if success:
                cursor += len(batch_points)
                consecutive_failures = 0
                logger.info(
                    "[QDRANT] Indexed %s/%s documents (upsert_batch=%s)",
                    min(cursor, total),
                    total,
                    batch_size,
                )
                continue

            consecutive_failures += 1
            if current_level + 1 < len(ladder):
                current_level += 1
                logger.warning(
                    "[QDRANT] Upsert failed at batch=%s, degrading to batch=%s. err=%s",
                    batch_size,
                    ladder[current_level],
                    last_err,
                )
                continue

            logger.error(
                "[QDRANT] Upsert failed at smallest batch=%s (consecutive_failures=%s): %s",
                batch_size,
                consecutive_failures,
                last_err,
            )
            if consecutive_failures > max_consecutive_failures:
                raise RuntimeError(
                    f"Upsert failed > {max_consecutive_failures} consecutive times at batch={batch_size}: {last_err}"
                )

        return {
            "indexed_chunks": total,
            "retries": retries,
            "timeouts": timeouts,
            "errors": errors,
            "consecutive_failures": consecutive_failures,
            "upsert_calls": upsert_calls,
        }

    def point_count(self, collection_name: Optional[str] = None) -> int:
        target = collection_name or self.collection_name
        info = self.client.get_collection(target)
        return int(getattr(info, "points_count", 0) or 0)

    def point_count_via_scroll(self, collection_name: Optional[str] = None) -> int:
        target = collection_name or self.collection_name
        count_result = self.client.count(
            collection_name=target,
            exact=True,
        )
        return int(getattr(count_result, "count", 0) or 0)

    def set_alias(self, alias_name: str, target_collection: str):
        self.client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=target_collection,
                        alias_name=alias_name,
                    )
                )
            ]
        )

    def delete_alias(self, alias_name: str):
        self.client.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=alias_name)
                )
            ]
        )

    def swap_alias(self, alias_name: str, new_collection: str):
        old_collection = None
        aliases = self.client.get_aliases().aliases
        for alias in aliases:
            if alias.alias_name == alias_name:
                old_collection = alias.collection_name
                break

        ops = []
        if old_collection:
            ops.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=alias_name)
                )
            )
        ops.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=new_collection,
                    alias_name=alias_name,
                )
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=ops)

    def resolve_alias(self, alias_name: str) -> Optional[str]:
        aliases = self.client.get_aliases().aliases
        for alias in aliases:
            if alias.alias_name == alias_name:
                return alias.collection_name
        return None
    
    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        filter_conditions: Optional[Dict] = None,
        collection_name: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Hybrid search combining dense and sparse vectors
        
        Args:
            query: Query text (for sparse search)
            query_embedding: Dense embedding of query
            top_k: Number of results
            dense_weight: Weight for dense vector (α)
            sparse_weight: Weight for sparse vector (β)
            filter_conditions: Optional metadata filters
            collection_name: Optional custom collection name (default: self.collection_name)
        """
        # Build sparse query vector
        sparse_indices, sparse_values = self.bm25_encoder.encode(query)
        if len(query_embedding) != self.dense_dim:
            raise ValueError(
                f"Dense embedding dimension mismatch: expected {self.dense_dim}, got {len(query_embedding)} "
                f"for collection={self.collection_name}"
            )
        dense_weight = max(float(dense_weight), 0.0)
        sparse_weight = max(float(sparse_weight), 0.0)
        if dense_weight == 0.0 and sparse_weight == 0.0:
            dense_weight = DEFAULT_DENSE_WEIGHT
            sparse_weight = DEFAULT_SPARSE_WEIGHT
        
        # Build filter if provided
        qdrant_filter = self._build_filter(filter_conditions)
        
        # Use custom collection name if provided
        target_collection = collection_name if collection_name else self.collection_name

        if hasattr(self.client, "query_points"):
            # Perform weighted retrieval strategy:
            # - both dense+sparse > 0: hybrid RRF
            # - dense only: semantic
            # - sparse only: BM25-like sparse
            if sparse_indices and dense_weight > 0 and sparse_weight > 0:
                results = self.client.query_points(
                    collection_name=target_collection,
                    prefetch=[
                        models.Prefetch(
                            query=query_embedding,
                            using="dense",
                            limit=top_k * 2  # Fetch more for fusion
                        ),
                        models.Prefetch(
                            query=SparseVector(
                                indices=sparse_indices,
                                values=sparse_values
                            ),
                            using="sparse",
                            limit=top_k * 2
                        )
                    ],
                    query=models.FusionQuery(
                        fusion=models.Fusion.RRF  # Reciprocal Rank Fusion
                    ),
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True
                )
            elif dense_weight >= sparse_weight or not sparse_indices:
                results = self.client.query_points(
                    collection_name=target_collection,
                    query=query_embedding,
                    using="dense",
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
            else:
                results = self.client.query_points(
                    collection_name=target_collection,
                    query=SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="sparse",
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )

            # Convert to SearchResult
            search_results = []
            for point in results.points:
                search_results.append(SearchResult(
                    chunk_id=point.payload.get("chunk_id", ""),
                    content=point.payload.get("content", ""),
                    score=point.score,
                    metadata=point.payload
                ))
            return search_results

        return self._legacy_hybrid_search(
            query_embedding=query_embedding,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            qdrant_filter=qdrant_filter,
            target_collection=target_collection,
        )

    def _legacy_hybrid_search(
        self,
        query_embedding: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
        qdrant_filter: Optional[Filter],
        target_collection: str,
    ) -> List[SearchResult]:
        """
        Compatibility path for qdrant-client versions without query_points().
        """
        dense_hits = []
        sparse_hits = []
        fetch_k = max(top_k * 2, top_k)

        if dense_weight > 0:
            dense_hits = self.client.search(
                collection_name=target_collection,
                query_vector=("dense", query_embedding),
                limit=fetch_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

        if sparse_weight > 0 and sparse_indices:
            sparse_query = models.NamedSparseVector(
                name="sparse",
                vector=SparseVector(indices=sparse_indices, values=sparse_values),
            )
            sparse_hits = self.client.search(
                collection_name=target_collection,
                query_vector=sparse_query,
                limit=fetch_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

        if dense_hits and sparse_hits and dense_weight > 0 and sparse_weight > 0:
            k_const = 60.0
            fused: Dict[str, Dict[str, object]] = {}
            for rank, point in enumerate(dense_hits, start=1):
                payload = point.payload or {}
                chunk_id = str(payload.get("chunk_id", "")).strip()
                if not chunk_id:
                    continue
                entry = fused.setdefault(
                    chunk_id,
                    {
                        "score": 0.0,
                        "content": payload.get("content", ""),
                        "metadata": payload,
                    },
                )
                entry["score"] = float(entry["score"]) + (dense_weight / (k_const + rank))

            for rank, point in enumerate(sparse_hits, start=1):
                payload = point.payload or {}
                chunk_id = str(payload.get("chunk_id", "")).strip()
                if not chunk_id:
                    continue
                entry = fused.setdefault(
                    chunk_id,
                    {
                        "score": 0.0,
                        "content": payload.get("content", ""),
                        "metadata": payload,
                    },
                )
                entry["score"] = float(entry["score"]) + (sparse_weight / (k_const + rank))

            ranked = sorted(fused.items(), key=lambda item: float(item[1]["score"]), reverse=True)[:top_k]
            return [
                SearchResult(
                    chunk_id=chunk_id,
                    content=str(data["content"]),
                    score=float(data["score"]),
                    metadata=dict(data["metadata"]),
                )
                for chunk_id, data in ranked
            ]

        points = dense_hits if dense_hits else sparse_hits
        return [
            SearchResult(
                chunk_id=(point.payload or {}).get("chunk_id", ""),
                content=(point.payload or {}).get("content", ""),
                score=float(getattr(point, "score", 0.0) or 0.0),
                metadata=(point.payload or {}),
            )
            for point in points[:top_k]
        ]
    
    def search_by_metadata(
        self,
        doc_id: Optional[str] = None,
        van_ban_id: Optional[str] = None,
        loai_van_ban: Optional[str] = None,
        dieu_so: Optional[str] = None,
        limit: int = 20
    ) -> List[SearchResult]:
        """Search by metadata filters only"""
        conditions = []
        
        if doc_id:
            conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if van_ban_id:
            conditions.append(FieldCondition(key="van_ban_id", match=MatchValue(value=van_ban_id)))
        if loai_van_ban:
            conditions.append(FieldCondition(key="loai_van_ban", match=MatchValue(value=loai_van_ban)))
        if dieu_so:
            conditions.append(FieldCondition(key="dieu_so", match=MatchValue(value=dieu_so)))
        
        if not conditions:
            return []
        
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=conditions),
            limit=limit,
            with_payload=True
        )
        
        search_results = []
        for point in results[0]:
            search_results.append(SearchResult(
                chunk_id=point.payload.get("chunk_id", ""),
                content=point.payload.get("content", ""),
                score=1.0,  # Exact match
                metadata=point.payload
            ))
        
        return search_results
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "indexed_vectors_count": getattr(info, 'indexed_vectors_count', info.points_count),
                "status": info.status.value,
                "collection_name": self.collection_name,
                "dense_dim": self.dense_dim,
            }
        except Exception as e:
            if self.use_cloud and self.cloud_url:
                try:
                    import requests

                    url = f"{self.cloud_url.rstrip('/')}/collections/{self.collection_name}"
                    headers = {"api-key": self.api_key} if self.api_key else {}
                    resp = requests.get(url, headers=headers, timeout=self.timeout)
                    resp.raise_for_status()
                    payload = resp.json().get("result", {})
                    status_obj = payload.get("status")
                    status_text = status_obj
                    if isinstance(status_obj, dict):
                        status_text = status_obj.get("status", "")
                    return {
                        "points_count": payload.get("points_count"),
                        "indexed_vectors_count": payload.get("indexed_vectors_count"),
                        "status": status_text,
                        "collection_name": self.collection_name,
                        "dense_dim": self.dense_dim,
                    }
                except Exception:
                    pass
            return {"error": str(e)}
    
    def delete_by_van_ban(self, van_ban_id: str):
        """Delete all chunks from a specific van_ban"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="van_ban_id",
                            match=MatchValue(value=van_ban_id)
                        )
                    ]
                )
            )
        )
        logger.info(f"[QDRANT] Deleted documents for: {van_ban_id}")

    def delete_by_doc_id(self, doc_id: str):
        """Delete all chunks by canonical doc_id."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )
        logger.info(f"[QDRANT] Deleted documents for doc_id: {doc_id}")

    def fetch_by_doc_ids(
        self,
        doc_ids: Sequence[str],
        limit: int = 50,
        filter_conditions: Optional[Dict[str, object]] = None,
        collection_name: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Fetch chunks by a list of doc_ids (used by graph expansion).
        """
        if not doc_ids:
            return []

        merged_filters = dict(filter_conditions or {})
        merged_filters["doc_id"] = list(set(doc_ids))
        scroll_filter = self._build_filter(merged_filters)
        if scroll_filter is None:
            return []

        target_collection = collection_name if collection_name else self.collection_name
        points, _ = self.client.scroll(
            collection_name=target_collection,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
        )

        results: List[SearchResult] = []
        for point in points:
            payload = point.payload or {}
            results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    content=payload.get("content", ""),
                    score=1.0,
                    metadata=payload,
                )
            )
        return results
    
    def hybrid_search_with_rerank(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        use_reranker: bool = USE_RERANKER,
        rerank_weight: float = 0.6,
        filter_conditions: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Advanced hybrid search with RRF + Cross-Encoder Reranking
        
        Pipeline:
        1. Dense search (semantic) with Qdrant
        2. Sparse search (BM25) with Qdrant
        3. RRF fusion (built-in Qdrant)
        4. Cross-Encoder rerank (optional, improves quality)
        
        Args:
            query: Query text
            query_embedding: Dense embedding of query
            top_k: Number of final results
            use_reranker: Whether to use cross-encoder reranking
            rerank_weight: Weight for reranker scores (0=original, 1=rerank)
            filter_conditions: Optional metadata filters
            
        Returns:
            List of SearchResult sorted by final score
        """
        import time
        start = time.time()
        
        # Fetch more candidates for reranking
        fetch_k = top_k * 3 if use_reranker else top_k
        
        # Get initial results using Qdrant's built-in RRF
        initial_results = self.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            top_k=fetch_k,
            filter_conditions=filter_conditions
        )
        
        if not initial_results:
            return []
        
        # Apply cross-encoder reranking if enabled
        if use_reranker and len(initial_results) > 0:
            try:
                from src.reranker import LegalReranker
                
                # Lazy load reranker
                if not hasattr(self, '_reranker') or self._reranker is None:
                    self._reranker = LegalReranker()
                
                # Prepare documents for reranking
                docs_for_rerank = [
                    {
                        "content": r.content,
                        "score": r.score,
                        "chunk_id": r.chunk_id,
                        "metadata": r.metadata
                    }
                    for r in initial_results
                ]
                
                # Rerank
                reranked = self._reranker.rerank(
                    query=query,
                    documents=docs_for_rerank,
                    top_k=top_k,
                    rrf_weight=rerank_weight
                )
                
                # Convert back to SearchResult
                reranked_results = []
                for r in reranked:
                    reranked_results.append(SearchResult(
                        chunk_id=r.metadata.get("chunk_id", ""),
                        content=r.content,
                        score=r.final_score,
                        metadata=r.metadata.get("metadata", r.metadata)
                    ))
                
                elapsed = time.time() - start
                logger.info(f"[QDRANT] Hybrid search + rerank: {len(reranked_results)} results in {elapsed:.2f}s")
                
                return reranked_results
                
            except ImportError as e:
                logger.warning(f"[QDRANT] Reranker not available: {e}")
            except Exception as e:
                logger.warning(f"[QDRANT] Reranking failed, using RRF results: {e}")
        
        # Return RRF results without reranking
        elapsed = time.time() - start
        logger.info(f"[QDRANT] Hybrid search (RRF only): {len(initial_results[:top_k])} results in {elapsed:.2f}s")
        
        return initial_results[:top_k]


# Singleton instance with thread safety
_qdrant_store: Optional[QdrantLegalStore] = None
_qdrant_lock = None

def _get_lock():
    """Get or create threading lock"""
    global _qdrant_lock
    if _qdrant_lock is None:
        import threading
        _qdrant_lock = threading.Lock()
    return _qdrant_lock

def get_qdrant_store(max_retries: int = 3) -> QdrantLegalStore:
    """Get singleton Qdrant store instance with retry logic"""
    global _qdrant_store

    desired_signature = _qdrant_runtime_signature()

    # Return existing instance if available
    if _qdrant_store is not None:
        if getattr(_qdrant_store, "runtime_signature", None) == desired_signature:
            return _qdrant_store
        reset_qdrant_store()
    
    lock = _get_lock()
    with lock:
        # Double-check after acquiring lock
        if _qdrant_store is not None:
            if getattr(_qdrant_store, "runtime_signature", None) == desired_signature:
                return _qdrant_store
            reset_qdrant_store()
        
        import time
        last_error = None
        
        for attempt in range(max_retries):
            try:
                _qdrant_store = QdrantLegalStore()
                return _qdrant_store
            except RuntimeError as e:
                if "already accessed" in str(e):
                    last_error = e
                    logger.warning(f"[QDRANT] Retry {attempt + 1}/{max_retries}: {e}")
                    time.sleep(1)  # Wait before retry
                else:
                    raise
        
        raise RuntimeError(f"Failed to initialize Qdrant after {max_retries} retries: {last_error}")


def reset_qdrant_store():
    """Reset singleton (useful for testing or reload)"""
    global _qdrant_store
    if _qdrant_store is not None:
        try:
            _qdrant_store.client.close()
        except:
            pass
        _qdrant_store = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    store = QdrantLegalStore()
    store.create_collection(force_recreate=True)
    
    print("Stats:", store.get_stats())
