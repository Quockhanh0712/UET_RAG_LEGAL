"""Qdrant store modules for Legal RAG."""

from .bm25_encoder import BM25Encoder
from .client_factory import (
    COLLECTION_NAME,
    DENSE_DIM,
    QDRANT_API_KEY,
    QDRANT_CLOUD_URL,
    QDRANT_HOST,
    QDRANT_PATH,
    QDRANT_PORT,
    QDRANT_TIMEOUT,
    QDRANT_USE_CLOUD,
    QDRANT_USE_SERVER,
    collection_name_for_model,
    model_slug,
    qdrant_runtime_signature,
    runtime_qdrant_config,
)
from .payload_profiles import LEAN_QDRANT_PAYLOAD_FIELDS, build_qdrant_payload, payload_value_is_present
from .repository import QdrantLegalStore, SearchResult

__all__ = [
    "QDRANT_PATH",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_USE_SERVER",
    "COLLECTION_NAME",
    "DENSE_DIM",
    "QDRANT_TIMEOUT",
    "QDRANT_CLOUD_URL",
    "QDRANT_API_KEY",
    "QDRANT_USE_CLOUD",
    "collection_name_for_model",
    "model_slug",
    "runtime_qdrant_config",
    "qdrant_runtime_signature",
    "LEAN_QDRANT_PAYLOAD_FIELDS",
    "build_qdrant_payload",
    "payload_value_is_present",
    "SearchResult",
    "BM25Encoder",
    "QdrantLegalStore",
]
