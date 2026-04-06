"""Compatibility facade for Qdrant legal store modules."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

from qdrant_client import QdrantClient

from src.legal_rag.stores.qdrant.bm25_encoder import BM25Encoder
from src.legal_rag.stores.qdrant.client_factory import (
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
    env_bool,
    model_slug,
    qdrant_runtime_signature,
    runtime_qdrant_config,
)
from src.legal_rag.stores.qdrant.payload_profiles import (
    LEAN_QDRANT_PAYLOAD_FIELDS,
    build_qdrant_payload,
    payload_value_is_present,
)
from src.legal_rag.stores.qdrant.repository import QdrantLegalStore as _QdrantLegalStore
from src.legal_rag.stores.qdrant.repository import SearchResult

logger = logging.getLogger(__name__)


# Backward-compatible helper aliases.
def _env_bool(name: str, default: bool) -> bool:
    return env_bool(name, default)


def _runtime_qdrant_config() -> Dict[str, Any]:
    return runtime_qdrant_config()


def _qdrant_runtime_signature():
    return qdrant_runtime_signature()


class QdrantLegalStore(_QdrantLegalStore):
    """Facade subclass that keeps `src.qdrant_store.QdrantClient` monkeypatchable."""

    def __init__(self, *args, client_cls=None, **kwargs):
        if client_cls is None:
            client_cls = QdrantClient
        kwargs.setdefault("client_cls", client_cls)
        super().__init__(*args, **kwargs)


_qdrant_store: Optional[QdrantLegalStore] = None
_qdrant_lock: Optional[threading.Lock] = None


def _get_lock() -> threading.Lock:
    global _qdrant_lock
    if _qdrant_lock is None:
        _qdrant_lock = threading.Lock()
    return _qdrant_lock


def get_qdrant_store(max_retries: int = 3) -> QdrantLegalStore:
    """Get singleton Qdrant store instance with runtime-signature invalidation."""

    global _qdrant_store

    desired_signature = qdrant_runtime_signature()

    if _qdrant_store is not None:
        if getattr(_qdrant_store, "runtime_signature", None) == desired_signature:
            return _qdrant_store
        reset_qdrant_store()

    lock = _get_lock()
    with lock:
        if _qdrant_store is not None:
            if getattr(_qdrant_store, "runtime_signature", None) == desired_signature:
                return _qdrant_store
            reset_qdrant_store()

        last_error: Optional[RuntimeError] = None
        for attempt in range(max_retries):
            try:
                _qdrant_store = QdrantLegalStore()
                return _qdrant_store
            except RuntimeError as exc:
                if "already accessed" in str(exc):
                    last_error = exc
                    logger.warning("[QDRANT] Retry %s/%s: %s", attempt + 1, max_retries, exc)
                    time.sleep(1)
                else:
                    raise

        raise RuntimeError(
            f"Failed to initialize Qdrant after {max_retries} retries: {last_error}"
        )


def reset_qdrant_store():
    """Reset singleton (used by tests and runtime reconfiguration)."""

    global _qdrant_store
    if _qdrant_store is not None:
        try:
            _qdrant_store.client.close()
        except Exception:
            pass
        _qdrant_store = None


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
    "LEAN_QDRANT_PAYLOAD_FIELDS",
    "SearchResult",
    "BM25Encoder",
    "QdrantLegalStore",
    "build_qdrant_payload",
    "payload_value_is_present",
    "get_qdrant_store",
    "reset_qdrant_store",
    "model_slug",
    "collection_name_for_model",
    "_env_bool",
    "_runtime_qdrant_config",
    "_qdrant_runtime_signature",
]
