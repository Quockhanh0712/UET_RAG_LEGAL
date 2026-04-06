"""Qdrant runtime configuration and client construction."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from qdrant_client import QdrantClient

QDRANT_PATH = os.getenv("QDRANT_PATH", "./data/qdrant_db")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_USE_SERVER = os.getenv("QDRANT_USE_SERVER", "true").lower() == "true"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "legal_documents")
DENSE_DIM = int(os.getenv("QDRANT_DENSE_DIM", "768"))
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "120"))
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_USE_CLOUD = os.getenv("QDRANT_USE_CLOUD", "false").lower() == "true"


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def runtime_qdrant_config() -> Dict[str, Any]:
    return {
        "path": os.getenv("QDRANT_PATH", QDRANT_PATH),
        "host": os.getenv("QDRANT_HOST", QDRANT_HOST),
        "port": int(os.getenv("QDRANT_PORT", str(QDRANT_PORT))),
        "use_server": env_bool("QDRANT_USE_SERVER", QDRANT_USE_SERVER),
        "collection_name": os.getenv("QDRANT_COLLECTION", COLLECTION_NAME),
        "dense_dim": int(os.getenv("QDRANT_DENSE_DIM", str(DENSE_DIM))),
        "timeout": int(os.getenv("QDRANT_TIMEOUT", str(QDRANT_TIMEOUT))),
        "use_cloud": env_bool("QDRANT_USE_CLOUD", QDRANT_USE_CLOUD),
        "cloud_url": os.getenv("QDRANT_CLOUD_URL", QDRANT_CLOUD_URL),
        "api_key": os.getenv("QDRANT_API_KEY", QDRANT_API_KEY),
        "payload_profile": os.getenv("QDRANT_PAYLOAD_PROFILE", "full"),
    }


def qdrant_runtime_signature() -> Tuple[Any, ...]:
    cfg = runtime_qdrant_config()
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


def model_slug(model_name: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "_", str(model_name or "").lower()).strip("_")
    return slug[:80] or "unknown_model"


def collection_name_for_model(base_name: str, model_name: str, dense_dim: int) -> str:
    return f"{base_name}__{model_slug(model_name)}__d{int(dense_dim)}"


def remove_qdrant_lock_files(path: str, logger: Optional[logging.Logger] = None):
    import glob

    lock_patterns = [
        os.path.join(path, ".lock"),
        os.path.join(path, "*.lock"),
        os.path.join(path, "storage.lock"),
    ]
    for pattern in lock_patterns:
        for lock_file in glob.glob(pattern):
            try:
                os.remove(lock_file)
                if logger:
                    logger.info("[QDRANT] Removed lock file: %s", lock_file)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                if logger:
                    logger.warning("[QDRANT] Could not remove lock file %s: %s", lock_file, exc)


def create_qdrant_client(
    *,
    path: str,
    host: str,
    port: int,
    timeout: int,
    use_server: bool,
    use_cloud: bool,
    cloud_url: str,
    api_key: str,
    force_unlock: bool = True,
    client_cls=QdrantClient,
    logger: Optional[logging.Logger] = None,
):
    if use_cloud and cloud_url:
        client = client_cls(url=cloud_url, api_key=api_key, timeout=timeout)
        if logger:
            logger.info("[QDRANT] Connected to Cloud: %s...", cloud_url[:50])
        return client

    if use_server:
        client = client_cls(host=host, port=port, timeout=timeout)
        if logger:
            logger.info("[QDRANT] Connected to server at %s:%s", host, port)
        return client

    os.makedirs(path, exist_ok=True)
    if force_unlock:
        remove_qdrant_lock_files(path, logger=logger)
    client = client_cls(path=path, timeout=timeout)
    if logger:
        logger.info("[QDRANT] Initialized local storage at %s", path)
    return client

