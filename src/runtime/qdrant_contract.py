from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from qdrant_client import QdrantClient

from src.config.runtime_settings import RuntimeSettings
from src.qdrant_store import collection_name_for_model


@dataclass(frozen=True)
class QdrantContractStatus:
    collection_name: str
    collection_status: str
    embedding_model_id: str
    vector_dim: int
    bm25_manifest_id: str
    contract_ok: bool
    contract_error: str = ""


def resolve_collection_name(settings: RuntimeSettings, embedding_dim: int) -> str:
    if settings.collection_override:
        return settings.collection_override
    if settings.collection_by_model:
        return collection_name_for_model(settings.collection_base, settings.embedding_model, embedding_dim)
    return settings.collection_base


def get_qdrant_client(settings: RuntimeSettings, timeout: int = 30) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=timeout)


def _extract_collection_status(info: Any) -> str:
    raw = getattr(info, "status", "")
    if hasattr(raw, "value"):
        raw = raw.value
    return str(raw or "").strip().lower()


def _extract_dense_vector_dim(info: Any) -> int:
    try:
        params = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(params, "vectors", None)
        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict):
            dense = vectors.get("dense")
            size = getattr(dense, "size", None)
            if size is not None:
                return int(size)
    except Exception:
        pass
    return 0


def validate_bm25_contract(settings: RuntimeSettings, collection_name: str) -> tuple[Path, str]:
    bm25_path = settings.bm25_path
    if not bm25_path.exists():
        raise RuntimeError(f"BM25 encoder not found: {bm25_path}")

    manifest_id = ""
    if settings.bm25_manifest:
        manifest_path = settings.bm25_manifest
        if not manifest_path.exists():
            raise RuntimeError(f"BM25 manifest path not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_id = str(payload.get("run_id", manifest_path.name))
        collections = payload.get("collections", {})
        if collection_name not in collections:
            raise RuntimeError(
                "BM25 manifest mismatch: target collection not present in manifest. "
                "Set correct CHAINLIT_BM25_MANIFEST or disable strict contract validation."
            )
    return bm25_path, manifest_id


def _ensure_queryable(client: QdrantClient, collection_name: str) -> None:
    client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=False,
        with_vectors=False,
    )


def validate_qdrant_contract(settings: RuntimeSettings, embedding_dim: int) -> QdrantContractStatus:
    if not settings.qdrant_use_cloud:
        raise RuntimeError("QDRANT_USE_CLOUD must be true for Chainlit runtime.")
    if not settings.qdrant_url:
        raise RuntimeError("Missing required env: QDRANT_CLOUD_URL")
    if not settings.qdrant_api_key:
        raise RuntimeError("Missing required env: QDRANT_API_KEY")
    if settings.collection_base != "legal_documents":
        raise RuntimeError("QDRANT_COLLECTION must be legal_documents.")
    if not settings.collection_by_model and not settings.collection_override:
        raise RuntimeError("QDRANT_COLLECTION_BY_MODEL must be true unless collection override is set.")
    if settings.embedding_model != "truro7/vn-law-embedding":
        raise RuntimeError("EMBEDDING_MODEL must be truro7/vn-law-embedding for this profile.")
    if not settings.normalize_embeddings:
        raise RuntimeError("EMBEDDING_NORMALIZE must be true for this profile.")
    if settings.llm_provider.lower() != "ollama":
        raise RuntimeError("LLM_PROVIDER must be ollama for this profile.")
    if not settings.ollama_url:
        raise RuntimeError("Missing required env: OLLAMA_URL")
    if not settings.ollama_model:
        raise RuntimeError("Missing required env: OLLAMA_MODEL")

    collection_name = resolve_collection_name(settings, embedding_dim)
    bm25_path, manifest_id = validate_bm25_contract(settings, collection_name)
    client = get_qdrant_client(settings)
    names = [c.name for c in client.get_collections().collections]
    if collection_name not in names:
        preview = ", ".join(names[:20]) if names else "<none>"
        raise RuntimeError(
            f"Collection not found on cloud: {collection_name} | url={settings.qdrant_url} | "
            f"available={preview} => Set QDRANT_CLOUD_URL to the indexed cluster or set CHAINLIT_COLLECTION_OVERRIDE."
        )

    info = client.get_collection(collection_name)
    status = _extract_collection_status(info)
    if status == "red":
        raise RuntimeError(f"Collection {collection_name} is not queryable (status=red).")
    if status not in {"green", "yellow"}:
        raise RuntimeError(f"Collection {collection_name} has unsupported status={status}.")

    _ensure_queryable(client, collection_name)
    vector_dim = _extract_dense_vector_dim(info) or int(embedding_dim)
    return QdrantContractStatus(
        collection_name=collection_name,
        collection_status=status,
        embedding_model_id=settings.embedding_model,
        vector_dim=vector_dim,
        bm25_manifest_id=manifest_id or str(bm25_path.name),
        contract_ok=True,
    )

