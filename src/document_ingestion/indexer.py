from __future__ import annotations

import hashlib
import logging
import os
from contextlib import contextmanager
from importlib import import_module
from typing import Dict, Iterator, List, Optional

from qdrant_client.http import models

from src.config.embedding_contract import resolve_embedding_model_name
from src.legal_rag.stores.qdrant.client_factory import collection_name_for_model
from src.legal_rag.stores.qdrant.payload_profiles import build_qdrant_payload
from src.legal_rag.stores.qdrant.repository import QdrantLegalStore

from .models import IngestedChunk

logger = logging.getLogger(__name__)


def _embeddings_module():
    return import_module("src.embeddings")


def _get_embedding_dimension() -> int:
    return int(_embeddings_module().get_embedding_dimension())


def _embed_texts(texts: List[str], *, text_type: str) -> List[List[float]]:
    return _embeddings_module().embed_texts(texts, text_type=text_type)


def _embed_query(query: str) -> List[float]:
    return _embeddings_module().embed_query(query)


def _current_embedding_model_name() -> str:
    return resolve_embedding_model_name()


def _extract_dense_vector_dim(info: object) -> int:
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


class UserDocumentIndexer:
    def __init__(self, collection_name: str = "user_docs_private", payload_profile: str = "lean"):
        self.collection_name = collection_name
        self.embedding_model_name = _current_embedding_model_name()
        self.required_dense_dim = _get_embedding_dimension()
        self.versioned_collection_name = collection_name_for_model(
            collection_name,
            self.embedding_model_name,
            self.required_dense_dim,
        )
        self.store = QdrantLegalStore(
            collection_name=collection_name,
            dense_dim=self.required_dense_dim,
            payload_profile=payload_profile,
        )
        self._active_collection_name: Optional[str] = None

    @contextmanager
    def _use_store_collection(self, collection_name: str) -> Iterator[None]:
        original = self.store.collection_name
        self.store.collection_name = collection_name
        try:
            yield
        finally:
            self.store.collection_name = original

    def _collection_exists(self, collection_name: str) -> bool:
        try:
            self.store.client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def _get_collection_dim(self, collection_name: str) -> int:
        info = self.store.client.get_collection(collection_name)
        return _extract_dense_vector_dim(info)

    def _ensure_indexes(self, collection_name: str) -> None:
        with self._use_store_collection(collection_name):
            self.store._create_payload_indexes()

    def _ensure_collection_named(self, collection_name: str) -> None:
        if self._collection_exists(collection_name):
            actual_dim = self._get_collection_dim(collection_name)
            if actual_dim and actual_dim != self.required_dense_dim:
                raise RuntimeError(
                    "User document collection schema mismatch: "
                    f"collection={collection_name} expected_dim={self.required_dense_dim} actual_dim={actual_dim}"
                )
            self._ensure_indexes(collection_name)
            return

        with self._use_store_collection(collection_name):
            self.store.create_collection(force_recreate=False)
        self._ensure_indexes(collection_name)

    def _ensure_alias_target(self, target_collection: str) -> bool:
        current_target = None
        try:
            current_target = self.store.resolve_alias(self.collection_name)
        except Exception as exc:
            logger.debug("[INGEST] Could not resolve alias %s: %s", self.collection_name, exc)

        if current_target == target_collection:
            return True

        try:
            if current_target:
                self.store.swap_alias(self.collection_name, target_collection)
            else:
                self.store.set_alias(self.collection_name, target_collection)
            logger.info(
                "[INGEST] Alias %s -> %s is ready for user-doc indexing",
                self.collection_name,
                target_collection,
            )
            return True
        except Exception as exc:
            logger.warning(
                "[INGEST] Could not bind alias %s -> %s, falling back to physical collection. err=%s",
                self.collection_name,
                target_collection,
                exc,
            )
            return False

    def _set_active_collection(self, collection_name: str) -> None:
        self._active_collection_name = collection_name

    def _active_collection(self) -> str:
        if self._active_collection_name is None:
            self.ensure_collection()
        return self._active_collection_name or self.collection_name

    def ensure_collection(self) -> None:
        self._ensure_collection_named(self.versioned_collection_name)

        alias_target = None
        try:
            alias_target = self.store.resolve_alias(self.collection_name)
        except Exception as exc:
            logger.debug("[INGEST] Alias lookup failed for %s: %s", self.collection_name, exc)

        if alias_target:
            alias_dim = self._get_collection_dim(alias_target)
            if alias_target == self.versioned_collection_name and alias_dim == self.required_dense_dim:
                self._ensure_indexes(self.versioned_collection_name)
                self._set_active_collection(self.collection_name)
                return

            if self._ensure_alias_target(self.versioned_collection_name):
                self._set_active_collection(self.collection_name)
            else:
                self._set_active_collection(self.versioned_collection_name)
            logger.warning(
                "[INGEST] Alias target mismatch for %s: expected_collection=%s expected_dim=%s actual_collection=%s actual_dim=%s",
                self.collection_name,
                self.versioned_collection_name,
                self.required_dense_dim,
                alias_target,
                alias_dim,
            )
            return

        if self._collection_exists(self.collection_name):
            base_dim = self._get_collection_dim(self.collection_name)
            self._set_active_collection(self.versioned_collection_name)
            logger.warning(
                "[INGEST] Legacy collection %s occupies logical user-doc name (dim=%s). "
                "Routing user-doc ingest to versioned collection %s",
                self.collection_name,
                base_dim,
                self.versioned_collection_name,
            )
            return

        if self._ensure_alias_target(self.versioned_collection_name):
            self._set_active_collection(self.collection_name)
        else:
            self._set_active_collection(self.versioned_collection_name)

    def upsert(self, chunks: List[IngestedChunk]) -> None:
        if not chunks:
            return

        self.ensure_collection()
        target_collection = self._active_collection()
        dense_vectors = _embed_texts([chunk.content for chunk in chunks], text_type="passage")
        points: List[models.PointStruct] = []
        for chunk, dense_vector in zip(chunks, dense_vectors):
            sparse_indices: List[int] = []
            sparse_values: List[float] = []
            if self.store.bm25_encoder.vocab:
                sparse_indices, sparse_values = self.store.bm25_encoder.encode(chunk.content)

            point_id = int(hashlib.md5(chunk.chunk_id.encode("utf-8")).hexdigest()[:16], 16)
            vectors: Dict[str, object] = {"dense": dense_vector}
            if sparse_indices:
                vectors["sparse"] = models.SparseVector(indices=sparse_indices, values=sparse_values)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload=build_qdrant_payload(chunk, profile=self.store.payload_profile),
                )
            )

        for start in range(0, len(points), 32):
            self.store.client.upsert(
                collection_name=target_collection,
                points=points[start : start + 32],
                wait=True,
            )

    def search(
        self,
        *,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        use_hybrid: bool = True,
    ) -> List[Dict]:
        self.ensure_collection()
        target_collection = self._active_collection()
        filter_conditions: Dict[str, object] = {"user_id": user_id}

        if not use_hybrid or not self.store.bm25_encoder.vocab:
            query_embedding = _embed_query(query)
            qdrant_filter = self.store._build_filter(filter_conditions)
            response = self.store.client.query_points(
                collection_name=target_collection,
                query=query_embedding,
                using="dense",
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            points = response.points if hasattr(response, "points") else response
            return [
                {
                    "content": point.payload.get("content", ""),
                    "score": float(getattr(point, "score", 0.0) or 0.0),
                    "filename": point.payload.get("filename", ""),
                    "doc_id": point.payload.get("doc_id", ""),
                    "page": point.payload.get("page_number"),
                    "source_type": "user_upload",
                    "payload": point.payload,
                }
                for point in points
            ]

        results = self.store.hybrid_search(
            query=query,
            query_embedding=_embed_query(query),
            top_k=top_k,
            filter_conditions=filter_conditions,
            collection_name=target_collection,
        )
        return [
            {
                "content": result.content,
                "score": result.score,
                "filename": result.metadata.get("filename", ""),
                "doc_id": result.metadata.get("doc_id", ""),
                "page": result.metadata.get("page_number"),
                "source_type": "user_upload",
                "payload": result.metadata,
            }
            for result in results
        ]

    def delete_document(self, *, doc_id: str, user_id: str) -> None:
        self.ensure_collection()
        target_collection = self._active_collection()
        self.store.client.delete(
            collection_name=target_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                    ]
                )
            ),
        )
