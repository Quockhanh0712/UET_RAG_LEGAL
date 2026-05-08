from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from qdrant_client.http import models

from src.config.embedding_contract import resolve_embedding_model_name
from src.legal_rag.stores.qdrant.client_factory import collection_name_for_model
from src.legal_rag.stores.qdrant.payload_profiles import build_qdrant_payload
from src.legal_rag.stores.qdrant.repository import QdrantLegalStore

from .models import IngestedChunk

logger = logging.getLogger(__name__)

DOCUMENT_SCOPE_CURRENT_SESSION = "current_session"
DOCUMENT_SCOPE_ALL_USER_DOCS = "all_user_docs"


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

    @staticmethod
    def _normalize_document_scope(document_scope: Optional[str]) -> str:
        value = str(document_scope or DOCUMENT_SCOPE_CURRENT_SESSION).strip().lower()
        if value == DOCUMENT_SCOPE_ALL_USER_DOCS:
            return DOCUMENT_SCOPE_ALL_USER_DOCS
        return DOCUMENT_SCOPE_CURRENT_SESSION

    @staticmethod
    def _point_id_from_chunk_id(chunk_id: str) -> int:
        return int(hashlib.md5(chunk_id.encode("utf-8")).hexdigest()[:16], 16)

    @staticmethod
    def _payload_chunk_id(payload: Dict[str, object]) -> str:
        chunk_id = str(payload.get("chunk_id") or "").strip()
        if chunk_id:
            return chunk_id
        doc_id = str(payload.get("doc_id") or "").strip()
        content = str(payload.get("content") or "").strip()
        page = payload.get("page_number")
        return f"user::{doc_id}::{page}::{hashlib.md5(content.encode('utf-8')).hexdigest()[:12]}"

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

    def _physical_collection_name(self, collection_name: Optional[str] = None) -> str:
        active = str(collection_name or self._active_collection_name or self.collection_name)
        try:
            resolved = self.store.resolve_alias(active)
        except Exception:
            resolved = None
        return str(resolved or active)

    def _reset_bm25_encoder(self) -> None:
        encoder_cls = type(self.store.bm25_encoder)
        self.store.bm25_encoder = encoder_cls()

    def _bm25_artifact_path(self, collection_name: str) -> str:
        return str(Path(self.store.path) / "bm25" / f"{collection_name}.json")

    def _sync_bm25_artifact(self, collection_name: Optional[str] = None) -> None:
        physical_collection = self._physical_collection_name(collection_name)
        artifact_path = self._bm25_artifact_path(physical_collection)
        self.store.set_bm25_path(artifact_path, load_if_exists=False)
        self._reset_bm25_encoder()
        if Path(artifact_path).exists():
            self.store.bm25_encoder.load(artifact_path)

    def _set_active_collection(self, collection_name: str) -> None:
        self._active_collection_name = collection_name
        self._sync_bm25_artifact(collection_name)

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

    def _scroll_points(
        self,
        *,
        collection_name: str,
        filter_conditions: Optional[Dict[str, object]] = None,
        with_vectors: bool = False,
        limit: int = 256,
    ) -> List[object]:
        scroll_filter = self.store._build_filter(filter_conditions)
        offset = None
        points: List[object] = []
        while True:
            batch, offset = self.store.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
                with_vectors=with_vectors,
                offset=offset,
            )
            points.extend(list(batch or []))
            if offset is None:
                break
        return points

    @staticmethod
    def _extract_dense_vector(point: object) -> Optional[List[float]]:
        vector = getattr(point, "vector", None)
        if isinstance(vector, dict):
            dense = vector.get("dense")
            if dense is not None:
                return list(dense)
        if isinstance(vector, list):
            return list(vector)
        if hasattr(vector, "get"):
            dense = vector.get("dense")
            if dense is not None:
                return list(dense)
        return None

    def _refit_sparse_encoder(self, collection_name: str) -> List[object]:
        points = self._scroll_points(collection_name=collection_name, with_vectors=True)
        contents = [
            str((getattr(point, "payload", {}) or {}).get("content") or "").strip()
            for point in points
        ]
        contents = [content for content in contents if content]

        self._sync_bm25_artifact(collection_name)
        artifact_path = Path(self.store.bm25_path)
        if not contents:
            if artifact_path.exists():
                artifact_path.unlink()
            self._reset_bm25_encoder()
            return points

        self.store.bm25_encoder.fit(contents)
        self.store.bm25_encoder.save(self.store.bm25_path)
        return points

    def _reindex_sparse_vectors(self, collection_name: str, points: List[object]) -> None:
        payloads: List[Dict[str, object]] = []
        dense_vectors: List[Optional[List[float]]] = []
        missing_dense_payloads: List[Dict[str, object]] = []

        for point in points:
            payload = dict(getattr(point, "payload", {}) or {})
            content = str(payload.get("content") or "").strip()
            if not content:
                continue
            payloads.append(payload)
            dense = self._extract_dense_vector(point)
            dense_vectors.append(dense)
            if dense is None:
                missing_dense_payloads.append(payload)

        if not payloads:
            return

        if missing_dense_payloads:
            embedded = _embed_texts([str(payload.get("content") or "") for payload in missing_dense_payloads], text_type="passage")
            iterator = iter(embedded)
            dense_vectors = [vector if vector is not None else next(iterator) for vector in dense_vectors]

        upsert_points: List[models.PointStruct] = []
        for payload, dense_vector in zip(payloads, dense_vectors):
            if dense_vector is None:
                continue
            sparse_indices, sparse_values = self.store.bm25_encoder.encode(str(payload.get("content") or ""))
            vectors: Dict[str, object] = {"dense": dense_vector}
            if sparse_indices:
                vectors["sparse"] = models.SparseVector(indices=sparse_indices, values=sparse_values)
            chunk_id = self._payload_chunk_id(payload)
            payload["chunk_id"] = chunk_id
            upsert_points.append(
                models.PointStruct(
                    id=self._point_id_from_chunk_id(chunk_id),
                    vector=vectors,
                    payload=build_qdrant_payload(payload, profile=self.store.payload_profile),
                )
            )

        for start in range(0, len(upsert_points), 32):
            self.store.client.upsert(
                collection_name=collection_name,
                points=upsert_points[start : start + 32],
                wait=True,
            )

    def _refresh_collection_sparse_index(self, collection_name: str) -> None:
        points = self._refit_sparse_encoder(collection_name)
        if points:
            self._reindex_sparse_vectors(collection_name, points)

    def upsert(self, chunks: List[IngestedChunk]) -> None:
        if not chunks:
            return

        self.ensure_collection()
        write_collection = self._active_collection()
        dense_vectors = _embed_texts([chunk.content for chunk in chunks], text_type="passage")
        points: List[models.PointStruct] = []
        for chunk, dense_vector in zip(chunks, dense_vectors):
            sparse_indices: List[int] = []
            sparse_values: List[float] = []
            if self.store.bm25_encoder.vocab:
                sparse_indices, sparse_values = self.store.bm25_encoder.encode(chunk.content)

            vectors: Dict[str, object] = {"dense": dense_vector}
            if sparse_indices:
                vectors["sparse"] = models.SparseVector(indices=sparse_indices, values=sparse_values)
            points.append(
                models.PointStruct(
                    id=self._point_id_from_chunk_id(chunk.chunk_id),
                    vector=vectors,
                    payload=build_qdrant_payload(chunk, profile=self.store.payload_profile),
                )
            )

        for start in range(0, len(points), 32):
            self.store.client.upsert(
                collection_name=write_collection,
                points=points[start : start + 32],
                wait=True,
            )

        self._refresh_collection_sparse_index(self._physical_collection_name(write_collection))

    def _build_scope_filter(
        self,
        *,
        user_id: str,
        session_id: Optional[str],
        document_scope: Optional[str],
    ) -> Dict[str, object]:
        scope = self._normalize_document_scope(document_scope)
        filter_conditions: Dict[str, object] = {"user_id": user_id}
        if scope == DOCUMENT_SCOPE_CURRENT_SESSION:
            filter_conditions["session_id"] = session_id if session_id not in (None, "") else "__missing_session__"
        return filter_conditions

    @staticmethod
    def _normalize_point_result(point: object) -> Dict:
        payload = dict(getattr(point, "payload", {}) or {})
        return {
            "content": payload.get("content", ""),
            "score": float(getattr(point, "score", 0.0) or 0.0),
            "filename": payload.get("filename", ""),
            "doc_id": payload.get("doc_id", ""),
            "page": payload.get("page_number"),
            "source_type": "user_upload",
            "payload": payload,
        }

    @staticmethod
    def _normalize_search_result(result: object) -> Dict:
        metadata = dict(getattr(result, "metadata", {}) or {})
        return {
            "content": getattr(result, "content", metadata.get("content", "")),
            "score": float(getattr(result, "score", 0.0) or 0.0),
            "filename": metadata.get("filename", ""),
            "doc_id": metadata.get("doc_id", ""),
            "page": metadata.get("page_number"),
            "source_type": "user_upload",
            "payload": metadata,
        }

    def search(
        self,
        *,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        use_hybrid: bool = True,
        document_scope: str = DOCUMENT_SCOPE_CURRENT_SESSION,
        dense_only: bool = False,
    ) -> List[Dict]:
        self.ensure_collection()
        target_collection = self._active_collection()
        filter_conditions = self._build_scope_filter(
            user_id=user_id,
            session_id=session_id,
            document_scope=document_scope,
        )

        if dense_only or not use_hybrid or not self.store.bm25_encoder.vocab:
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
            return [self._normalize_point_result(point) for point in list(points or [])]

        results = self.store.hybrid_search(
            query=query,
            query_embedding=_embed_query(query),
            top_k=top_k,
            filter_conditions=filter_conditions,
            collection_name=target_collection,
        )
        return [self._normalize_search_result(result) for result in list(results or [])]

    def fetch_section_context(
        self,
        *,
        user_id: str,
        session_id: Optional[str],
        parent_section_id: str,
        document_scope: str = DOCUMENT_SCOPE_CURRENT_SESSION,
        limit: int = 8,
    ) -> List[Dict]:
        if not str(parent_section_id or "").strip():
            return []
        self.ensure_collection()
        filter_conditions = self._build_scope_filter(
            user_id=user_id,
            session_id=session_id,
            document_scope=document_scope,
        )
        filter_conditions["parent_section_id"] = str(parent_section_id)
        points = self._scroll_points(
            collection_name=self._active_collection(),
            filter_conditions=filter_conditions,
            with_vectors=False,
            limit=max(limit, 32),
        )
        normalized = [self._normalize_point_result(point) for point in list(points or [])]
        normalized.sort(
            key=lambda item: (
                int((item.get("payload") or {}).get("chunk_index", 0) or 0),
                str((item.get("payload") or {}).get("chunk_id", "") or ""),
            )
        )
        return normalized[:limit]

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
        self._refresh_collection_sparse_index(self._physical_collection_name(target_collection))
