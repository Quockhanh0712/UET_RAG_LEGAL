"""
Legal RAG pipeline (hybrid retrieval + optional graph expansion).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.embeddings import DEFAULT_MODEL, embed_query, embed_texts, get_embedding_dimension
from src.qdrant_store import COLLECTION_NAME, collection_name_for_model

logger = logging.getLogger(__name__)

# Lazy imports
QdrantLegalStore = None
SearchResult = None
QueryIntentClassifier = None
IntentResult = None


def _lazy_imports():
    global QdrantLegalStore, SearchResult, QueryIntentClassifier, IntentResult
    if QdrantLegalStore is None:
        from src.qdrant_store import QdrantLegalStore, SearchResult
        from src.query_intent import IntentResult, QueryIntentClassifier
    return QdrantLegalStore, SearchResult, QueryIntentClassifier, IntentResult


@dataclass
class LegalAnswer:
    answer: str
    citations: List[Dict]
    intent: str
    retrieval_time: float
    generation_time: float
    rerank_time: float = 0.0
    used_reranker: bool = False
    graph_expansion_used: bool = False
    graph_mode: str = "none"

    def format_with_citations(self) -> str:
        if not self.citations:
            return self.answer

        citation_text = "\n\n---\n\n**CAN CU PHAP LY:**\n\n"
        seen = set()
        visible_index = 1

        for cite in self.citations:
            key = (
                cite.get("doc_id", cite.get("van_ban_id", "")),
                cite.get("dieu_so", ""),
                cite.get("khoan_so", ""),
                cite.get("diem_so", ""),
            )
            if key in seen:
                continue
            seen.add(key)

            citation_full = cite.get("citation_full", "")
            if citation_full:
                citation_text += f"{visible_index}. {citation_full}\n"
                visible_index += 1
                continue

            ten_vb = cite.get("ten_van_ban", "Van ban phap luat")
            loai_vb = cite.get("loai_van_ban", "")
            co_quan = cite.get("co_quan", "")
            dieu = cite.get("dieu_so", "")
            khoan = cite.get("khoan_so", "")
            diem = cite.get("diem_so", "")

            detail_parts = []
            if dieu:
                detail_parts.append(f"Dieu {dieu}")
            if khoan and khoan != "0":
                detail_parts.append(f"Khoan {khoan}")
            if diem:
                detail_parts.append(f"Diem {diem})")

            source = f"[{loai_vb}] {ten_vb}" if loai_vb else ten_vb
            if co_quan:
                source = f"{source} - {co_quan}"

            if detail_parts:
                citation_text += f"{visible_index}. {' - '.join(detail_parts)} | {source}\n"
            else:
                citation_text += f"{visible_index}. {source}\n"
            visible_index += 1

        return self.answer + citation_text


class LegalEmbedding:
    """
    Wrapper around src.embeddings to keep one embedding implementation across
    ingestion and retrieval.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, normalize_embeddings: Optional[bool] = None):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.embedding_dim = 0

    def load(self):
        # Model is lazily loaded in embed_texts() via src.embeddings.get_model()
        self.embedding_dim = int(get_embedding_dimension(self.model_name))
        return

    def encode(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        return embed_texts(
            texts,
            batch_size=batch_size,
            model_name=self.model_name,
            normalize_embeddings=self.normalize_embeddings,
            text_type="passage",
            add_prefix=True,
        )

    def encode_query(self, query: str) -> List[float]:
        return embed_query(
            query,
            model_name=self.model_name,
            normalize_embeddings=self.normalize_embeddings,
        )


class LegalLLM:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "ollama")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.system_prompt = (
            "Ban la tro ly phap ly Viet Nam. "
            "Chi duoc tra loi dua tren context da cung cap. "
            "Neu khong du thong tin, phai noi ro khong tim thay quy dinh lien quan."
        )

    def generate(self, query: str, context: str) -> str:
        prompt = f"CONTEXT:\n{context}\n\nCAU HOI: {query}\n\nTRA LOI:"
        if self.provider == "ollama":
            return self._generate_ollama(prompt)
        return self._generate_fallback(prompt)

    def _generate_ollama(self, prompt: str) -> str:
        import requests

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": self.system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "num_predict": 2048,
                        "num_ctx": 32768,
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            if "<think>" in text:
                import re

                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text
        except Exception as exc:
            logger.error("[LLM] Error: %s", exc)
            return f"Loi khi goi LLM: {exc}"

    @staticmethod
    def _generate_fallback(prompt: str) -> str:
        return "LLM khong kha dung. Vui long kiem tra cau hinh."


class LegalRAGPipeline:
    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        normalize_embeddings: Optional[bool] = None,
    ):
        _lazy_imports()
        self.qdrant_store = None
        self.graph_store = None
        self.intent_classifier = None
        self.embedding_model = None
        self.llm = None
        self._reranker = None
        self._initialized = False
        self.embedding_model_name = embedding_model_name or DEFAULT_MODEL
        self.normalize_embeddings = normalize_embeddings

    def initialize(self):
        if self._initialized:
            return

        from src.neo4j_store import get_graph_store
        from src.qdrant_store import get_qdrant_store
        from src.query_intent import get_intent_classifier

        logger.info("[PIPELINE] Initializing Legal RAG pipeline...")
        self.qdrant_store = get_qdrant_store()
        self.graph_store = get_graph_store()
        self.intent_classifier = get_intent_classifier()
        self.embedding_model = LegalEmbedding(
            model_name=self.embedding_model_name,
            normalize_embeddings=self.normalize_embeddings,
        )
        self.llm = LegalLLM()
        self.embedding_model.load()
        self.qdrant_store.dense_dim = int(self.embedding_model.embedding_dim)

        # Keep vector spaces isolated by embedding model and dimension.
        if os.getenv("QDRANT_COLLECTION_BY_MODEL", "true").lower() == "true":
            collection_base = os.getenv("QDRANT_COLLECTION", COLLECTION_NAME)
            self.qdrant_store.collection_name = collection_name_for_model(
                collection_base,
                self.embedding_model_name,
                self.embedding_model.embedding_dim,
            )
        self._initialized = True
        logger.info("[PIPELINE] Initialization complete")

    def _merge_results(self, primary: List, secondary: List) -> List:
        merged: Dict[str, object] = {}
        for result in list(primary) + list(secondary):
            chunk_id = getattr(result, "chunk_id", "")
            if not chunk_id:
                continue
            if chunk_id not in merged or getattr(result, "score", 0.0) > getattr(merged[chunk_id], "score", 0.0):
                merged[chunk_id] = result
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)

    def _rerank_results(self, question: str, results: List, top_k: int) -> Tuple[List, float]:
        if not results:
            return [], 0.0

        from src.qdrant_store import SearchResult
        from src.reranker import LegalReranker

        if self._reranker is None:
            self._reranker = LegalReranker()

        docs = [
            {
                "content": r.content,
                "score": r.score,
                "chunk_id": r.chunk_id,
                "metadata": r.metadata,
            }
            for r in results
        ]

        start = time.time()
        reranked = self._reranker.rerank(
            query=question,
            documents=docs,
            top_k=top_k,
            rrf_weight=0.6,
        )
        elapsed = time.time() - start

        reranked_results = []
        for item in reranked:
            metadata = item.metadata.get("metadata", item.metadata)
            reranked_results.append(
                SearchResult(
                    chunk_id=metadata.get("chunk_id", ""),
                    content=item.content,
                    score=item.final_score,
                    metadata=metadata,
                )
            )
        return reranked_results, elapsed

    def _graph_expand(
        self,
        base_results: List,
        filter_conditions: Dict[str, object],
        graph_hops: int,
        graph_expand_docs: int,
        top_k: int,
        graph_mode: str = "local",
    ) -> List:
        if not self.graph_store or not getattr(self.graph_store, "enabled", False):
            return []

        doc_ids = []
        for result in base_results:
            doc_id = result.metadata.get("doc_id") or result.metadata.get("van_ban_id")
            if doc_id:
                doc_ids.append(doc_id)

        if not doc_ids:
            return []

        related_doc_ids = self.graph_store.expand_related_doc_ids(
            anchor_doc_ids=list(set(doc_ids)),
            hops=max(1, min(int(graph_hops), 1)),
            limit=graph_expand_docs,
            include_same_type=(graph_mode == "local"),
            mode=graph_mode,
        )
        related_doc_ids = [doc_id for doc_id in related_doc_ids if doc_id not in doc_ids]
        if not related_doc_ids:
            return []

        expanded = self.qdrant_store.fetch_by_doc_ids(
            doc_ids=related_doc_ids,
            limit=max(top_k, graph_expand_docs),
            filter_conditions=filter_conditions,
        )
        # Slight score discount for graph-expanded results vs direct vector hits.
        for item in expanded:
            item.score = item.score * 0.85
        return expanded

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_conditions: Optional[Dict] = None,
        use_reranker: bool = True,
        include_inactive: bool = False,
        use_graph_context: bool = True,
        graph_hops: int = 1,
        graph_expand_docs: int = 20,
    ) -> LegalAnswer:
        self.initialize()
        intent_result = self.intent_classifier.classify(question)
        if top_k is None:
            top_k = intent_result.top_k

        effective_filters: Dict[str, object] = dict(filter_conditions or {})
        if not include_inactive and "is_active" not in effective_filters:
            effective_filters["is_active"] = True

        start_retrieval = time.time()
        query_embedding = self.embedding_model.encode_query(question)
        retrieval_pool_k = 100 if use_reranker else max(top_k * 2, top_k)

        base_results = self.qdrant_store.hybrid_search(
            query=question,
            query_embedding=query_embedding,
            top_k=retrieval_pool_k,
            dense_weight=intent_result.dense_weight,
            sparse_weight=intent_result.sparse_weight,
            filter_conditions=effective_filters,
        )
        if not base_results and effective_filters.get("is_active") is True:
            # Backward compatibility for legacy indexes without is_active payload.
            fallback_filters = dict(effective_filters)
            fallback_filters.pop("is_active", None)
            base_results = self.qdrant_store.hybrid_search(
                query=question,
                query_embedding=query_embedding,
                top_k=retrieval_pool_k,
                dense_weight=intent_result.dense_weight,
                sparse_weight=intent_result.sparse_weight,
                filter_conditions=fallback_filters,
            )

        graph_results: List = []
        graph_expansion_used = False
        graph_mode = getattr(intent_result, "graph_mode", "none")
        if use_graph_context and graph_mode != "none":
            shortlist_size = max(top_k, min(graph_expand_docs, 30))
            shortlist = base_results[:shortlist_size]
            graph_results = self._graph_expand(
                base_results=shortlist,
                filter_conditions=effective_filters,
                graph_hops=max(1, min(graph_hops, 1)),
                graph_expand_docs=graph_expand_docs,
                top_k=top_k,
                graph_mode=graph_mode,
            )
            graph_expansion_used = len(graph_results) > 0

        merged_results = self._merge_results(base_results, graph_results)
        rerank_time = 0.0
        used_reranker = False

        if use_reranker and merged_results:
            try:
                search_results, rerank_time = self._rerank_results(question, merged_results, top_k=top_k)
                used_reranker = True
            except Exception as exc:
                logger.warning("[PIPELINE] Reranking failed, fallback to base ranking: %s", exc)
                search_results = merged_results[:top_k]
        else:
            search_results = merged_results[:top_k]

        retrieval_time = time.time() - start_retrieval
        logger.info(
            "[PIPELINE] Retrieved %s docs in %.2fs (graph_expansion=%s)",
            len(search_results),
            retrieval_time,
            graph_expansion_used,
        )

        context, citations = self._format_context(search_results)
        start_generation = time.time()
        if not context:
            answer = "Khong tim thay thong tin phap ly lien quan den cau hoi cua ban."
        else:
            answer = self.llm.generate(question, context)
        generation_time = time.time() - start_generation

        return LegalAnswer(
            answer=answer,
            citations=citations,
            intent=intent_result.intent.value,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            rerank_time=rerank_time,
            used_reranker=used_reranker,
            graph_expansion_used=graph_expansion_used,
            graph_mode=graph_mode if graph_expansion_used else "none",
        )

    def _format_context(self, results: List) -> Tuple[str, List[Dict]]:
        if not results:
            return "", []

        context_parts = []
        citations = []

        for index, result in enumerate(results):
            meta = result.metadata
            header_parts = [f"[{index + 1}]"]

            if meta.get("citation_full"):
                header_parts.append(meta["citation_full"])
            else:
                if meta.get("loai_van_ban"):
                    header_parts.append(f"[{meta['loai_van_ban']}]")
                if meta.get("ten_van_ban"):
                    header_parts.append(meta["ten_van_ban"])
                if meta.get("dieu_so"):
                    header_parts.append(f"Dieu {meta['dieu_so']}")
                if meta.get("khoan_so"):
                    header_parts.append(f"Khoan {meta['khoan_so']}")
                if meta.get("diem_so"):
                    header_parts.append(f"Diem {meta['diem_so']})")

            context_parts.append(f"{' | '.join(header_parts)}\n{result.content}")
            citations.append(
                {
                    "content": result.content,
                    "doc_id": meta.get("doc_id", meta.get("van_ban_id", "")),
                    "van_ban_id": meta.get("van_ban_id", ""),
                    "so_ky_hieu": meta.get("so_ky_hieu", ""),
                    "ten_van_ban": meta.get("ten_van_ban", ""),
                    "loai_van_ban": meta.get("loai_van_ban", ""),
                    "co_quan": meta.get("co_quan", meta.get("co_quan_ban_hanh", "")),
                    "law_family": meta.get("law_family", ""),
                    "chuong": meta.get("chuong", ""),
                    "ten_chuong": meta.get("ten_chuong", ""),
                    "dieu_so": meta.get("dieu_so", ""),
                    "tieu_de_dieu": meta.get("tieu_de_dieu", ""),
                    "khoan_so": meta.get("khoan_so", ""),
                    "diem_so": meta.get("diem_so", ""),
                    "chunk_type": meta.get("chunk_type", ""),
                    "segment_level": meta.get("segment_level", ""),
                    "legal_path": meta.get("legal_path", meta.get("path", "")),
                    "is_active": meta.get("is_active", True),
                    "citation_base": meta.get("citation_base", ""),
                    "citation_full": meta.get("citation_full", ""),
                    "token_len_est": meta.get("token_len_est", 0),
                    "encoder_profile": meta.get("encoder_profile", ""),
                    "score": result.score,
                }
            )

        return "\n\n---\n\n".join(context_parts), citations

    def get_stats(self) -> Dict:
        self.initialize()
        graph_enabled = bool(self.graph_store and getattr(self.graph_store, "enabled", False))
        return {
            "qdrant": self.qdrant_store.get_stats(),
            "embedding_model": self.embedding_model.model_name,
            "embedding_dim": self.embedding_model.embedding_dim,
            "llm_model": f"{self.llm.provider}/{self.llm.model}",
            "graph_enabled": graph_enabled,
        }


_pipeline: Optional[LegalRAGPipeline] = None
_pipeline_lock = None


def _get_pipeline_lock():
    global _pipeline_lock
    if _pipeline_lock is None:
        import threading

        _pipeline_lock = threading.Lock()
    return _pipeline_lock


def get_legal_rag_pipeline() -> LegalRAGPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    lock = _get_pipeline_lock()
    with lock:
        if _pipeline is None:
            _pipeline = LegalRAGPipeline()
    return _pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = LegalRAGPipeline()
    pipeline.initialize()
    print("Stats:", pipeline.get_stats())
