"""Legal RAG pipeline orchestrator (hybrid retrieval + optional graph expansion)."""

from __future__ import annotations

import logging
import os
import time
from os import PathLike
from typing import Dict, List, Optional, Tuple

from src.embeddings import DEFAULT_MODEL
from src.legal_rag.answering.llm_gateway import LegalLLM
from src.legal_rag.contracts.results import LegalAnswer, LegalSearchResult
from src.legal_rag.embeddings.encoder import LegalEmbedding
from src.legal_rag.pipeline.context_builder import build_rewrite_context_block, format_context
from src.qdrant_store import COLLECTION_NAME, collection_name_for_model

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class LegalRAGPipeline:
    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        normalize_embeddings: Optional[bool] = None,
        query_rewrite_service=None,
    ):
        self.qdrant_store = None
        self.graph_store = None
        self.intent_classifier = None
        self.embedding_model = None
        self.llm = None
        self._reranker = None
        self._query_rewriter = None
        self.query_rewrite_service = query_rewrite_service
        self._initialized = False
        self.embedding_model_name = embedding_model_name or DEFAULT_MODEL
        self.normalize_embeddings = normalize_embeddings
        self.rerank_alpha = max(0.0, min(1.0, float(os.getenv("RERANKER_ALPHA", "1.0"))))
        self.query_rewriter_enabled = _env_bool("QUERY_REWRITER_ENABLED", False)
        self.query_rewriter_force = _env_bool("QUERY_REWRITER_FORCE", False)
        self.rewrite_queries_compact_max = max(1, int(os.getenv("REWRITE_QUERIES_COMPACT_MAX", "3")))

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

    def _get_query_rewriter(self):
        if self._query_rewriter is not None:
            return self._query_rewriter
        from src.legal_query_rewriter import LegalQueryRewriter

        self._query_rewriter = LegalQueryRewriter()
        return self._query_rewriter

    def _rewrite_query(self, question: str, query_rewrite_service=None):
        from src.legal_query_rewriter import QueryRewriteResult

        if query_rewrite_service is not None:
            return query_rewrite_service.rewrite(question)
        if self.query_rewrite_service is not None:
            return self.query_rewrite_service.rewrite(question)
        if not self.query_rewriter_enabled:
            return QueryRewriteResult(
                original_query=question,
                queries=[question],
                status="disabled",
            )
        try:
            rewriter = self._get_query_rewriter()
            return rewriter.rewrite_query(question, force=self.query_rewriter_force)
        except Exception as exc:
            logger.warning("[PIPELINE] Query rewrite failed, fallback to original query: %s", exc)
            return QueryRewriteResult(
                original_query=question,
                queries=[question],
                status="api_error",
                fallback_reason=str(exc),
            )

    def configure_runtime(
        self,
        collection_name: Optional[str] = None,
        bm25_path: Optional[PathLike[str] | str] = None,
    ):
        self.initialize()
        self.qdrant_store.dense_dim = int(self.embedding_model.embedding_dim)
        if collection_name:
            self.qdrant_store.collection_name = str(collection_name)
        if bm25_path:
            self.qdrant_store.set_bm25_path(str(bm25_path), load_if_exists=True)
        return self

    def _hybrid_search_safe(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
        filter_conditions: Dict[str, object],
    ) -> List:
        try:
            results = self.qdrant_store.hybrid_search(
                query=query_text,
                query_embedding=query_embedding,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                filter_conditions=filter_conditions,
            )
        except Exception as exc:
            if filter_conditions.get("is_active") is True:
                logger.warning(
                    "[PIPELINE] is_active filter failed (likely strict index requirement), retry without it: %s",
                    exc,
                )
                fallback_filters = dict(filter_conditions)
                fallback_filters.pop("is_active", None)
                results = self.qdrant_store.hybrid_search(
                    query=query_text,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    filter_conditions=fallback_filters,
                )
            else:
                raise
        if not results and filter_conditions.get("is_active") is True:
            fallback_filters = dict(filter_conditions)
            fallback_filters.pop("is_active", None)
            results = self.qdrant_store.hybrid_search(
                query=query_text,
                query_embedding=query_embedding,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                filter_conditions=fallback_filters,
            )
        return results

    def _merge_query_rankings_rrf(self, rankings: List[List], limit: int) -> List:
        if not rankings:
            return []
        if len(rankings) == 1:
            return rankings[0][:limit]

        from src.qdrant_store import SearchResult

        fused: Dict[str, Dict[str, object]] = {}
        k_const = 60.0
        for ranking in rankings:
            for rank, result in enumerate(ranking, start=1):
                chunk_id = getattr(result, "chunk_id", "")
                if not chunk_id:
                    continue
                entry = fused.setdefault(
                    chunk_id,
                    {
                        "score": 0.0,
                        "content": result.content,
                        "metadata": result.metadata,
                        "best_score": float(getattr(result, "score", 0.0) or 0.0),
                    },
                )
                entry["score"] = float(entry["score"]) + (1.0 / (k_const + rank))
                current_best = float(entry.get("best_score", 0.0) or 0.0)
                result_score = float(getattr(result, "score", 0.0) or 0.0)
                if result_score >= current_best:
                    entry["best_score"] = result_score
                    entry["content"] = result.content
                    entry["metadata"] = result.metadata

        ranked = sorted(fused.items(), key=lambda item: float(item[1]["score"]), reverse=True)[:limit]
        return [
            SearchResult(
                chunk_id=chunk_id,
                content=str(data["content"]),
                score=float(data["score"]),
                metadata=dict(data["metadata"]),
            )
            for chunk_id, data in ranked
        ]

    def _rerank_results(self, question: str, results: List, top_k: int) -> Tuple[List, float]:
        if not results:
            return [], 0.0

        from src.qdrant_store import SearchResult
        from src.reranker import get_legal_reranker

        if self._reranker is None:
            self._reranker = get_legal_reranker(
                model_name=os.getenv("RERANKER_MODEL", "AITeamVN/Vietnamese_Reranker"),
                use_gpu=True,
                max_length=int(os.getenv("RERANKER_MAX_LENGTH", "2304")),
                batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "8")),
            )

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
            rrf_weight=self.rerank_alpha,
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

    def _retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_conditions: Optional[Dict] = None,
        use_reranker: bool = True,
        include_inactive: bool = False,
        use_graph_context: bool = True,
        graph_hops: int = 1,
        graph_expand_docs: int = 20,
        query_rewrite_service=None,
    ) -> Tuple[List, Dict[str, object]]:
        self.initialize()
        rewrite_result = self._rewrite_query(question, query_rewrite_service=query_rewrite_service)
        intent_result = self.intent_classifier.classify(question)
        if top_k is None:
            top_k = intent_result.top_k

        effective_filters: Dict[str, object] = dict(filter_conditions or {})
        if not include_inactive and "is_active" not in effective_filters:
            effective_filters["is_active"] = True

        start_retrieval = time.time()
        retrieval_pool_k = 100 if use_reranker else max(top_k * 2, top_k)
        search_queries = rewrite_result.queries or [question]
        rewrite_mode = "rewrite_combine" if bool(getattr(query_rewrite_service, "enabled", False)) else "original"
        per_subquery_hits: List[Dict[str, object]] = []
        subquery_rankings: List[List] = []

        for query_text in search_queries:
            query_embedding = self.embedding_model.encode_query(query_text)
            ranking = self._hybrid_search_safe(
                query_text=query_text,
                query_embedding=query_embedding,
                top_k=retrieval_pool_k,
                dense_weight=intent_result.dense_weight,
                sparse_weight=intent_result.sparse_weight,
                filter_conditions=effective_filters,
            )
            per_subquery_hits.append(
                {
                    "query": query_text,
                    "hits": len(ranking),
                }
            )
            subquery_rankings.append(ranking)

        base_results = self._merge_query_rankings_rrf(subquery_rankings, limit=retrieval_pool_k)
        retrieval_union_count = len(base_results)

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

        meta = {
            "intent": intent_result.intent.value,
            "retrieval_time": retrieval_time,
            "rerank_time": rerank_time,
            "used_reranker": used_reranker,
            "graph_expansion_used": graph_expansion_used,
            "graph_mode": graph_mode if graph_expansion_used else "none",
            "rewrite_time": float(getattr(rewrite_result, "latency_ms", 0.0)) / 1000.0,
            "rewrite_used": bool(getattr(rewrite_result, "rewrite_used", False)),
            "rewrite_status": str(getattr(rewrite_result, "status", "disabled")),
            "rewrite_fallback_reason": str(getattr(rewrite_result, "fallback_reason", "")),
            "rewritten_queries": list(getattr(rewrite_result, "queries", []) or [question]),
            "missing_numbers": list(getattr(rewrite_result, "missing_numbers", [])),
            "per_subquery_hits": per_subquery_hits,
            "retrieval_union_count": retrieval_union_count,
            "rewrite_mode": rewrite_mode,
            "rewrite_original_query": question,
            "rewrite_strategy": str(getattr(rewrite_result, "rewrite_strategy", "structured_synthesis")),
            "rewrite_quality_passed": bool(getattr(rewrite_result, "quality_passed", False)),
            "retrieval_query_set": list(getattr(rewrite_result, "retrieval_query_set", []) or search_queries),
            "retrieval_query_plan": list(getattr(rewrite_result, "retrieval_query_plan", []) or []),
            "query_facets": dict(getattr(rewrite_result, "query_facets", {}) or {}),
            "rewrite_debug": dict(getattr(rewrite_result, "debug_payload", {}) or {}),
            "rewrite_queries_compact": self._compact_rewrite_queries(
                original_query=question,
                rewritten_queries=list(getattr(rewrite_result, "queries", []) or [question]),
            ),
        }
        return search_results, meta

    def _compact_rewrite_queries(self, original_query: str, rewritten_queries: List[str]) -> List[str]:
        original = str(original_query or "").strip()
        original_lower = original.lower()
        compact: List[str] = []
        seen = set()
        for query in rewritten_queries:
            value = str(query or "").strip()
            if not value:
                continue
            value_lower = value.lower()
            if value_lower == original_lower:
                continue
            if value_lower in seen:
                continue
            seen.add(value_lower)
            compact.append(value)
            if len(compact) >= self.rewrite_queries_compact_max:
                break
        return compact

    def search(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_conditions: Optional[Dict] = None,
        use_reranker: bool = True,
        include_inactive: bool = False,
        use_graph_context: bool = True,
        graph_hops: int = 1,
        graph_expand_docs: int = 20,
        query_rewrite_service=None,
    ) -> LegalSearchResult:
        search_results, meta = self._retrieve(
            question=question,
            top_k=top_k,
            filter_conditions=filter_conditions,
            use_reranker=use_reranker,
            include_inactive=include_inactive,
            use_graph_context=use_graph_context,
            graph_hops=graph_hops,
            graph_expand_docs=graph_expand_docs,
            query_rewrite_service=query_rewrite_service,
        )
        _, citations = format_context(search_results)
        rewritten_queries = list(meta.get("rewritten_queries", []) or [])
        rewrite_mode = str(meta.get("rewrite_mode") or "original")
        rewrite_original_query = str(meta.get("rewrite_original_query") or question)
        rewrite_queries_compact = list(meta.get("rewrite_queries_compact", []) or [])
        rewrite_strategy = str(meta.get("rewrite_strategy") or "structured_synthesis")
        rewrite_quality_passed = bool(meta.get("rewrite_quality_passed") or False)
        retrieval_query_set = list(meta.get("retrieval_query_set", rewritten_queries) or rewritten_queries)
        retrieval_query_plan = list(meta.get("retrieval_query_plan", []) or [])
        query_facets = dict(meta.get("query_facets", {}) or {})
        rewrite_debug = dict(meta.get("rewrite_debug", {}) or {})
        return LegalSearchResult(
            citations=citations,
            intent=str(meta["intent"]),
            retrieval_time=float(meta["retrieval_time"]),
            rerank_time=float(meta["rerank_time"]),
            used_reranker=bool(meta["used_reranker"]),
            graph_expansion_used=bool(meta["graph_expansion_used"]),
            graph_mode=str(meta["graph_mode"]),
            rewrite_time=float(meta["rewrite_time"]),
            rewrite_used=bool(meta["rewrite_used"]),
            rewrite_status=str(meta["rewrite_status"]),
            rewrite_fallback_reason=str(meta["rewrite_fallback_reason"]),
            rewritten_queries=rewritten_queries,
            missing_numbers=list(meta["missing_numbers"]),
            per_subquery_hits=list(meta["per_subquery_hits"]),
            retrieval_union_count=int(meta["retrieval_union_count"]),
            rewrite_mode=rewrite_mode,
            rewrite_original_query=rewrite_original_query,
            rewrite_queries_compact=rewrite_queries_compact,
            rewrite_strategy=rewrite_strategy,
            rewrite_quality_passed=rewrite_quality_passed,
            retrieval_query_set=retrieval_query_set,
            retrieval_query_plan=retrieval_query_plan,
            query_facets=query_facets,
            rewrite_debug=rewrite_debug,
        )

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
        query_rewrite_service=None,
        llm_client=None,
    ) -> LegalAnswer:
        search_results, meta = self._retrieve(
            question=question,
            top_k=top_k,
            filter_conditions=filter_conditions,
            use_reranker=use_reranker,
            include_inactive=include_inactive,
            use_graph_context=use_graph_context,
            graph_hops=graph_hops,
            graph_expand_docs=graph_expand_docs,
            query_rewrite_service=query_rewrite_service,
        )
        rewritten_queries = list(meta.get("rewritten_queries", []) or [])
        rewrite_mode = str(meta.get("rewrite_mode") or "original")
        rewrite_original_query = str(meta.get("rewrite_original_query") or question)
        rewrite_queries_compact = list(meta.get("rewrite_queries_compact", []) or [])
        rewrite_strategy = str(meta.get("rewrite_strategy") or "structured_synthesis")
        rewrite_quality_passed = bool(meta.get("rewrite_quality_passed") or False)
        retrieval_query_set = list(meta.get("retrieval_query_set", rewritten_queries) or rewritten_queries)
        retrieval_query_plan = list(meta.get("retrieval_query_plan", []) or [])
        query_facets = dict(meta.get("query_facets", {}) or {})
        rewrite_debug = dict(meta.get("rewrite_debug", {}) or {})
        context, citations = format_context(search_results)
        query_context = build_rewrite_context_block(
            original_query=rewrite_original_query,
            rewrite_mode=rewrite_mode,
            rewrite_status=str(meta["rewrite_status"]),
            rewrite_quality_passed=rewrite_quality_passed,
            rewrite_queries_compact=rewrite_queries_compact,
            rewrite_fallback_reason=str(meta["rewrite_fallback_reason"]),
            query_facets=query_facets,
            rewrite_debug=rewrite_debug,
        )
        structured_context = {
            "query_context": query_context,
            "evidence_context": context,
        }
        start_generation = time.time()
        if not context:
            answer = "Khong tim thay thong tin phap ly lien quan den cau hoi cua ban."
        else:
            answer_provider = llm_client or self.llm
            answer = answer_provider.generate(question, structured_context)
        generation_time = time.time() - start_generation

        return LegalAnswer(
            answer=answer,
            citations=citations,
            intent=str(meta["intent"]),
            retrieval_time=float(meta["retrieval_time"]),
            generation_time=generation_time,
            rerank_time=float(meta["rerank_time"]),
            used_reranker=bool(meta["used_reranker"]),
            graph_expansion_used=bool(meta["graph_expansion_used"]),
            graph_mode=str(meta["graph_mode"]),
            rewrite_time=float(meta["rewrite_time"]),
            rewrite_used=bool(meta["rewrite_used"]),
            rewrite_status=str(meta["rewrite_status"]),
            rewrite_fallback_reason=str(meta["rewrite_fallback_reason"]),
            rewritten_queries=rewritten_queries,
            missing_numbers=list(meta["missing_numbers"]),
            per_subquery_hits=list(meta["per_subquery_hits"]),
            retrieval_union_count=int(meta["retrieval_union_count"]),
            rewrite_mode=rewrite_mode,
            rewrite_original_query=rewrite_original_query,
            rewrite_queries_compact=rewrite_queries_compact,
            rewrite_strategy=rewrite_strategy,
            rewrite_quality_passed=rewrite_quality_passed,
            retrieval_query_set=retrieval_query_set,
            retrieval_query_plan=retrieval_query_plan,
            query_facets=query_facets,
            rewrite_debug=rewrite_debug,
        )

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
