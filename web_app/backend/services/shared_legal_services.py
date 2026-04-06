"""Application services for shared runtime search/chat paths."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Dict, List

from src.legal_rag.contracts.results import LegalAnswer, LegalSearchResult
from src.legal_rag.pipeline.context_builder import build_rewrite_context_block, format_context
from src.qdrant_store import SearchResult

try:
    from ..api.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse
except ImportError:
    from api.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse


def _intent_value(intent_result: Any) -> str:
    intent = getattr(intent_result, "intent", intent_result)
    return str(getattr(intent, "value", intent) or "factual")


def _dedupe_queries(original_query: str, rewritten_queries: List[str], limit: int = 3) -> List[str]:
    original = str(original_query or "").strip().lower()
    compact: List[str] = []
    seen = set()
    for query in list(rewritten_queries or []):
        value = str(query or "").strip()
        if not value:
            continue
        lowered = value.lower()
        if lowered == original or lowered in seen:
            continue
        seen.add(lowered)
        compact.append(value)
        if len(compact) >= max(1, int(limit or 3)):
            break
    return compact


class _SharedRuntimeBase:
    def __init__(
        self,
        *,
        normalize_query_mode: Callable[[str | None], str],
        backend_runtime_settings: Callable[[bool, bool], Any],
        get_legal_pipeline: Callable[[Any], Any],
        validate_qdrant_contract: Callable[[Any, int], Any],
        build_query_rewrite_service: Callable[[str | None, str], Any],
        get_model_profile: Callable[[str | None], Any],
        get_setting: Callable[[str, Any], Any],
        get_user_store: Callable[[], Any] | None = None,
    ) -> None:
        self._normalize_query_mode = normalize_query_mode
        self._backend_runtime_settings = backend_runtime_settings
        self._get_legal_pipeline = get_legal_pipeline
        self._validate_qdrant_contract = validate_qdrant_contract
        self._build_query_rewrite_service = build_query_rewrite_service
        self._get_model_profile = get_model_profile
        self._get_setting = get_setting
        self._get_user_store = get_user_store

    @staticmethod
    def _normalize_mode(mode: str | None) -> str:
        value = str(mode or "legal").strip().lower()
        return value if value in {"legal", "user", "hybrid"} else "legal"

    def _use_reranker(self, request: SearchRequest | ChatRequest) -> bool:
        return bool(
            request.use_reranker
            if request.use_reranker is not None
            else self._get_setting("reranker_enabled", True)
        )

    def _prepare_runtime(self, request: SearchRequest | ChatRequest):
        mode = self._normalize_mode(getattr(request, "search_mode", "legal"))
        query_mode = self._normalize_query_mode(getattr(request, "query_mode", "original"))
        settings = self._backend_runtime_settings(
            query_rewriter_enabled=(query_mode == "rewrite_combine"),
            query_rewriter_force=(query_mode == "rewrite_combine"),
        )
        settings.apply_env()
        pipeline = self._get_legal_pipeline(settings)
        contract = None
        if mode in {"legal", "hybrid"}:
            contract = self._validate_qdrant_contract(settings, int(pipeline.embedding_model.embedding_dim))
        query_rewrite_service = self._build_query_rewrite_service(request.model_profile_id, query_mode)
        model_profile = self._get_model_profile(request.model_profile_id)
        return mode, query_mode, pipeline, contract, query_rewrite_service, model_profile

    def _build_meta(
        self,
        *,
        question: str,
        query_mode: str,
        rewrite_result: Any,
        intent_result: Any,
        retrieval_time: float,
        rerank_time: float,
        used_reranker: bool,
        per_subquery_hits: List[Dict[str, Any]],
        retrieval_union_count: int,
        graph_expansion_used: bool = False,
        graph_mode: str = "none",
        pipeline: Any = None,
    ) -> Dict[str, Any]:
        rewritten_queries = list(getattr(rewrite_result, "queries", []) or [question])
        compact_fn = getattr(pipeline, "_compact_rewrite_queries", None)
        rewrite_queries_compact = (
            compact_fn(original_query=question, rewritten_queries=rewritten_queries)
            if callable(compact_fn)
            else _dedupe_queries(question, rewritten_queries)
        )
        retrieval_query_set = list(getattr(rewrite_result, "retrieval_query_set", []) or rewritten_queries)
        return {
            "intent": _intent_value(intent_result),
            "retrieval_time": float(retrieval_time),
            "rerank_time": float(rerank_time),
            "used_reranker": bool(used_reranker),
            "graph_expansion_used": bool(graph_expansion_used),
            "graph_mode": str(graph_mode if graph_expansion_used else "none"),
            "rewrite_time": float(getattr(rewrite_result, "latency_ms", 0.0) or 0.0) / 1000.0,
            "rewrite_used": bool(getattr(rewrite_result, "rewrite_used", False)),
            "rewrite_status": str(getattr(rewrite_result, "status", "disabled") or "disabled"),
            "rewrite_fallback_reason": str(getattr(rewrite_result, "fallback_reason", "") or ""),
            "rewritten_queries": rewritten_queries,
            "missing_numbers": list(getattr(rewrite_result, "missing_numbers", []) or []),
            "per_subquery_hits": list(per_subquery_hits or []),
            "retrieval_union_count": int(retrieval_union_count or 0),
            "rewrite_mode": str(query_mode or "original"),
            "rewrite_original_query": str(question or ""),
            "rewrite_strategy": str(
                getattr(rewrite_result, "rewrite_strategy", "structured_synthesis")
                or "structured_synthesis"
            ),
            "rewrite_quality_passed": bool(getattr(rewrite_result, "quality_passed", False)),
            "retrieval_query_set": retrieval_query_set,
            "retrieval_query_plan": list(getattr(rewrite_result, "retrieval_query_plan", []) or []),
            "query_facets": dict(getattr(rewrite_result, "query_facets", {}) or {}),
            "rewrite_debug": dict(getattr(rewrite_result, "debug_payload", {}) or {}),
            "rewrite_queries_compact": rewrite_queries_compact,
        }

    @staticmethod
    def _build_search_contract(search_results: List[SearchResult], meta: Dict[str, Any]) -> LegalSearchResult:
        _, citations = format_context(search_results)
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
            rewritten_queries=list(meta["rewritten_queries"]),
            missing_numbers=list(meta["missing_numbers"]),
            per_subquery_hits=list(meta["per_subquery_hits"]),
            retrieval_union_count=int(meta["retrieval_union_count"]),
            rewrite_mode=str(meta["rewrite_mode"]),
            rewrite_original_query=str(meta["rewrite_original_query"]),
            rewrite_queries_compact=list(meta["rewrite_queries_compact"]),
            rewrite_strategy=str(meta["rewrite_strategy"]),
            rewrite_quality_passed=bool(meta["rewrite_quality_passed"]),
            retrieval_query_set=list(meta["retrieval_query_set"]),
            retrieval_query_plan=list(meta["retrieval_query_plan"]),
            query_facets=dict(meta["query_facets"]),
            rewrite_debug=dict(meta["rewrite_debug"]),
        )

    @staticmethod
    def _build_answer_contract(
        search_results: List[SearchResult],
        meta: Dict[str, Any],
        *,
        answer: str,
        generation_time: float,
    ) -> LegalAnswer:
        _, citations = format_context(search_results)
        return LegalAnswer(
            answer=str(answer or ""),
            citations=citations,
            intent=str(meta["intent"]),
            retrieval_time=float(meta["retrieval_time"]),
            generation_time=float(generation_time),
            rerank_time=float(meta["rerank_time"]),
            used_reranker=bool(meta["used_reranker"]),
            graph_expansion_used=bool(meta["graph_expansion_used"]),
            graph_mode=str(meta["graph_mode"]),
            rewrite_time=float(meta["rewrite_time"]),
            rewrite_used=bool(meta["rewrite_used"]),
            rewrite_status=str(meta["rewrite_status"]),
            rewrite_fallback_reason=str(meta["rewrite_fallback_reason"]),
            rewritten_queries=list(meta["rewritten_queries"]),
            missing_numbers=list(meta["missing_numbers"]),
            per_subquery_hits=list(meta["per_subquery_hits"]),
            retrieval_union_count=int(meta["retrieval_union_count"]),
            rewrite_mode=str(meta["rewrite_mode"]),
            rewrite_original_query=str(meta["rewrite_original_query"]),
            rewrite_queries_compact=list(meta["rewrite_queries_compact"]),
            rewrite_strategy=str(meta["rewrite_strategy"]),
            rewrite_quality_passed=bool(meta["rewrite_quality_passed"]),
            retrieval_query_set=list(meta["retrieval_query_set"]),
            retrieval_query_plan=list(meta["retrieval_query_plan"]),
            query_facets=dict(meta["query_facets"]),
            rewrite_debug=dict(meta["rewrite_debug"]),
        )

    def _get_user_store_instance(self):
        if self._get_user_store is None:
            raise RuntimeError("User document store is not configured for shared runtime.")
        return self._get_user_store()

    @staticmethod
    def _user_collection_name(user_store: Any) -> str:
        indexer = getattr(user_store, "_indexer", None)
        if indexer is not None:
            try:
                indexer.ensure_collection()
                active = indexer._active_collection()
                if active:
                    return str(active)
            except Exception:
                pass
        return str(getattr(user_store, "COLLECTION_NAME", "user_docs_private"))

    @staticmethod
    def _normalize_user_hit(hit: Dict[str, Any]) -> SearchResult:
        payload = dict(hit.get("payload", {}) or {})
        content = str(hit.get("content") or payload.get("content") or "")
        doc_id = str(hit.get("doc_id") or payload.get("doc_id") or "")
        filename = str(hit.get("filename") or payload.get("filename") or "")
        page_number = hit.get("page", payload.get("page_number"))
        payload.setdefault("content", content)
        payload.setdefault("doc_id", doc_id)
        payload.setdefault("filename", filename)
        if page_number not in (None, ""):
            payload.setdefault("page_number", page_number)
        payload["source_type"] = "user"
        chunk_id = str(
            payload.get("chunk_id")
            or hit.get("chunk_id")
            or f"user::{doc_id or filename}::{page_number or 'na'}::{hashlib.md5(content.encode('utf-8')).hexdigest()[:12]}"
        )
        payload.setdefault("chunk_id", chunk_id)
        return SearchResult(
            chunk_id=chunk_id,
            content=content,
            score=float(hit.get("score", 0.0) or 0.0),
            metadata=payload,
        )

    def _retrieve_user_results(
        self,
        *,
        question: str,
        retrieval_queries: List[str],
        user_id: str,
        session_id: str | None,
        top_k: int,
        use_reranker: bool,
        user_store: Any,
        pipeline: Any,
    ) -> tuple[List[SearchResult], List[Dict[str, Any]], float]:
        retrieval_pool_k = 100 if use_reranker else max(top_k * 2, top_k)
        rankings: List[List[SearchResult]] = []
        per_subquery_hits: List[Dict[str, Any]] = []
        seen_queries = set()

        for query in list(retrieval_queries or [question]):
            query_text = str(query or "").strip()
            if not query_text:
                continue
            lowered = query_text.lower()
            if lowered in seen_queries:
                continue
            seen_queries.add(lowered)
            hits = user_store.search(
                query=query_text,
                user_id=user_id,
                session_id=session_id,
                top_k=retrieval_pool_k,
                use_hybrid=True,
            )
            ranking = [self._normalize_user_hit(hit) for hit in list(hits or [])]
            per_subquery_hits.append({"query": query_text, "hits": len(ranking)})
            if ranking:
                rankings.append(ranking)

        if not rankings:
            return [], per_subquery_hits, 0.0

        merged_results = pipeline._merge_query_rankings_rrf(rankings, limit=retrieval_pool_k)
        return merged_results, per_subquery_hits, float(len(merged_results))

    @staticmethod
    def _combine_per_subquery_hits(
        legal_hits: List[Dict[str, Any]],
        user_hits: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        combined: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []

        def _ensure(query: str) -> Dict[str, Any]:
            if query not in combined:
                combined[query] = {"query": query, "hits": 0, "legal_hits": 0, "user_hits": 0}
                order.append(query)
            return combined[query]

        for item in list(legal_hits or []):
            query = str(item.get("query") or "").strip()
            if not query:
                continue
            entry = _ensure(query)
            entry["legal_hits"] = int(item.get("hits", 0) or 0)

        for item in list(user_hits or []):
            query = str(item.get("query") or "").strip()
            if not query:
                continue
            entry = _ensure(query)
            entry["user_hits"] = int(item.get("hits", 0) or 0)

        results: List[Dict[str, Any]] = []
        for query in order:
            entry = combined[query]
            entry["hits"] = int(entry["legal_hits"]) + int(entry["user_hits"])
            results.append(entry)
        return results

    @staticmethod
    def _empty_chat_message(mode: str, doc_count: int) -> str:
        if mode == "user":
            if doc_count == 0:
                return (
                    "❌ Bạn chưa tải lên tài liệu nào. "
                    "Vui lòng tải lên tài liệu trước khi tìm kiếm trong 'Tài liệu của tôi'."
                )
            return (
                f"❌ Không tìm thấy nội dung phù hợp trong {doc_count} tài liệu của bạn.\n\n"
                "💡 Gợi ý:\n"
                "• Chuyển sang chế độ 'Kết hợp' (tìm cả tài liệu của bạn và văn bản luật)\n"
                "• Chuyển sang 'Văn bản pháp luật' (tìm trong kho luật)\n"
                "• Đặt câu hỏi khác liên quan đến nội dung tài liệu đã tải lên"
            )
        if mode == "hybrid":
            return (
                f"❌ Không tìm thấy tài liệu phù hợp (đã tìm trong {doc_count} tài liệu riêng của bạn và kho luật).\n\n"
                "💡 Gợi ý:\n"
                "• Đặt câu hỏi rõ ràng hơn với từ khóa cụ thể\n"
                "• Kiểm tra chính tả và ngữ pháp\n"
                "• Tải lên thêm tài liệu liên quan nếu cần"
            )
        return (
            "❌ Không tìm thấy văn bản pháp luật phù hợp.\n\n"
            "💡 Gợi ý:\n"
            "• Thử câu hỏi khác với từ khóa rõ ràng hơn\n"
            "• Sử dụng thuật ngữ pháp lý chính xác\n"
            "• Tải lên tài liệu riêng và chuyển sang chế độ 'Tài liệu của tôi'"
        )


class SharedLegalSearchService(_SharedRuntimeBase):
    """Application service that runs shared retrieval and builds API contract output."""

    def __init__(
        self,
        *,
        normalize_query_mode: Callable[[str | None], str],
        backend_runtime_settings: Callable[[bool, bool], Any],
        get_legal_pipeline: Callable[[Any], Any],
        validate_qdrant_contract: Callable[[Any, int], Any],
        build_query_rewrite_service: Callable[[str | None, str], Any],
        build_search_payload: Callable[..., dict[str, Any]],
        get_model_profile: Callable[[str | None], Any],
        get_setting: Callable[[str, Any], Any],
        get_user_store: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(
            normalize_query_mode=normalize_query_mode,
            backend_runtime_settings=backend_runtime_settings,
            get_legal_pipeline=get_legal_pipeline,
            validate_qdrant_contract=validate_qdrant_contract,
            build_query_rewrite_service=build_query_rewrite_service,
            get_model_profile=get_model_profile,
            get_setting=get_setting,
            get_user_store=get_user_store,
        )
        self._build_search_payload = build_search_payload

    def run(self, request: SearchRequest) -> SearchResponse:
        mode, query_mode, pipeline, contract, query_rewrite_service, model_profile = self._prepare_runtime(request)
        use_reranker = self._use_reranker(request)

        if mode == "legal":
            result = pipeline.search(
                question=request.query,
                top_k=request.top_k,
                filter_conditions=None,
                use_reranker=use_reranker,
                include_inactive=not request.require_active_only,
                use_graph_context=False,
                graph_hops=1,
                graph_expand_docs=20,
                query_rewrite_service=query_rewrite_service,
            )
            payload = self._build_search_payload(
                result,
                mode=mode,
                collection=contract.collection_name,
                model_profile_id=model_profile.id,
                query_mode=query_mode,
            )
            return SearchResponse(**payload)

        user_store = self._get_user_store_instance()
        user_collection = self._user_collection_name(user_store)

        if mode == "user":
            pipeline.initialize()
            rewrite_result = query_rewrite_service.rewrite(request.query)
            intent_result = pipeline.intent_classifier.classify(request.query)
            retrieval_queries = list(
                getattr(rewrite_result, "retrieval_query_set", [])
                or getattr(rewrite_result, "queries", [])
                or [request.query]
            )
            start_retrieval = time.time()
            search_results, user_hits, union_count = self._retrieve_user_results(
                question=request.query,
                retrieval_queries=retrieval_queries,
                user_id=request.user_id,
                session_id=request.session_id,
                top_k=request.top_k,
                use_reranker=use_reranker,
                user_store=user_store,
                pipeline=pipeline,
            )
            rerank_time = 0.0
            used_reranker = False
            if use_reranker and search_results:
                search_results, rerank_time = pipeline._rerank_results(
                    request.query,
                    search_results,
                    top_k=request.top_k,
                )
                used_reranker = True
            else:
                search_results = search_results[: request.top_k]
            meta = self._build_meta(
                question=request.query,
                query_mode=query_mode,
                rewrite_result=rewrite_result,
                intent_result=intent_result,
                retrieval_time=time.time() - start_retrieval,
                rerank_time=rerank_time,
                used_reranker=used_reranker,
                per_subquery_hits=user_hits,
                retrieval_union_count=int(union_count),
                pipeline=pipeline,
            )
            result = self._build_search_contract(search_results, meta)
            payload = self._build_search_payload(
                result,
                mode=mode,
                collection=user_collection,
                model_profile_id=model_profile.id,
                query_mode=query_mode,
            )
            return SearchResponse(**payload)

        legal_results, legal_meta = pipeline._retrieve(
            question=request.query,
            top_k=request.top_k,
            filter_conditions=None,
            use_reranker=False,
            include_inactive=not request.require_active_only,
            use_graph_context=False,
            graph_hops=1,
            graph_expand_docs=20,
            query_rewrite_service=query_rewrite_service,
        )
        retrieval_queries = list(legal_meta.get("retrieval_query_set", []) or [request.query])
        user_start = time.time()
        user_results, user_hits, _ = self._retrieve_user_results(
            question=request.query,
            retrieval_queries=retrieval_queries,
            user_id=request.user_id,
            session_id=request.session_id,
            top_k=request.top_k,
            use_reranker=use_reranker,
            user_store=user_store,
            pipeline=pipeline,
        )
        merged_results = pipeline._merge_results(legal_results, user_results)
        rerank_time = 0.0
        used_reranker = False
        if use_reranker and merged_results:
            merged_results, rerank_time = pipeline._rerank_results(
                request.query,
                merged_results,
                top_k=request.top_k,
            )
            used_reranker = True
        else:
            merged_results = merged_results[: request.top_k]
        meta = dict(legal_meta)
        meta["retrieval_time"] = float(legal_meta.get("retrieval_time", 0.0) or 0.0) + (time.time() - user_start)
        meta["rerank_time"] = rerank_time
        meta["used_reranker"] = used_reranker
        meta["per_subquery_hits"] = self._combine_per_subquery_hits(
            list(legal_meta.get("per_subquery_hits", []) or []),
            user_hits,
        )
        meta["retrieval_union_count"] = len(pipeline._merge_results(legal_results, user_results))
        result = self._build_search_contract(merged_results, meta)
        payload = self._build_search_payload(
            result,
            mode=mode,
            collection=contract.collection_name,
            model_profile_id=model_profile.id,
            query_mode=query_mode,
        )
        return SearchResponse(**payload)


class SharedLegalChatService(_SharedRuntimeBase):
    """Application service that runs shared chat and persists shared-path history."""

    def __init__(
        self,
        *,
        normalize_query_mode: Callable[[str | None], str],
        backend_runtime_settings: Callable[[bool, bool], Any],
        get_legal_pipeline: Callable[[Any], Any],
        validate_qdrant_contract: Callable[[Any, int], Any],
        build_query_rewrite_service: Callable[[str | None, str], Any],
        build_remote_llm: Callable[[str | None], Any],
        build_chat_payload: Callable[..., dict[str, Any]],
        get_model_profile: Callable[[str | None], Any],
        get_setting: Callable[[str, Any], Any],
        save_chat_history: Callable[..., None],
        get_user_store: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(
            normalize_query_mode=normalize_query_mode,
            backend_runtime_settings=backend_runtime_settings,
            get_legal_pipeline=get_legal_pipeline,
            validate_qdrant_contract=validate_qdrant_contract,
            build_query_rewrite_service=build_query_rewrite_service,
            get_model_profile=get_model_profile,
            get_setting=get_setting,
            get_user_store=get_user_store,
        )
        self._build_remote_llm = build_remote_llm
        self._build_chat_payload = build_chat_payload
        self._save_chat_history = save_chat_history

    def _persist_and_return(
        self,
        *,
        request: ChatRequest,
        payload: Dict[str, Any],
        answer: str,
    ) -> ChatResponse:
        self._save_chat_history(
            request=request,
            answer=answer,
            sources=payload["sources"],
            payload=payload,
        )
        return ChatResponse(**payload)

    def _build_structured_context(self, search_results: List[SearchResult], meta: Dict[str, Any]) -> Dict[str, Any]:
        context, _ = format_context(search_results)
        query_context = build_rewrite_context_block(
            original_query=str(meta["rewrite_original_query"]),
            rewrite_mode=str(meta["rewrite_mode"]),
            rewrite_status=str(meta["rewrite_status"]),
            rewrite_quality_passed=bool(meta["rewrite_quality_passed"]),
            rewrite_queries_compact=list(meta["rewrite_queries_compact"]),
            rewrite_fallback_reason=str(meta["rewrite_fallback_reason"]),
            query_facets=dict(meta["query_facets"]),
            rewrite_debug=dict(meta["rewrite_debug"]),
        )
        return {"query_context": query_context, "evidence_context": context}

    def _generate_answer(self, llm_client: Any, question: str, context: Dict[str, Any]) -> tuple[str, float, bool, str]:
        start = time.time()
        fallback_used = False
        fallback_error = ""
        try:
            answer = llm_client.generate(question, context)
        except Exception as exc:
            fallback_used = True
            fallback_error = str(exc)
            answer = "LLM unavailable or failed. Returning retrieval-only sources. Please inspect citations below."
        generation_time = time.time() - start
        if not answer:
            fallback_used = True
            fallback_error = fallback_error or "answer_generation_failed"
            answer = "LLM unavailable or failed. Returning retrieval-only sources. Please inspect citations below."
        return answer, generation_time, fallback_used, fallback_error

    def run(self, request: ChatRequest) -> ChatResponse:
        mode, query_mode, pipeline, contract, query_rewrite_service, model_profile = self._prepare_runtime(request)
        use_reranker = self._use_reranker(request)
        llm_client = self._build_remote_llm(request.model_profile_id)

        if mode == "legal":
            fallback_used = False
            fallback_error = ""
            try:
                result = pipeline.query(
                    question=request.message,
                    top_k=request.top_k,
                    filter_conditions=None,
                    use_reranker=use_reranker,
                    include_inactive=not request.require_active_only,
                    use_graph_context=bool(request.use_graph),
                    graph_hops=1,
                    graph_expand_docs=20,
                    query_rewrite_service=query_rewrite_service,
                    llm_client=llm_client,
                )
                answer = str(result.answer or "").strip()
            except Exception as exc:
                fallback_used = True
                fallback_error = str(exc)
                search_result = pipeline.search(
                    question=request.message,
                    top_k=request.top_k,
                    filter_conditions=None,
                    use_reranker=use_reranker,
                    include_inactive=not request.require_active_only,
                    use_graph_context=bool(request.use_graph),
                    graph_hops=1,
                    graph_expand_docs=20,
                    query_rewrite_service=query_rewrite_service,
                )
                result = LegalAnswer(
                    answer="",
                    citations=list(search_result.citations or []),
                    intent=str(search_result.intent),
                    retrieval_time=float(search_result.retrieval_time),
                    generation_time=0.0,
                    rerank_time=float(search_result.rerank_time),
                    used_reranker=bool(search_result.used_reranker),
                    graph_expansion_used=bool(search_result.graph_expansion_used),
                    graph_mode=str(search_result.graph_mode),
                    rewrite_time=float(search_result.rewrite_time),
                    rewrite_used=bool(search_result.rewrite_used),
                    rewrite_status=str(search_result.rewrite_status),
                    rewrite_fallback_reason=str(search_result.rewrite_fallback_reason),
                    rewrite_strategy=str(search_result.rewrite_strategy),
                    rewrite_quality_passed=bool(search_result.rewrite_quality_passed),
                    rewritten_queries=list(search_result.rewritten_queries),
                    rewrite_mode=str(search_result.rewrite_mode),
                    rewrite_original_query=str(search_result.rewrite_original_query),
                    rewrite_queries_compact=list(search_result.rewrite_queries_compact),
                    retrieval_query_set=list(search_result.retrieval_query_set),
                    retrieval_query_plan=list(search_result.retrieval_query_plan),
                    query_facets=dict(search_result.query_facets),
                    rewrite_debug=dict(search_result.rewrite_debug),
                    missing_numbers=list(search_result.missing_numbers),
                    per_subquery_hits=list(search_result.per_subquery_hits),
                    retrieval_union_count=int(search_result.retrieval_union_count),
                )
                answer = ""
            if not answer or answer.lower().startswith("loi khi goi llm") or answer.lower().startswith("llm khong kha dung"):
                fallback_used = True
                fallback_error = fallback_error or answer or "answer_generation_failed"
                answer = "LLM unavailable or failed. Returning retrieval-only sources. Please inspect citations below."
            payload = self._build_chat_payload(
                result,
                answer=answer,
                collection=contract.collection_name,
                model_profile_id=model_profile.id,
                query_mode=query_mode,
                fallback_used=fallback_used,
                fallback_error=fallback_error,
            )
            return self._persist_and_return(request=request, payload=payload, answer=answer)

        user_store = self._get_user_store_instance()
        user_collection = self._user_collection_name(user_store)

        if mode == "user":
            pipeline.initialize()
            rewrite_result = query_rewrite_service.rewrite(request.message)
            intent_result = pipeline.intent_classifier.classify(request.message)
            retrieval_queries = list(
                getattr(rewrite_result, "retrieval_query_set", [])
                or getattr(rewrite_result, "queries", [])
                or [request.message]
            )
            start_retrieval = time.time()
            search_results, user_hits, union_count = self._retrieve_user_results(
                question=request.message,
                retrieval_queries=retrieval_queries,
                user_id=request.user_id,
                session_id=request.session_id,
                top_k=request.top_k,
                use_reranker=use_reranker,
                user_store=user_store,
                pipeline=pipeline,
            )
            rerank_time = 0.0
            used_reranker = False
            if use_reranker and search_results:
                search_results, rerank_time = pipeline._rerank_results(
                    request.message,
                    search_results,
                    top_k=request.top_k,
                )
                used_reranker = True
            else:
                search_results = search_results[: request.top_k]
            meta = self._build_meta(
                question=request.message,
                query_mode=query_mode,
                rewrite_result=rewrite_result,
                intent_result=intent_result,
                retrieval_time=time.time() - start_retrieval,
                rerank_time=rerank_time,
                used_reranker=used_reranker,
                per_subquery_hits=user_hits,
                retrieval_union_count=int(union_count),
                pipeline=pipeline,
            )
            if not search_results:
                doc_count = len(user_store.get_user_documents(request.user_id))
                result = self._build_answer_contract(
                    search_results,
                    meta,
                    answer=self._empty_chat_message(mode, doc_count),
                    generation_time=0.0,
                )
                payload = self._build_chat_payload(
                    result,
                    answer=result.answer,
                    collection=user_collection,
                    model_profile_id=model_profile.id,
                    query_mode=query_mode,
                    fallback_used=False,
                    fallback_error="",
                )
                return self._persist_and_return(request=request, payload=payload, answer=result.answer)
            answer, generation_time, fallback_used, fallback_error = self._generate_answer(
                llm_client,
                request.message,
                self._build_structured_context(search_results, meta),
            )
            result = self._build_answer_contract(
                search_results,
                meta,
                answer=answer,
                generation_time=generation_time,
            )
            payload = self._build_chat_payload(
                result,
                answer=answer,
                collection=user_collection,
                model_profile_id=model_profile.id,
                query_mode=query_mode,
                fallback_used=fallback_used,
                fallback_error=fallback_error,
            )
            return self._persist_and_return(request=request, payload=payload, answer=answer)

        legal_results, legal_meta = pipeline._retrieve(
            question=request.message,
            top_k=request.top_k,
            filter_conditions=None,
            use_reranker=False,
            include_inactive=not request.require_active_only,
            use_graph_context=bool(request.use_graph),
            graph_hops=1,
            graph_expand_docs=20,
            query_rewrite_service=query_rewrite_service,
        )
        retrieval_queries = list(legal_meta.get("retrieval_query_set", []) or [request.message])
        user_start = time.time()
        user_results, user_hits, _ = self._retrieve_user_results(
            question=request.message,
            retrieval_queries=retrieval_queries,
            user_id=request.user_id,
            session_id=request.session_id,
            top_k=request.top_k,
            use_reranker=use_reranker,
            user_store=user_store,
            pipeline=pipeline,
        )
        combined_results = pipeline._merge_results(legal_results, user_results)
        rerank_time = 0.0
        used_reranker = False
        if use_reranker and combined_results:
            combined_results, rerank_time = pipeline._rerank_results(
                request.message,
                combined_results,
                top_k=request.top_k,
            )
            used_reranker = True
        else:
            combined_results = combined_results[: request.top_k]
        meta = dict(legal_meta)
        meta["retrieval_time"] = float(legal_meta.get("retrieval_time", 0.0) or 0.0) + (time.time() - user_start)
        meta["rerank_time"] = rerank_time
        meta["used_reranker"] = used_reranker
        meta["per_subquery_hits"] = self._combine_per_subquery_hits(
            list(legal_meta.get("per_subquery_hits", []) or []),
            user_hits,
        )
        meta["retrieval_union_count"] = len(pipeline._merge_results(legal_results, user_results))
        if not combined_results:
            doc_count = len(user_store.get_user_documents(request.user_id))
            result = self._build_answer_contract(
                combined_results,
                meta,
                answer=self._empty_chat_message(mode, doc_count),
                generation_time=0.0,
            )
            payload = self._build_chat_payload(
                result,
                answer=result.answer,
                collection=contract.collection_name,
                model_profile_id=model_profile.id,
                query_mode=query_mode,
                fallback_used=False,
                fallback_error="",
            )
            return self._persist_and_return(request=request, payload=payload, answer=result.answer)
        answer, generation_time, fallback_used, fallback_error = self._generate_answer(
            llm_client,
            request.message,
            self._build_structured_context(combined_results, meta),
        )
        result = self._build_answer_contract(
            combined_results,
            meta,
            answer=answer,
            generation_time=generation_time,
        )
        payload = self._build_chat_payload(
            result,
            answer=answer,
            collection=contract.collection_name,
            model_profile_id=model_profile.id,
            query_mode=query_mode,
            fallback_used=fallback_used,
            fallback_error=fallback_error,
        )
        return self._persist_and_return(request=request, payload=payload, answer=answer)
