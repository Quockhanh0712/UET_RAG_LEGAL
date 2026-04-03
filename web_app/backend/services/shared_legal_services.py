"""Application services for shared legal search/chat paths."""

from __future__ import annotations

from typing import Any, Callable

from src.legal_rag_pipeline import LegalAnswer

try:
    from ..api.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse
except ImportError:
    from api.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse


class SharedLegalSearchService:
    """Application service that runs shared legal retrieval and builds API contract output."""

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
    ) -> None:
        self._normalize_query_mode = normalize_query_mode
        self._backend_runtime_settings = backend_runtime_settings
        self._get_legal_pipeline = get_legal_pipeline
        self._validate_qdrant_contract = validate_qdrant_contract
        self._build_query_rewrite_service = build_query_rewrite_service
        self._build_search_payload = build_search_payload
        self._get_model_profile = get_model_profile
        self._get_setting = get_setting

    def run(self, request: SearchRequest) -> SearchResponse:
        query_mode = self._normalize_query_mode(request.query_mode)
        settings = self._backend_runtime_settings(
            query_rewriter_enabled=(query_mode == "rewrite_combine"),
            query_rewriter_force=(query_mode == "rewrite_combine"),
        )
        settings.apply_env()
        pipeline = self._get_legal_pipeline(settings)
        contract = self._validate_qdrant_contract(settings, int(pipeline.embedding_model.embedding_dim))
        query_rewrite_service = self._build_query_rewrite_service(request.model_profile_id, query_mode)
        result = pipeline.search(
            question=request.query,
            top_k=request.top_k,
            filter_conditions=None,
            use_reranker=bool(
                request.use_reranker
                if request.use_reranker is not None
                else self._get_setting("reranker_enabled", True)
            ),
            include_inactive=not request.require_active_only,
            use_graph_context=False,
            graph_hops=1,
            graph_expand_docs=20,
            query_rewrite_service=query_rewrite_service,
        )
        payload = self._build_search_payload(
            result,
            mode=request.search_mode,
            collection=contract.collection_name,
            model_profile_id=self._get_model_profile(request.model_profile_id).id,
            query_mode=query_mode,
        )
        return SearchResponse(**payload)


class SharedLegalChatService:
    """Application service that runs shared legal chat and persists shared-path history."""

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
    ) -> None:
        self._normalize_query_mode = normalize_query_mode
        self._backend_runtime_settings = backend_runtime_settings
        self._get_legal_pipeline = get_legal_pipeline
        self._validate_qdrant_contract = validate_qdrant_contract
        self._build_query_rewrite_service = build_query_rewrite_service
        self._build_remote_llm = build_remote_llm
        self._build_chat_payload = build_chat_payload
        self._get_model_profile = get_model_profile
        self._get_setting = get_setting
        self._save_chat_history = save_chat_history

    def run(self, request: ChatRequest) -> ChatResponse:
        query_mode = self._normalize_query_mode(request.query_mode)
        settings = self._backend_runtime_settings(
            query_rewriter_enabled=(query_mode == "rewrite_combine"),
            query_rewriter_force=(query_mode == "rewrite_combine"),
        )
        settings.apply_env()
        pipeline = self._get_legal_pipeline(settings)
        contract = self._validate_qdrant_contract(settings, int(pipeline.embedding_model.embedding_dim))
        query_rewrite_service = self._build_query_rewrite_service(request.model_profile_id, query_mode)
        llm_client = self._build_remote_llm(request.model_profile_id)

        fallback_used = False
        fallback_error = ""
        try:
            result = pipeline.query(
                question=request.message,
                top_k=request.top_k,
                filter_conditions=None,
                use_reranker=bool(
                    request.use_reranker
                    if request.use_reranker is not None
                    else self._get_setting("reranker_enabled", True)
                ),
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
                use_reranker=bool(
                    request.use_reranker
                    if request.use_reranker is not None
                    else self._get_setting("reranker_enabled", True)
                ),
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
            answer = (
                "LLM unavailable or failed. Returning retrieval-only sources. "
                "Please inspect citations below."
            )

        payload = self._build_chat_payload(
            result,
            answer=answer,
            collection=contract.collection_name,
            model_profile_id=self._get_model_profile(request.model_profile_id).id,
            query_mode=query_mode,
            fallback_used=fallback_used,
            fallback_error=fallback_error,
        )
        self._save_chat_history(
            request=request,
            answer=answer,
            sources=payload["sources"],
            payload=payload,
        )
        return ChatResponse(**payload)
