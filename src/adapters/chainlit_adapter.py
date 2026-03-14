from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ChatExecutionResult:
    answer_text: str
    citations: List[Dict[str, Any]]
    retrieval_time_ms: float
    generation_time_ms: float
    rerank_time_ms: float
    rewrite_time_ms: float
    used_reranker: bool
    rewrite_used: bool
    rewrite_status: str
    rewrite_fallback_reason: str
    rewritten_queries: List[str] = field(default_factory=list)
    missing_numbers: List[str] = field(default_factory=list)
    per_subquery_hits: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_union_count: int = 0
    fallback_used: bool = False
    fallback_error: str = ""
    status: str = "ok"

    def to_smoke_row(self, run_id: str, query: str, top_k: int) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "query": query,
            "status": self.status,
            "top_k": top_k,
            "results": list(self.citations),
            "retrieval_time_ms": self.retrieval_time_ms,
            "rewrite_time_ms": self.rewrite_time_ms,
            "rewrite_used": self.rewrite_used,
            "rewrite_status": self.rewrite_status,
            "rewrite_fallback_reason": self.rewrite_fallback_reason,
            "rewrite_query_count": len(self.rewritten_queries),
            "retrieval_union_count": self.retrieval_union_count,
            "rewritten_queries": list(self.rewritten_queries),
            "missing_numbers": list(self.missing_numbers),
            "per_subquery_hits": list(self.per_subquery_hits),
            "rerank_time_ms": self.rerank_time_ms,
            "used_reranker": self.used_reranker,
            "generation_time_ms": self.generation_time_ms,
            "error": self.fallback_error,
        }


def is_llm_failure(answer: str) -> bool:
    text = str(answer or "").strip().lower()
    if not text:
        return True
    markers = [
        "loi khi goi llm",
        "llm khong kha dung",
        "error calling ollama api",
        "failed",
    ]
    return any(marker in text for marker in markers)


def to_citation_items(results: List[Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for result in results:
        meta = result.metadata or {}
        items.append(
            {
                "doc_id": meta.get("doc_id", meta.get("van_ban_id", "")),
                "citation_full": meta.get("citation_full", ""),
                "legal_path": meta.get("legal_path", meta.get("path", "")),
                "chunk_type": meta.get("chunk_type", ""),
                "score": float(getattr(result, "score", 0.0)),
                "is_active": bool(meta.get("is_active", True)),
            }
        )
    return items


def retrieve_only(pipeline, query: str, top_k: int, require_active_only: bool) -> List[Dict[str, Any]]:
    filters: Dict[str, object] = {}
    if require_active_only:
        filters["is_active"] = True

    query_embedding = pipeline.embedding_model.encode_query(query)
    try:
        results = pipeline.qdrant_store.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            dense_weight=0.6,
            sparse_weight=0.4,
            filter_conditions=filters,
        )
    except Exception:
        results = pipeline.qdrant_store.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            dense_weight=0.6,
            sparse_weight=0.4,
            filter_conditions={},
        )
    if not results and require_active_only:
        results = pipeline.qdrant_store.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            dense_weight=0.6,
            sparse_weight=0.4,
            filter_conditions={},
        )
    return to_citation_items(results[:top_k])


def execute_chat_query(
    pipeline,
    query: str,
    top_k: int,
    use_reranker: bool,
    require_active_only: bool,
    use_graph: bool,
) -> ChatExecutionResult:
    started = time.perf_counter()
    try:
        legal_answer = pipeline.query(
            question=query,
            top_k=top_k,
            filter_conditions=None,
            use_reranker=use_reranker,
            include_inactive=not require_active_only,
            use_graph_context=use_graph,
            graph_hops=1,
            graph_expand_docs=20,
        )
        answer_text = str(legal_answer.answer or "").strip()
        if is_llm_failure(answer_text):
            raise RuntimeError(answer_text or "LLM generation failed.")
        return ChatExecutionResult(
            answer_text=answer_text,
            citations=list(legal_answer.citations or []),
            retrieval_time_ms=float(legal_answer.retrieval_time) * 1000.0,
            generation_time_ms=float(legal_answer.generation_time) * 1000.0,
            rerank_time_ms=float(getattr(legal_answer, "rerank_time", 0.0)) * 1000.0,
            rewrite_time_ms=float(getattr(legal_answer, "rewrite_time", 0.0)) * 1000.0,
            used_reranker=bool(getattr(legal_answer, "used_reranker", False)),
            rewrite_used=bool(getattr(legal_answer, "rewrite_used", False)),
            rewrite_status=str(getattr(legal_answer, "rewrite_status", "disabled")),
            rewrite_fallback_reason=str(getattr(legal_answer, "rewrite_fallback_reason", "")),
            rewritten_queries=list(getattr(legal_answer, "rewritten_queries", []) or []),
            missing_numbers=list(getattr(legal_answer, "missing_numbers", []) or []),
            per_subquery_hits=list(getattr(legal_answer, "per_subquery_hits", []) or []),
            retrieval_union_count=int(getattr(legal_answer, "retrieval_union_count", 0) or 0),
            fallback_used=False,
            fallback_error="",
            status="ok",
        )
    except Exception as exc:
        fallback_start = time.perf_counter()
        citations = retrieve_only(pipeline, query, top_k, require_active_only=require_active_only)
        retrieval_time_ms = (time.perf_counter() - fallback_start) * 1000.0
        _ = time.perf_counter() - started
        return ChatExecutionResult(
            answer_text=(
                "LLM unavailable or failed. Returning retrieval-only sources. "
                "Please inspect citations below."
            ),
            citations=citations,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=0.0,
            rerank_time_ms=0.0,
            rewrite_time_ms=0.0,
            used_reranker=False,
            rewrite_used=False,
            rewrite_status="disabled",
            rewrite_fallback_reason="",
            rewritten_queries=[],
            missing_numbers=[],
            per_subquery_hits=[],
            retrieval_union_count=len(citations),
            fallback_used=True,
            fallback_error=str(exc),
            status="fallback" if citations else "error",
        )


def format_sources(citations: List[Dict[str, Any]], top_k: int) -> str:
    lines = []
    for idx, citation in enumerate(citations[:top_k], start=1):
        citation_text = str(citation.get("citation_full", "")).replace("\n", " ").strip()
        if len(citation_text) > 260:
            citation_text = citation_text[:260].rstrip() + "..."
        lines.append(
            f"{idx}. doc_id=`{citation.get('doc_id', '')}` | path=`{citation.get('legal_path', '')}` "
            f"| type=`{citation.get('chunk_type', '')}` | score=`{float(citation.get('score', 0.0)):.4f}`\n"
            f"   - {citation_text}"
        )
    return "\n".join(lines) if lines else "No sources."


def format_chainlit_message(result: ChatExecutionResult, top_k: int) -> str:
    source_text = format_sources(result.citations, top_k=top_k)
    return (
        f"{result.answer_text}\n\n"
        f"---\n\n"
        f"**Metrics**\n"
        f"- retrieval_time_ms: `{result.retrieval_time_ms:.2f}`\n"
        f"- rewrite_time_ms: `{result.rewrite_time_ms:.2f}`\n"
        f"- rewrite_used: `{result.rewrite_used}`\n"
        f"- rewrite_status: `{result.rewrite_status}`\n"
        f"- rewrite_fallback_reason: `{result.rewrite_fallback_reason}`\n"
        f"- rewrite_query_count: `{len(result.rewritten_queries)}`\n"
        f"- retrieval_union_count: `{result.retrieval_union_count}`\n"
        f"- rerank_time_ms: `{result.rerank_time_ms:.2f}`\n"
        f"- used_reranker: `{result.used_reranker}`\n"
        f"- generation_time_ms: `{result.generation_time_ms:.2f}`\n"
        f"- fallback_used: `{result.fallback_used}`\n"
        f"- fallback_error: `{result.fallback_error}`\n\n"
        f"**Rewrite Trace**\n"
        f"- rewritten_queries: `{result.rewritten_queries}`\n"
        f"- missing_numbers: `{result.missing_numbers}`\n"
        f"- per_subquery_hits: `{result.per_subquery_hits}`\n\n"
        f"**Top Sources**\n{source_text}"
    )

