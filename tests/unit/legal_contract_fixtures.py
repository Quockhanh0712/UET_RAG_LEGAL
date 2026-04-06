from __future__ import annotations

from typing import Any, Dict

from src.legal_rag.contracts.results import LegalAnswer, LegalSearchResult
from src.qdrant_store import SearchResult


def make_citation(**overrides: Any) -> Dict[str, Any]:
    citation = {
        "content": "Noi dung phap ly mau",
        "doc_id": "VB1",
        "van_ban_id": "VB1",
        "so_ky_hieu": "01/2024/QH15",
        "ten_van_ban": "Luat VB1",
        "loai_van_ban": "Luat",
        "co_quan": "Quoc hoi",
        "law_family": "administrative",
        "chuong": "I",
        "ten_chuong": "Quy dinh chung",
        "dieu_so": "1",
        "tieu_de_dieu": "Pham vi dieu chinh",
        "khoan_so": "1",
        "diem_so": "a",
        "chunk_type": "ProvisionChunk",
        "segment_level": "dieu",
        "legal_path": "Dieu 1, Khoan 1, Diem a",
        "is_active": True,
        "citation_base": "Luat VB1",
        "citation_full": "Luat VB1 | Dieu 1 | Khoan 1 | Diem a",
        "token_len_est": 42,
        "encoder_profile": "default",
        "score": 0.91,
    }
    citation.update(overrides)
    return citation


def make_search_result(**metadata_overrides: Any) -> SearchResult:
    metadata = {
        "doc_id": "VB1",
        "citation_full": "Luat VB1 | Dieu 1 | Khoan 1 | Diem a",
        "legal_path": "Dieu 1, Khoan 1, Diem a",
        "chunk_type": "ProvisionChunk",
        "ten_van_ban": "Luat VB1",
        "loai_van_ban": "Luat",
        "dieu_so": "1",
        "khoan_so": "1",
        "diem_so": "a",
        "is_active": True,
    }
    metadata.update(metadata_overrides)
    return SearchResult(
        chunk_id="chunk-1",
        content="Noi dung phap ly mau",
        score=0.91,
        metadata=metadata,
    )


def make_query_facets(**overrides: Any) -> Dict[str, Any]:
    facets = {
        "actors": ["nguoi su dung dat"],
        "issues": ["thu hoi dat"],
        "focus": "dieu_kien_ap_dung",
        "main_act": "Luat Dat dai",
        "conditions": ["co thong bao thu hoi"],
    }
    facets.update(overrides)
    return facets


def make_rewrite_debug(**overrides: Any) -> Dict[str, Any]:
    debug = {
        "main_act": "Luat Dat dai",
        "conditions": ["co thong bao thu hoi"],
        "clause_units": ["dieu_khoan_1"],
        "named_law_hints": ["Luat Dat dai 2024"],
        "question_units": [
            {
                "question_label": "dieu_kien",
                "user_asked_text": "thu hoi dat khi nao",
                "intent": "factual",
            }
        ],
        "main_act_candidate_reasons": ["named_law_detected"],
        "decomposition_policy": "objective_question_unit",
        "quality_fail_reasons": [],
    }
    debug.update(overrides)
    return debug


def make_pipeline_meta(**overrides: Any) -> Dict[str, Any]:
    rewritten_queries = list(overrides.pop("rewritten_queries", ["thu hoi dat", "dieu kien thu hoi dat"]))
    primary_query = rewritten_queries[0] if rewritten_queries else ""
    retrieval_query_plan = [
        {"query": query, "source": "rewrite", "weight": 1.0 if idx == 0 else 0.8}
        for idx, query in enumerate(rewritten_queries[:2])
    ]
    meta = {
        "intent": "factual",
        "retrieval_time": 0.25,
        "rerank_time": 0.05,
        "used_reranker": True,
        "graph_expansion_used": False,
        "graph_mode": "none",
        "rewrite_time": 0.01,
        "rewrite_used": True,
        "rewrite_status": "ok",
        "rewrite_fallback_reason": "",
        "rewritten_queries": rewritten_queries,
        "missing_numbers": [],
        "per_subquery_hits": [{"query": primary_query, "hits": 10}] if primary_query else [],
        "retrieval_union_count": 12,
        "rewrite_mode": "rewrite_combine",
        "rewrite_original_query": "thu hoi dat khi nao",
        "rewrite_queries_compact": rewritten_queries,
        "rewrite_strategy": "structured_synthesis",
        "rewrite_quality_passed": True,
        "retrieval_query_set": rewritten_queries,
        "retrieval_query_plan": retrieval_query_plan,
        "query_facets": make_query_facets(),
        "rewrite_debug": make_rewrite_debug(),
    }
    meta.update(overrides)
    return meta


def make_legal_search_result(**overrides: Any) -> LegalSearchResult:
    result = {
        "citations": [make_citation()],
        "intent": "factual",
        "retrieval_time": 0.25,
        "rerank_time": 0.05,
        "used_reranker": True,
        "graph_expansion_used": False,
        "graph_mode": "none",
        "rewrite_time": 0.01,
        "rewrite_used": True,
        "rewrite_status": "ok",
        "rewrite_fallback_reason": "",
        "rewritten_queries": ["thu hoi dat", "dieu kien thu hoi dat"],
        "missing_numbers": [],
        "per_subquery_hits": [{"query": "thu hoi dat", "hits": 10}],
        "retrieval_union_count": 12,
        "rewrite_mode": "rewrite_combine",
        "rewrite_original_query": "thu hoi dat khi nao",
        "rewrite_queries_compact": ["thu hoi dat", "dieu kien thu hoi dat"],
        "rewrite_strategy": "structured_synthesis",
        "rewrite_quality_passed": True,
        "retrieval_query_set": ["thu hoi dat", "dieu kien thu hoi dat"],
        "retrieval_query_plan": [{"query": "thu hoi dat", "source": "rewrite", "weight": 1.0}],
        "query_facets": make_query_facets(),
        "rewrite_debug": make_rewrite_debug(),
    }
    result.update(overrides)
    return LegalSearchResult(**result)


def make_legal_answer(**overrides: Any) -> LegalAnswer:
    answer = {
        "answer": "Tra loi mau",
        "citations": [make_citation()],
        "intent": "factual",
        "retrieval_time": 0.25,
        "generation_time": 0.4,
        "rerank_time": 0.05,
        "used_reranker": True,
        "graph_expansion_used": False,
        "graph_mode": "none",
        "rewrite_time": 0.01,
        "rewrite_used": True,
        "rewrite_status": "ok",
        "rewrite_fallback_reason": "",
        "rewritten_queries": ["thu hoi dat", "dieu kien thu hoi dat"],
        "missing_numbers": [],
        "per_subquery_hits": [{"query": "thu hoi dat", "hits": 10}],
        "retrieval_union_count": 12,
        "rewrite_mode": "rewrite_combine",
        "rewrite_original_query": "thu hoi dat khi nao",
        "rewrite_queries_compact": ["thu hoi dat", "dieu kien thu hoi dat"],
        "rewrite_strategy": "structured_synthesis",
        "rewrite_quality_passed": True,
        "retrieval_query_set": ["thu hoi dat", "dieu kien thu hoi dat"],
        "retrieval_query_plan": [{"query": "thu hoi dat", "source": "rewrite", "weight": 1.0}],
        "query_facets": make_query_facets(),
        "rewrite_debug": make_rewrite_debug(),
    }
    answer.update(overrides)
    return LegalAnswer(**answer)
