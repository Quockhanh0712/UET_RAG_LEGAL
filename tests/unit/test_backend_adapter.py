import pytest

from src.adapters.backend_adapter import build_chat_payload, build_search_payload
from tests.unit.legal_contract_fixtures import make_citation, make_legal_answer, make_legal_search_result


def test_build_search_payload_exposes_trace_contract():
    result = make_legal_search_result()

    payload = build_search_payload(
        result,
        mode="legal",
        collection="legal_rag_vn",
        model_profile_id="default",
        query_mode="rewrite_combine",
    )

    assert payload["mode"] == "legal"
    assert payload["search_time"] == 0.25
    assert payload["rerank_time"] == 0.05
    assert payload["reranker_used"] is True
    assert payload["model_profile_id"] == "default"
    assert payload["query_mode"] == "rewrite_combine"
    assert payload["rewrite_used"] is True
    assert payload["rewrite_status"] == "ok"
    assert payload["rewrite_fallback_reason"] == ""
    assert payload["rewrite_strategy"] == "structured_synthesis"
    assert payload["rewrite_quality_passed"] is True
    assert payload["rewritten_queries"] == ["thu hoi dat", "dieu kien thu hoi dat"]
    assert payload["retrieval_query_set"] == ["thu hoi dat", "dieu kien thu hoi dat"]
    assert payload["retrieval_query_plan"] == [{"query": "thu hoi dat", "source": "rewrite", "weight": 1.0}]
    assert payload["per_subquery_hits"] == [{"query": "thu hoi dat", "hits": 10}]
    assert payload["retrieval_union_count"] == 12
    assert payload["rewrite_debug"]["named_law_hints"] == ["Luat Dat dai 2024"]
    assert payload["rewrite_context"] == {
        "original_query": "thu hoi dat khi nao",
        "rewrite_mode": "rewrite_combine",
        "rewrite_status": "ok",
        "rewrite_queries_compact": ["thu hoi dat", "dieu kien thu hoi dat"],
        "rewrite_fallback_reason": "",
        "query_facets": {
            "actors": ["nguoi su dung dat"],
            "issues": ["thu hoi dat"],
            "focus": "dieu_kien_ap_dung",
            "main_act": "Luat Dat dai",
            "conditions": ["co thong bao thu hoi"],
        },
        "rewrite_quality_passed": True,
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
    assert payload["fallback_used"] is False
    assert payload["fallback_error"] == ""
    assert payload["collection"] == "legal_rag_vn"
    assert payload["results"][0]["label"] == "Luat VB1 | Dieu 1 | Khoan 1 | Diem a"


def test_build_chat_payload_exposes_trace_contract():
    result = make_legal_answer()

    payload = build_chat_payload(
        result,
        answer="Tra loi mau",
        collection="legal_rag_vn",
        model_profile_id="default",
        query_mode="rewrite_combine",
        fallback_used=True,
        fallback_error="llm_timeout",
    )

    assert payload["answer"] == "Tra loi mau"
    assert payload["search_time"] == 0.25
    assert payload["generation_time"] == 0.4
    assert payload["total_time"] == pytest.approx(0.7)
    assert payload["rerank_time"] == 0.05
    assert payload["reranker_used"] is True
    assert payload["model_profile_id"] == "default"
    assert payload["query_mode"] == "rewrite_combine"
    assert payload["rewrite_used"] is True
    assert payload["rewrite_status"] == "ok"
    assert payload["rewrite_fallback_reason"] == ""
    assert payload["rewrite_strategy"] == "structured_synthesis"
    assert payload["rewrite_quality_passed"] is True
    assert payload["rewritten_queries"] == ["thu hoi dat", "dieu kien thu hoi dat"]
    assert payload["retrieval_query_set"] == ["thu hoi dat", "dieu kien thu hoi dat"]
    assert payload["retrieval_query_plan"] == [{"query": "thu hoi dat", "source": "rewrite", "weight": 1.0}]
    assert payload["per_subquery_hits"] == [{"query": "thu hoi dat", "hits": 10}]
    assert payload["retrieval_union_count"] == 12
    assert payload["rewrite_debug"]["named_law_hints"] == ["Luat Dat dai 2024"]
    assert payload["rewrite_context"]["original_query"] == "thu hoi dat khi nao"
    assert payload["rewrite_context"]["rewrite_mode"] == "rewrite_combine"
    assert payload["fallback_used"] is True
    assert payload["fallback_error"] == "llm_timeout"
    assert payload["collection"] == "legal_rag_vn"
    assert payload["sources"][0]["label"] == "Luat VB1 | Dieu 1 | Khoan 1 | Diem a"


def test_build_chat_payload_preserves_user_uploaded_legal_metadata():
    result = make_legal_answer(
        citations=[
            make_citation(
                source_type="user",
                doc_type="USER_UPLOAD",
                filename="35_2024_QH15_588811.pdf",
                page_number=2,
                ten_van_ban="35_2024_QH15_588811.pdf",
            )
        ]
    )

    payload = build_chat_payload(
        result,
        answer="Tra loi mau",
        collection="user_docs_private__truro7_vn_law_embedding__d768",
        model_profile_id="default",
        query_mode="rewrite_combine",
    )

    source = payload["sources"][0]
    assert source["type"] == "user"
    assert source["label"] == "Luat VB1 | Dieu 1 | Khoan 1 | Diem a"
    assert "35_2024_QH15_588811.pdf" in source["detail"]
    assert "Trang 2" in source["detail"]
    assert source["citation_full"] == "Luat VB1 | Dieu 1 | Khoan 1 | Diem a"
    assert source["legal_path"] == "Dieu 1, Khoan 1, Diem a"
