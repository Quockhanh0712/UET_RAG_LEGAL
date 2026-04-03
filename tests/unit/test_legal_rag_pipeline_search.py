from src.legal_rag_pipeline import LegalRAGPipeline
from tests.unit.legal_contract_fixtures import make_pipeline_meta, make_search_result


def test_search_returns_full_retrieval_metadata_contract(monkeypatch):
    pipeline = LegalRAGPipeline(embedding_model_name="truro7/vn-law-embedding")
    fake_results = [make_search_result()]

    def _fake_retrieve(**kwargs):
        return fake_results, make_pipeline_meta()

    monkeypatch.setattr(pipeline, "_retrieve", _fake_retrieve)

    result = pipeline.search("thu hoi dat khi nao", top_k=5, use_reranker=True)

    assert result.intent == "factual"
    assert result.used_reranker is True
    assert result.rewrite_used is True
    assert result.rewrite_status == "ok"
    assert result.rewrite_mode == "rewrite_combine"
    assert result.rewrite_original_query == "thu hoi dat khi nao"
    assert result.rewrite_queries_compact == ["thu hoi dat", "dieu kien thu hoi dat"]
    assert result.rewrite_strategy == "structured_synthesis"
    assert result.rewrite_quality_passed is True
    assert result.retrieval_query_set == ["thu hoi dat", "dieu kien thu hoi dat"]
    assert result.retrieval_query_plan == [
        {"query": "thu hoi dat", "source": "rewrite", "weight": 1.0},
        {"query": "dieu kien thu hoi dat", "source": "rewrite", "weight": 0.8},
    ]
    assert result.query_facets["main_act"] == "Luat Dat dai"
    assert result.rewrite_debug["named_law_hints"] == ["Luat Dat dai 2024"]
    assert result.per_subquery_hits == [{"query": "thu hoi dat", "hits": 10}]
    assert result.retrieval_union_count == 12
    assert len(result.citations) == 1
    assert result.citations[0]["citation_full"] == "Luat VB1 | Dieu 1 | Khoan 1 | Diem a"


def test_search_tolerates_missing_optional_rewrite_fields(monkeypatch):
    pipeline = LegalRAGPipeline(embedding_model_name="truro7/vn-law-embedding")
    fake_results = [make_search_result()]

    def _fake_retrieve(**kwargs):
        return fake_results, make_pipeline_meta(
            rewritten_queries=["thu hoi dat"],
            rewrite_mode=None,
            rewrite_original_query=None,
            rewrite_queries_compact=None,
            rewrite_strategy=None,
            rewrite_quality_passed=None,
            retrieval_query_set=None,
            retrieval_query_plan=None,
            query_facets=None,
            rewrite_debug=None,
        )

    monkeypatch.setattr(pipeline, "_retrieve", _fake_retrieve)

    result = pipeline.search("thu hoi dat khi nao", top_k=5, use_reranker=True)

    assert result.rewrite_mode == "original"
    assert result.rewrite_original_query == "thu hoi dat khi nao"
    assert result.rewrite_queries_compact == []
    assert result.rewrite_strategy == "structured_synthesis"
    assert result.rewrite_quality_passed is False
    assert result.retrieval_query_set == ["thu hoi dat"]
    assert result.retrieval_query_plan == []
    assert result.query_facets == {}
    assert result.rewrite_debug == {}
    assert len(result.citations) == 1
    assert result.citations[0]["citation_full"] == "Luat VB1 | Dieu 1 | Khoan 1 | Diem a"


def test_query_tolerates_missing_optional_rewrite_fields(monkeypatch):
    pipeline = LegalRAGPipeline(embedding_model_name="truro7/vn-law-embedding")
    fake_results = [make_search_result()]

    def _fake_retrieve(**kwargs):
        return fake_results, make_pipeline_meta(
            rewritten_queries=["thu hoi dat"],
            rewrite_mode=None,
            rewrite_original_query=None,
            rewrite_queries_compact=None,
            rewrite_strategy=None,
            rewrite_quality_passed=None,
            retrieval_query_set=None,
            retrieval_query_plan=None,
            query_facets=None,
            rewrite_debug=None,
        )

    class _StubLLM:
        @staticmethod
        def generate(question, structured_context):
            return "Tra loi test"

    monkeypatch.setattr(pipeline, "_retrieve", _fake_retrieve)

    result = pipeline.query("thu hoi dat khi nao", top_k=5, use_reranker=True, llm_client=_StubLLM())

    assert result.answer == "Tra loi test"
    assert result.rewrite_mode == "original"
    assert result.rewrite_original_query == "thu hoi dat khi nao"
    assert result.rewrite_queries_compact == []
    assert result.rewrite_strategy == "structured_synthesis"
    assert result.rewrite_quality_passed is False
    assert result.retrieval_query_set == ["thu hoi dat"]
    assert result.retrieval_query_plan == []
    assert result.query_facets == {}
    assert result.rewrite_debug == {}
    assert len(result.citations) == 1
    assert result.citations[0]["citation_full"] == "Luat VB1 | Dieu 1 | Khoan 1 | Diem a"
