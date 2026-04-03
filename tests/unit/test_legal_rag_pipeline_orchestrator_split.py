from src.legal_rag.pipeline.context_builder import format_context
from src.legal_rag_pipeline import LegalRAGPipeline
from tests.unit.legal_contract_fixtures import make_pipeline_meta, make_search_result


def test_search_returns_contract_after_orchestrator_split(monkeypatch):
    pipeline = LegalRAGPipeline(embedding_model_name="truro7/vn-law-embedding")
    fake_results = [make_search_result()]

    def _fake_retrieve(**kwargs):
        return fake_results, make_pipeline_meta()

    monkeypatch.setattr(pipeline, "_retrieve", _fake_retrieve)

    result = pipeline.search("thu hoi dat khi nao", top_k=5, use_reranker=True)

    assert result.intent == "factual"
    assert result.rewrite_mode == "rewrite_combine"
    assert result.rewrite_original_query == "thu hoi dat khi nao"
    assert result.rewrite_queries_compact == ["thu hoi dat", "dieu kien thu hoi dat"]
    assert result.retrieval_query_set == ["thu hoi dat", "dieu kien thu hoi dat"]
    assert len(result.citations) == 1


def test_query_uses_evidence_context_not_query_context_as_ground_truth(monkeypatch):
    pipeline = LegalRAGPipeline(embedding_model_name="truro7/vn-law-embedding")
    fake_results = [make_search_result()]

    def _fake_retrieve(**kwargs):
        return fake_results, make_pipeline_meta()

    captured = {}

    class _ProbeLLM:
        @staticmethod
        def generate(question, structured_context):
            captured["question"] = question
            captured["context"] = structured_context
            return "Tra loi test"

    monkeypatch.setattr(pipeline, "_retrieve", _fake_retrieve)

    result = pipeline.query("thu hoi dat khi nao", top_k=5, use_reranker=True, llm_client=_ProbeLLM())

    assert result.answer == "Tra loi test"
    assert captured["question"] == "thu hoi dat khi nao"
    assert "query_context" in captured["context"]
    assert "evidence_context" in captured["context"]
    assert "REWRITE_GUARDRAIL" in captured["context"]["query_context"]
    assert "Noi dung phap ly mau" in captured["context"]["evidence_context"]


def test_context_builder_returns_citations_and_context_text():
    fake_results = [make_search_result(content="Noi dung phap ly mau")]

    context_text, citations = format_context(fake_results)

    assert "Luat VB1 | Dieu 1 | Khoan 1 | Diem a" in context_text
    assert len(citations) == 1
    assert citations[0]["citation_full"] == "Luat VB1 | Dieu 1 | Khoan 1 | Diem a"
    assert citations[0]["content"] == "Noi dung phap ly mau"
