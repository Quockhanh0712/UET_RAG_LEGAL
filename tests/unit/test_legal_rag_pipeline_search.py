from src.legal_rag_pipeline import LegalRAGPipeline
from src.qdrant_store import SearchResult


def test_search_returns_retrieval_metadata(monkeypatch):
    pipeline = LegalRAGPipeline(embedding_model_name="truro7/vn-law-embedding")

    fake_results = [
        SearchResult(
            chunk_id="chunk-1",
            content="content-1",
            score=0.9,
            metadata={
                "doc_id": "VB1",
                "citation_full": "Luật VB1 | Dieu 1",
                "legal_path": "Dieu 1",
                "chunk_type": "ProvisionChunk",
            },
        )
    ]

    def _fake_retrieve(**kwargs):
        return fake_results, {
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
            "rewritten_queries": ["q1", "q2"],
            "missing_numbers": [],
            "per_subquery_hits": [{"query": "q1", "hits": 10}],
            "retrieval_union_count": 12,
        }

    monkeypatch.setattr(pipeline, "_retrieve", _fake_retrieve)

    result = pipeline.search("test query", top_k=5, use_reranker=True)

    assert result.intent == "factual"
    assert result.used_reranker is True
    assert result.rewrite_used is True
    assert result.rewrite_status == "ok"
    assert len(result.citations) == 1
    assert result.citations[0]["citation_full"] == "Luật VB1 | Dieu 1"
