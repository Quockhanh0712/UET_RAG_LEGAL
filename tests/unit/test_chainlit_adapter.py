from src.adapters.chainlit_adapter import execute_chat_query, format_chainlit_message


class _DummyPipeline:
    def __init__(self, answer_obj=None, fallback_hits=None):
        self._answer_obj = answer_obj
        self._fallback_hits = fallback_hits or []
        self.embedding_model = self
        self.qdrant_store = self

    def encode_query(self, query: str):
        return [0.1, 0.2]

    def hybrid_search(self, **kwargs):
        return self._fallback_hits

    def query(self, **kwargs):
        if isinstance(self._answer_obj, Exception):
            raise self._answer_obj
        return self._answer_obj


class _Hit:
    def __init__(self):
        self.score = 0.9
        self.metadata = {
            "doc_id": "100/2015/QH13",
            "citation_full": "Bo luat 100/2015/QH13 - BO LUAT HINH SU | Dieu 77",
            "legal_path": "Dieu 77",
            "chunk_type": "ProvisionChunk",
            "is_active": True,
        }


class _Answer:
    answer = "Cau tra loi hop le"
    citations = [{"doc_id": "100/2015/QH13", "citation_full": "Bo luat ... | Dieu 77", "legal_path": "Dieu 77", "chunk_type": "ProvisionChunk", "score": 0.9}]
    retrieval_time = 1.0
    generation_time = 2.0
    rerank_time = 0.5
    rewrite_time = 0.2
    used_reranker = True
    rewrite_used = True
    rewrite_status = "ok"
    rewrite_fallback_reason = ""
    rewritten_queries = ["subquery", "query goc"]
    missing_numbers = []
    per_subquery_hits = [{"query": "subquery", "hits": 3}]
    retrieval_union_count = 5


def test_execute_chat_query_success():
    pipeline = _DummyPipeline(answer_obj=_Answer())

    result = execute_chat_query(
        pipeline=pipeline,
        query="query goc",
        top_k=5,
        use_reranker=True,
        require_active_only=True,
        use_graph=False,
    )

    assert result.status == "ok"
    assert result.rewrite_used is True
    assert result.used_reranker is True
    assert result.citations[0]["legal_path"] == "Dieu 77"


def test_execute_chat_query_falls_back_to_retrieval_only():
    pipeline = _DummyPipeline(answer_obj=RuntimeError("boom"), fallback_hits=[_Hit(), _Hit(), _Hit()])

    result = execute_chat_query(
        pipeline=pipeline,
        query="query goc",
        top_k=3,
        use_reranker=True,
        require_active_only=True,
        use_graph=False,
    )

    assert result.fallback_used is True
    assert result.status == "fallback"
    assert len(result.citations) == 3
    assert "retrieval-only" in result.answer_text
    assert "fallback_error" in format_chainlit_message(result, top_k=3)
