from src.services.query_rewrite_service import QueryRewriteService


class _StubRewriter:
    def __init__(self, result):
        self._result = result

    def rewrite_query(self, question: str, force: bool = False):
        return self._result


class _StubResult:
    def __init__(self, status: str, queries):
        self.status = status
        self.queries = list(queries)
        self.original_query = queries[-1] if queries else ""
        self.missing_numbers = []
        self.fallback_reason = ""
        self.latency_ms = 0.0

    @property
    def rewrite_used(self):
        return self.status == "ok"


def test_query_rewrite_service_returns_disabled_when_flag_off():
    service = QueryRewriteService(enabled=False, force=False)
    result = service.rewrite("abc")

    assert result.status == "disabled"
    assert result.queries == ["abc"]


def test_query_rewrite_service_uses_wrapped_rewriter():
    expected = _StubResult(status="ok", queries=["q1", "q2"])
    service = QueryRewriteService(enabled=True, force=True, rewriter=_StubRewriter(expected))

    result = service.rewrite("abc")

    assert result is expected

