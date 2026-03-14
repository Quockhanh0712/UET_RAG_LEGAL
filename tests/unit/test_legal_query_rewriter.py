from src.legal_query_rewriter import LegalQueryRewriter


def test_rewrite_query_returns_subqueries_and_anchor(monkeypatch):
    rewriter = LegalQueryRewriter()
    raw_output = """
<analysis>
- Hanh vi chinh: Lai xe say ruou gay tai nan.
</analysis>
<queries>
<query>Trach nhiem hinh su cua tai xe say ruou gay tai nan?</query>
<query>Trach nhiem boi thuong dan su cua cong ty khi nhan vien gay tai nan?</query>
</queries>
"""

    monkeypatch.setattr(rewriter, "_call_llm", lambda prompt: raw_output)

    result = rewriter.rewrite_query(
        "Tai xe A lai xe giao hang cua cong ty B, do say ruou nen dam chet nguoi. Trach nhiem boi thuong va di tu thuoc ve ai?",
        force=True,
    )

    assert result.status == "ok"
    assert len(result.queries) == 3
    assert result.queries[-1].startswith("Tai xe A")
    assert result.rewrite_used is True


def test_rewrite_query_falls_back_when_numbers_are_lost(monkeypatch):
    rewriter = LegalQueryRewriter()
    raw_output = """
<queries>
<query>Muc phat doi voi hanh vi vi pham?</query>
</queries>
"""

    monkeypatch.setattr(rewriter, "_call_llm", lambda prompt: raw_output)

    result = rewriter.rewrite_query(
        "Nguoi 17 tuoi chay xe 80 km/h thi bi phat bao nhieu?",
        force=True,
    )

    assert result.status == "fallback_original"
    assert result.queries == ["Nguoi 17 tuoi chay xe 80 km/h thi bi phat bao nhieu?"]
    assert result.missing_numbers == ["17", "80"]
    assert result.fallback_reason == "missing_numbers"
    assert result.rewrite_used is False


def test_rewrite_query_falls_back_on_api_error(monkeypatch):
    rewriter = LegalQueryRewriter()

    def _boom(prompt: str) -> str:
        raise RuntimeError("timeout")

    monkeypatch.setattr(rewriter, "_call_llm", _boom)

    result = rewriter.rewrite_query(
        "Giam doc cong ty ra quyet dinh xa thai trai phep de tang loi nhuan, nhan vien ky thuat van thuc hien. Trach nhiem thuoc ve ai?",
        force=True,
    )

    assert result.status == "api_error"
    assert result.queries == [
        "Giam doc cong ty ra quyet dinh xa thai trai phep de tang loi nhuan, nhan vien ky thuat van thuc hien. Trach nhiem thuoc ve ai?"
    ]
    assert result.rewrite_used is False
