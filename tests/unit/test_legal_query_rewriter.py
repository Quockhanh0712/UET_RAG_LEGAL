import pytest

from src.legal_query_rewriter import LegalQueryRewriter


def test_rewrite_query_returns_subqueries_and_anchor(monkeypatch):
    rewriter = LegalQueryRewriter()
    raw_output = """
{
  "needs_decomposition": true,
      "main_act": "gay tai nan chet nguoi",
  "surface_actors": [
    {"name": "Tai xe A", "actor_type": "nhan_vien", "source_span": "Tai xe A"},
    {"name": "Cong ty B", "actor_type": "to_chuc", "source_span": "Cong ty B"}
  ],
  "conditions": [],
  "explicit_legal_axes": ["hinh_su", "boi_thuong"],
  "question_focus": "chu_the_chiu_trach_nhiem",
  "clause_units": [],
  "named_law_hints": [],
  "question_units": [
    {
      "question_label": "Xac dinh chu the boi thuong",
      "actor_refs": ["Tai xe A", "Cong ty B"],
          "act_span": "gay tai nan chet nguoi",
      "condition_refs": [],
      "legal_axis": "boi_thuong",
      "user_asked_text": "trach nhiem boi thuong thuoc ve ai",
      "intent": "boi_thuong"
    }
  ]
}
"""

    monkeypatch.setattr(rewriter, "_call_llm", lambda prompt: raw_output)

    result = rewriter.rewrite_query(
        "Tai xe A lai xe giao hang cua cong ty B, do say ruou nen gay tai nan chet nguoi. Trach nhiem boi thuong va di tu thuoc ve ai?",
        force=True,
    )

    assert result.status == "ok"
    assert len(result.queries) >= 2
    assert any("gay tai nan chet nguoi" in query.lower() for query in result.queries)
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

    assert result.status == "ok"
    assert result.queries
    assert any("17" in query or "80" in query for query in result.queries)
    assert result.missing_numbers == []
    assert result.rewrite_used is True


def test_rewrite_query_raises_on_api_error(monkeypatch):
    rewriter = LegalQueryRewriter()

    def _boom(prompt: str) -> str:
        raise RuntimeError("timeout")

    monkeypatch.setattr(rewriter, "_call_llm", _boom)

    with pytest.raises(RuntimeError, match="timeout"):
        rewriter.rewrite_query(
            "Giam doc cong ty ra quyet dinh xa thai trai phep de tang loi nhuan, nhan vien ky thuat van thuc hien. Trach nhiem thuoc ve ai?",
            force=True,
        )
