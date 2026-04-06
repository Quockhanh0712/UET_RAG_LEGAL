from src.legal_query_rewriter import (
    LAW_HINT_PATTERNS,
    KeyCondition,
    QuestionUnit,
    StructuredRewriteExtraction,
    SurfaceActor,
)
from src.legal_rag.rewrite.heuristics import extract_named_law_hints, normalize_key, normalize_text
from src.legal_rag.rewrite.models import (
    KeyCondition as RewriteKeyCondition,
    QuestionUnit as RewriteQuestionUnit,
    StructuredRewriteExtraction as RewriteStructuredRewriteExtraction,
    SurfaceActor as RewriteSurfaceActor,
)
from src.legal_rag.rewrite.parser import merge_question_units
from src.legal_rag.rewrite.quality_gate import evaluate_quality_gate
from src.legal_rag.rewrite.synthesis import synthesize_query_set
from src.services.query_rewrite_service import QueryRewriteService


def test_heuristic_named_law_hints_extracts_expected_values():
    query = "Can cu theo Luat Lao dong 2019 va Nghi dinh 12/2022/ND-CP, ai chiu trach nhiem?"

    hints = extract_named_law_hints(query, law_hint_patterns=LAW_HINT_PATTERNS)

    assert any("Luat Lao dong 2019" in hint for hint in hints)
    assert any("Nghi dinh 12/2022/ND-CP" in hint for hint in hints)


def test_merge_question_units_deduplicates_and_preserves_axis():
    preferred = [
        QuestionUnit(
            question_label="Xac dinh chu the boi thuong",
            actor_refs=["Cong ty A", "Nhan vien B"],
            act_span="gay tai nan",
            condition_refs=["lam theo chi dao cap tren"],
            legal_axis="boi_thuong",
            user_asked_text="ai boi thuong",
            intent="boi_thuong",
        )
    ]
    fallback = [
        QuestionUnit(
            question_label="Xac dinh chu the boi thuong",
            actor_refs=["Cong ty A", "Nhan vien B"],
            act_span="gay tai nan",
            condition_refs=["lam theo chi dao cap tren"],
            legal_axis="boi_thuong",
            user_asked_text="ai boi thuong",
            intent="boi_thuong",
        ),
        QuestionUnit(
            question_label="Dieu kien mien trach nhiem",
            actor_refs=["Nhan vien B"],
            act_span="gay tai nan",
            condition_refs=["17 tuoi"],
            legal_axis="hinh_su",
            user_asked_text="mien trach nhiem hinh su",
            intent="dieu_kien_ap_dung",
        ),
    ]

    merged = merge_question_units(preferred, fallback, normalize_key_fn=normalize_key, max_units=3)

    assert len(merged) == 2
    assert merged[0].legal_axis == "boi_thuong"
    assert merged[1].legal_axis == "hinh_su"


def test_synthesize_query_plan_keeps_original_query():
    query_plan = [
        {"query": "Trach nhiem boi thuong cua cong ty A khi gay tai nan"},
        {"query": "Dieu kien mien trach nhiem cua nhan vien B"},
    ]

    query_set = synthesize_query_set(
        original_query="Cong ty A va nhan vien B ai chiu trach nhiem?",
        query_plan=query_plan,
        max_queries=4,
    )

    assert query_set[-1] == "Cong ty A va nhan vien B ai chiu trach nhiem?"
    assert len(query_set) == 3


def test_quality_gate_marks_invalid_main_act():
    extraction = StructuredRewriteExtraction(
        main_act="hanh vi duoc neu",
        surface_actors=[SurfaceActor(name="Cong ty A", actor_type="to_chuc", source_span="Cong ty A")],
        conditions=[KeyCondition(kind="quan_he_lao_dong", value="lam theo chi dao", source_span="lam theo chi dao")],
        question_units=[
            QuestionUnit(
                question_label="Xac dinh chu the",
                actor_refs=["Cong ty A"],
                act_span="hanh vi duoc neu",
                condition_refs=["lam theo chi dao"],
                legal_axis="boi_thuong",
                user_asked_text="ai chiu trach nhiem",
                intent="xac_dinh_chu_the",
            )
        ],
    )
    query_plan = [
        {
            "query": "Xac dinh chu the lien quan den hanh vi duoc neu cua cong ty A",
            "source_question_unit": "Xac dinh chu the",
        }
    ]

    reasons = evaluate_quality_gate(
        extraction=extraction,
        query_plan=query_plan,
        original_query="Cong ty A gay tai nan thi ai chiu trach nhiem?",
        normalize_key_fn=normalize_key,
        normalize_text_fn=normalize_text,
        is_generic_main_act_fn=lambda value: "hanh vi duoc neu" in normalize_key(value),
        drift_tokens=("mang thai ho",),
    )

    assert "generic_main_act" in reasons


def test_query_rewrite_service_fallbacks_on_exception():
    class _BoomRewriter:
        def rewrite_query(self, question: str, force: bool = False):
            raise RuntimeError("gateway timeout")

    service = QueryRewriteService(enabled=True, force=True, rewriter=_BoomRewriter())

    result = service.rewrite("Ai chiu trach nhiem boi thuong?")

    assert result.status == "api_error"
    assert result.queries == ["Ai chiu trach nhiem boi thuong?"]
    assert "gateway timeout" in result.fallback_reason


def test_public_rewrite_dtos_are_reexported_from_rewrite_models():
    assert SurfaceActor is RewriteSurfaceActor
    assert KeyCondition is RewriteKeyCondition
    assert QuestionUnit is RewriteQuestionUnit
    assert StructuredRewriteExtraction is RewriteStructuredRewriteExtraction
