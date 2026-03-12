from src.legal_graph_extractor import (
    REL_ABROGATES,
    REL_AMENDS,
    REL_CITES,
    REL_REPLACES,
    LegalGraphExtractor,
    extract_doc_refs,
)


def test_extract_doc_refs():
    text = "Van ban nay sua doi 12/2020/ND-CP va thay the 99/2015/TT-BTC."
    refs = extract_doc_refs(text)
    assert "12/2020/ND-CP" in refs
    assert "99/2015/TT-BTC" in refs


def test_extract_amendment_relations():
    extractor = LegalGraphExtractor()
    mo_ta = (
        "Van ban nay sua doi 12/2020/ND-CP; "
        "thay the 34/2018/TT-BTC; "
        "bai bo 55/2010/QD-TTg."
    )
    relations = extractor.extract_amendment_relations("01/2025/ND-CP", mo_ta)
    relation_types = {r.relation_type for r in relations}

    assert REL_AMENDS in relation_types
    assert REL_REPLACES in relation_types
    assert REL_ABROGATES in relation_types
    assert all(r.source_doc_id == "01/2025/ND-CP" for r in relations)
    assert all(r.evidence_source == "mo_ta" for r in relations)
    assert all(r.snapshot_date for r in relations)


def test_extract_cites_when_no_amendment_keyword():
    extractor = LegalGraphExtractor()
    mo_ta = "Van ban nay dan chieu 12/2020/ND-CP de ap dung."
    relations = extractor.extract_amendment_relations("01/2026/TT-BTC", mo_ta)

    assert len(relations) >= 1
    assert any(r.relation_type == REL_CITES for r in relations)
    assert all(r.confidence_rule for r in relations)
