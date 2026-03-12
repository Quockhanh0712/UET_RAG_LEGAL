from src.query_intent import QueryIntent, QueryIntentClassifier


def test_intent_amendment_enables_legal_impact_graph():
    clf = QueryIntentClassifier()
    result = clf.classify("Văn bản nào sửa đổi Nghị định 12/2020/NĐ-CP?")

    assert result.intent == QueryIntent.AMENDMENT
    assert result.graph_mode == "legal_impact"
    assert result.sparse_weight >= result.dense_weight


def test_intent_temporal_enables_temporal_graph():
    clf = QueryIntentClassifier()
    result = clf.classify("Nghị định này còn hiệu lực đến ngày nào?")

    assert result.intent == QueryIntent.TEMPORAL
    assert result.graph_mode == "temporal"


def test_general_intent_disables_graph_by_default():
    clf = QueryIntentClassifier()
    result = clf.classify("Thủ tục đăng ký kinh doanh gồm những gì?")

    assert result.graph_mode == "none"
