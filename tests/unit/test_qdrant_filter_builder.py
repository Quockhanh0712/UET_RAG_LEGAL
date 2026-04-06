from src.qdrant_store import QdrantLegalStore


def test_qdrant_store_build_filter_for_active_only():
    store = QdrantLegalStore.__new__(QdrantLegalStore)
    qdrant_filter = store._build_filter({"is_active": True})

    assert qdrant_filter is not None
    assert len(qdrant_filter.must) == 1
    assert qdrant_filter.must[0].key == "is_active"
    assert qdrant_filter.must[0].match.value is True

