from src.qdrant_store import QdrantLegalStore, get_qdrant_store, reset_qdrant_store


class _FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def close(self):
        return None


def test_qdrant_store_prefers_cloud_runtime_env(monkeypatch):
    captured = {}

    def _fake_client(**kwargs):
        captured.update(kwargs)
        return _FakeClient(**kwargs)

    monkeypatch.setenv("QDRANT_USE_CLOUD", "true")
    monkeypatch.setenv("QDRANT_USE_SERVER", "false")
    monkeypatch.setenv("QDRANT_CLOUD_URL", "https://example.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "secret")
    monkeypatch.setattr("src.qdrant_store.QdrantClient", _fake_client)

    store = QdrantLegalStore()

    assert store.use_cloud is True
    assert store.use_server is False
    assert store.cloud_url == "https://example.qdrant.io:6333"
    assert captured["url"] == "https://example.qdrant.io:6333"
    assert captured["api_key"] == "secret"


def test_get_qdrant_store_recreates_when_runtime_signature_changes(monkeypatch):
    created = []

    def _fake_client(**kwargs):
        created.append(kwargs)
        return _FakeClient(**kwargs)

    monkeypatch.setattr("src.qdrant_store.QdrantClient", _fake_client)

    reset_qdrant_store()
    monkeypatch.setenv("QDRANT_USE_CLOUD", "true")
    monkeypatch.setenv("QDRANT_USE_SERVER", "false")
    monkeypatch.setenv("QDRANT_CLOUD_URL", "https://cluster-a.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "key-a")
    store_a = get_qdrant_store()

    monkeypatch.setenv("QDRANT_CLOUD_URL", "https://cluster-b.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "key-b")
    store_b = get_qdrant_store()

    assert store_a is not store_b
    assert created[0]["url"] == "https://cluster-a.qdrant.io:6333"
    assert created[1]["url"] == "https://cluster-b.qdrant.io:6333"

    reset_qdrant_store()
