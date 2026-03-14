from pathlib import Path

from src.config.runtime_settings import load_runtime_settings
from src.runtime.qdrant_contract import resolve_collection_name, validate_qdrant_contract


class _FakeCollection:
    def __init__(self, name: str):
        self.name = name


class _FakeCollections:
    def __init__(self, names):
        self.collections = [_FakeCollection(name) for name in names]


class _FakeStatus:
    def __init__(self, value: str):
        self.value = value


class _FakeDense:
    def __init__(self, size: int):
        self.size = size


class _FakeParams:
    def __init__(self, size: int):
        self.vectors = {"dense": _FakeDense(size)}


class _FakeConfig:
    def __init__(self, size: int):
        self.params = _FakeParams(size)


class _FakeInfo:
    def __init__(self, status: str, size: int):
        self.status = _FakeStatus(status)
        self.config = _FakeConfig(size)


class _FakeClient:
    def __init__(self, status: str, size: int):
        self._status = status
        self._size = size

    def get_collections(self):
        return _FakeCollections(["legal_documents__truro7_vn_law_embedding__d768"])

    def get_collection(self, collection_name: str):
        return _FakeInfo(self._status, self._size)

    def scroll(self, **kwargs):
        return [], None


def test_validate_qdrant_contract_allows_yellow_when_queryable(monkeypatch, tmp_path):
    bm25_path = tmp_path / "bm25_encoder.json"
    bm25_path.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("QDRANT_CLOUD_URL", "https://example.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "secret")
    monkeypatch.setenv("CHAINLIT_BM25_PATH", str(bm25_path))

    settings = load_runtime_settings("chainlit_demo")
    monkeypatch.setattr(
        "src.runtime.qdrant_contract.get_qdrant_client",
        lambda settings, timeout=30: _FakeClient(status="yellow", size=768),
    )

    contract = validate_qdrant_contract(settings, embedding_dim=768)

    assert contract.contract_ok is True
    assert contract.collection_status == "yellow"
    assert contract.vector_dim == 768
    assert contract.collection_name == "legal_documents__truro7_vn_law_embedding__d768"


def test_resolve_collection_name_prefers_override(monkeypatch):
    monkeypatch.setenv("QDRANT_CLOUD_URL", "https://example.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "secret")
    monkeypatch.setenv("CHAINLIT_COLLECTION_OVERRIDE", "legal_rag_vn")

    settings = load_runtime_settings("chainlit_demo")

    assert resolve_collection_name(settings, embedding_dim=768) == "legal_rag_vn"

