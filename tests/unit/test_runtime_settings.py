from pathlib import Path

from src.config.runtime_settings import load_runtime_settings


def test_chainlit_profile_locks_model_and_reads_runtime_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding")
    monkeypatch.setenv("QUERY_REWRITER_ENABLED", "true")
    monkeypatch.setenv("CHAINLIT_TOP_K", "7")
    monkeypatch.setenv("CHAINLIT_BM25_PATH", str(tmp_path / "bm25.json"))
    monkeypatch.setenv("QDRANT_CLOUD_URL", "https://example.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "secret")

    settings = load_runtime_settings("chainlit_demo")

    assert settings.embedding_model == "truro7/vn-law-embedding"
    assert settings.query_rewriter_enabled is True
    assert settings.chainlit_top_k == 7
    assert settings.bm25_path == Path(tmp_path / "bm25.json")
    assert settings.qdrant_url == "https://example.qdrant.io:6333"
    assert settings.qdrant_api_key == "secret"

