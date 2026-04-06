from __future__ import annotations

import importlib
import sys
import types


def _install_embedding_stubs(monkeypatch) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda index: "cpu",
    )
    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = object
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = object

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_embedding_modules_default_to_legal_contract(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL_NAME", raising=False)
    _install_embedding_stubs(monkeypatch)

    embeddings = importlib.import_module("src.embeddings")
    chunking = importlib.import_module("src.document_ingestion.chunking")
    indexer = importlib.import_module("src.document_ingestion.indexer")

    importlib.reload(embeddings)
    importlib.reload(chunking)
    importlib.reload(indexer)

    assert embeddings.DEFAULT_MODEL == "truro7/vn-law-embedding"
    assert chunking.DEFAULT_MODEL == "truro7/vn-law-embedding"
    assert indexer._current_embedding_model_name() == "truro7/vn-law-embedding"
