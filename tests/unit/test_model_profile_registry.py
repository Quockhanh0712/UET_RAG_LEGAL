from __future__ import annotations

import importlib


def test_default_legal_model_profile_is_qwen35_9b():
    registry = importlib.import_module("src.services.model_profile_registry")
    importlib.reload(registry)

    assert registry.get_default_model_profile_id() == "qwen3_5_9b_ollama_remote"
    profile = registry.get_model_profile(None)

    assert profile.id == "qwen3_5_9b_ollama_remote"
    assert profile.label == "Qwen3.5 9B"
    assert profile.model == "qwen3.5:9b"


def test_qwen3_8b_profile_does_not_drift_with_global_ollama_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:9b")
    monkeypatch.setenv("OLLAMA_QWEN3_ENABLED", "true")
    monkeypatch.delenv("OLLAMA_QWEN3_MODEL", raising=False)

    registry = importlib.import_module("src.services.model_profile_registry")
    importlib.reload(registry)

    profile = registry.get_model_profile("qwen3_8b_ollama_remote")

    assert profile.model == "qwen3:8b"
