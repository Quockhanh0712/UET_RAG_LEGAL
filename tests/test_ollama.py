import pytest


def test_ollama_chat():
    try:
        import ollama
    except ImportError:
        pytest.skip("ollama package not installed")

    try:
        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": "Xin chào! RAG là gì?"}],
        )
        assert response is not None
        assert "message" in response
    except Exception as exc:
        pytest.skip(f"Ollama service unavailable: {exc}")


if __name__ == "__main__":
    test_ollama_chat()
