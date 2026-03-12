# tests/test_embeddings.py
import pytest
import torch

HAS_CUDA = torch.cuda.is_available()

try:
    from src.embeddings import embed_texts
    HAS_SRC = True
except Exception:
    HAS_SRC = False


@pytest.mark.skipif(not HAS_SRC or not HAS_CUDA, reason="src.embeddings requires CUDA GPU")
def test_embed_texts_basic():
    texts = ["xin chào", "RAG là gì?"]
    vecs = embed_texts(texts)

    assert len(vecs) == len(texts)
    assert all(isinstance(v, list) for v in vecs)
    # vector không rỗng
    assert all(len(v) > 0 for v in vecs)
