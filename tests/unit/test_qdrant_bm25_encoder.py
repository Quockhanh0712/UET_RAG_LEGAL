import pytest

from src.qdrant_store import BM25Encoder


def test_bm25_encoder_roundtrip(tmp_path):
    encoder = BM25Encoder()
    encoder.fit(
        [
            "Dieu 1 quy dinh ve pham vi dieu chinh",
            "Dieu 2 quy dinh ve doi tuong ap dung",
            "Khoan 3 neu ro nghia vu cua co quan nha nuoc",
        ]
    )
    indices_before, values_before = encoder.encode("quy dinh doi tuong")
    assert indices_before
    assert values_before

    save_path = tmp_path / "bm25_encoder.json"
    encoder.save(str(save_path))

    restored = BM25Encoder()
    restored.load(str(save_path))
    indices_after, values_after = restored.encode("quy dinh doi tuong")

    assert indices_after == indices_before
    assert values_after == pytest.approx(values_before)
    assert restored.vocab == encoder.vocab

