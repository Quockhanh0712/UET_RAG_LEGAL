from src.legal_chunker_v2 import LegalChunkerV2


def test_chunk_document_by_legal_hierarchy():
    text = (
        "Chương I Quy định chung Điều 1. Phạm vi điều chỉnh 1. Khoản một về nội dung chung. "
        "a) Điểm a nội dung. b) Điểm b nội dung. 2. Khoản hai về trách nhiệm. "
        "Điều 2. Đối tượng áp dụng 1. Khoản một của điều hai."
    ).strip()

    record = {
        "doc_id": "01/2020/QH14",
        "so_ky_hieu": "01/2020/QH14",
        "ten_van_ban": "Van ban thu nghiem",
        "loai_van_ban": "Luật",
        "co_quan_ban_hanh": "Quoc hoi",
        "is_active": True,
        "toan_van_clean": text,
        "citation_base": "Luật 01/2020/QH14 - Van ban thu nghiem",
    }

    chunker = LegalChunkerV2(min_chars=40, max_chars=120, overlap_chars=20)
    chunks = chunker.chunk_document(record)

    assert len(chunks) >= 2
    assert all(chunk.doc_id == "01/2020/QH14" for chunk in chunks)
    assert all(chunk.citation_full for chunk in chunks)
    assert any(chunk.dieu_so == "1" for chunk in chunks)
    assert any(chunk.dieu_so == "2" for chunk in chunks)
    assert all(chunk.content for chunk in chunks)
    assert all(chunk.legal_path for chunk in chunks)
    assert all(chunk.chunk_type in {"ProvisionChunk", "ClauseChunk", "FallbackChunk"} for chunk in chunks)
