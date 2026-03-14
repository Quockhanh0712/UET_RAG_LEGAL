from src.legal_chunker_v2 import LegalChunkerV2


def _base_record(text: str):
    return {
        "doc_id": "100/2015/QH13",
        "so_ky_hieu": "100/2015/QH13",
        "ten_van_ban": "BỘ LUẬT HÌNH SỰ",
        "loai_van_ban": "Bộ luật",
        "doc_type": "LUAT",
        "doc_type_rank": 1,
        "co_quan_ban_hanh": "Quốc hội",
        "law_family": "BỘ LUẬT HÌNH SỰ",
        "is_active": True,
        "status_label": "active",
        "citation_base": "Bộ luật 100/2015/QH13 - BỘ LUẬT HÌNH SỰ",
        "toan_van_clean": text,
    }


def test_chunk_document_article_first_keeps_short_articles_as_provision_chunks():
    text = (
        "Chuong I Quy dinh chung "
        "Dieu 1. Pham vi dieu chinh 1. Noi dung cua khoan mot. 2. Noi dung cua khoan hai. "
        "Dieu 2. Doi tuong ap dung 1. Khoan mot cua dieu hai."
    )
    chunker = LegalChunkerV2()

    chunks = chunker.chunk_document(_base_record(text))

    assert len(chunks) == 2
    assert {chunk.dieu_so for chunk in chunks} == {"1", "2"}
    assert all(chunk.chunk_type == "ProvisionChunk" for chunk in chunks)
    assert all(chunk.segment_level == "article" for chunk in chunks)
    assert all(chunk.khoan_so == "" for chunk in chunks)
    assert all(chunk.citation_full.endswith(f"Dieu {chunk.dieu_so}") for chunk in chunks)


def test_chunk_document_numeric_groups_do_not_create_khoan_000():
    text = "Dieu 77. Phat tien 1. Phat tien duoc ap dung doi voi phap nhan. 100. 000. 000 dong."
    chunker = LegalChunkerV2()

    chunks = chunker.chunk_document(_base_record(text))

    assert len(chunks) == 1
    assert chunks[0].dieu_so == "77"
    assert chunks[0].legal_path == "Dieu 77"
    assert "Khoan 000" not in chunks[0].citation_full
    assert chunks[0].khoan_so == ""
    assert all("Khoan 000" not in span for span in chunks[0].clause_spans)


def test_chunk_document_long_article_splits_into_article_windows():
    repeated = "1. Phat tien duoc ap dung doi voi phap nhan thuong mai pham toi. " * 80
    text = f"Dieu 77. Phat tien {repeated}"
    chunker = LegalChunkerV2(target_chunk_tokens=80, hard_cap_tokens=96, overlap_tokens=16)

    chunks = chunker.chunk_document(_base_record(text))

    assert len(chunks) > 1
    assert all(chunk.chunk_type == "ProvisionChunk" for chunk in chunks)
    assert all(chunk.segment_level == "article_window" for chunk in chunks)
    assert all(chunk.provision_id == "100/2015/QH13_D77" for chunk in chunks)
    assert {chunk.overflow_total for chunk in chunks} == {len(chunks)}
    assert [chunk.overflow_index for chunk in chunks] == list(range(1, len(chunks) + 1))


def test_chunk_document_preserves_clause_spans_metadata_without_clause_chunks():
    text = (
        "Dieu 303. Toi pha huy cong trinh 1. Gay thiet hai nghiem trong. "
        "2. Gay thiet hai rat nghiem trong. a) Co to chuc. b) Dung chat no."
    )
    chunker = LegalChunkerV2()

    chunks = chunker.chunk_document(_base_record(text))

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "ProvisionChunk"
    assert chunks[0].segment_level == "article"
    assert chunks[0].clause_spans
    assert any(span.startswith("Khoan 1") or span.startswith("Khoan 1-2") for span in chunks[0].clause_spans)


def test_chunk_document_cleans_dirty_document_title():
    text = "Dieu 1. Pham vi dieu chinh 1. Noi dung khoan mot."
    record = _base_record(text)
    record["ten_van_ban"] = "BỘ LUẬT HÌNH SỰ __ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Bộ luật hình sự."
    record["law_family"] = record["ten_van_ban"]
    record["citation_base"] = "Bộ luật 100/2015/QH13 - " + record["ten_van_ban"]

    chunks = LegalChunkerV2().chunk_document(record)

    assert chunks
    assert chunks[0].ten_van_ban == "BỘ LUẬT HÌNH SỰ"
    assert chunks[0].law_family == "BỘ LUẬT HÌNH SỰ"
    assert chunks[0].citation_base == "Bộ luật 100/2015/QH13 - BỘ LUẬT HÌNH SỰ"
