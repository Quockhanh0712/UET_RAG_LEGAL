from datetime import date

from src.legal_data_processing import (
    IngestionStateStore,
    clean_serving_title,
    normalize_document_row,
)


def test_normalize_document_row_active_flag_and_citation(tmp_path):
    row = {
        "so_ky_hieu": "12/2020/ND-CP",
        "ten_van_ban": "Nghi dinh ve xu phat",
        "loai_van_ban": "Nghi dinh",
        "co_quan_ban_hanh": "Chinh phu",
        "ngay_ban_hanh": "01/01/2020",
        "ngay_co_hieu_luc": "15/01/2020",
        "ngay_het_hieu_luc": "01/01/2021",
        "toan_van_clean": "Dieu 1. Noi dung van ban.",
        "mo_ta": "Sua doi mot so dieu.",
    }

    normalized = normalize_document_row(row, as_of_date=date(2026, 1, 1))
    assert normalized["doc_id"] == "12/2020/ND-CP"
    assert normalized["is_active"] is False
    assert normalized["doc_type"] == "NGHI_DINH"
    assert normalized["citation_base"]
    assert normalized["content_hash"]


def test_normalize_document_row_cleans_serving_title_before_citation():
    row = {
        "so_ky_hieu": "100/2015/QH13",
        "ten_van_ban": "BỘ LUẬT HÌNH SỰ __ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam;",
        "loai_van_ban": "Bộ luật",
        "co_quan_ban_hanh": "Quốc hội",
        "toan_van_clean": "Điều 1. Nội dung văn bản.",
    }

    normalized = normalize_document_row(row, as_of_date=date(2026, 1, 1))

    assert normalized["ten_van_ban"] == "BỘ LUẬT HÌNH SỰ"
    assert normalized["law_family"] == "BỘ LUẬT HÌNH SỰ"
    assert normalized["citation_base"] == "Bộ luật 100/2015/QH13 - BỘ LUẬT HÌNH SỰ"


def test_ingestion_state_store_diff(tmp_path):
    state_path = tmp_path / "state.json"
    store = IngestionStateStore(str(state_path))

    records = [
        {"doc_id": "A", "content_hash": "h1", "is_active": True, "doc_type": "LUẬT", "doc_type_rank": 1},
        {"doc_id": "B", "content_hash": "h2", "is_active": True, "doc_type": "NGHỊ ĐỊNH", "doc_type_rank": 3},
    ]
    diff_1 = store.diff(records)
    assert len(diff_1.to_update) == 2
    store.update_state(records)

    diff_2 = store.diff(records)
    assert len(diff_2.to_update) == 0
    assert len(diff_2.unchanged) == 2

    changed = [
        {"doc_id": "A", "content_hash": "h1-new", "is_active": True, "doc_type": "LUẬT", "doc_type_rank": 1}
    ]
    diff_3 = store.diff(changed)
    assert len(diff_3.to_update) == 1


def test_clean_serving_title_trims_preamble_noise():
    cleaned, flags, confident = clean_serving_title(
        "LUẬT Đo lường ____ Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam;",
        "Luật",
        "04/2011/QH13",
    )

    assert cleaned == "LUẬT Đo lường"
    assert confident is True
    assert "trimmed_preamble_tail" in flags
    assert "removed_ocr_noise" in flags


def test_clean_serving_title_keeps_ban_hanh_titles_when_valid():
    cleaned, flags, confident = clean_serving_title(
        "NGHỊ ĐỊNH BAN HÀNH ĐIỀU LỆ TỔ CHỨC VÀ HOẠT ĐỘNG CỦA TẬP ĐOÀN HÓA CHẤT VIỆT NAM Căn cứ Luật tổ chức Chính phủ ngày 19 tháng 6 năm 2015;",
        "Nghị định",
        "20/2018/NĐ-CP",
    )

    assert cleaned == "NGHỊ ĐỊNH BAN HÀNH ĐIỀU LỆ TỔ CHỨC VÀ HOẠT ĐỘNG CỦA TẬP ĐOÀN HÓA CHẤT VIỆT NAM"
    assert confident is True
    assert "trimmed_preamble_tail" in flags


def test_clean_serving_title_trims_trailing_issuer_header():
    cleaned, flags, confident = clean_serving_title(
        "NGHỊ ĐỊNH Về chào bán cổ phần riêng lẻ CHÍNH PHỦ Căn cứ Luật Tổ chức Chính phủ ngày 25 tháng 12 năm 2001;",
        "Nghị định",
        "01/2010/NĐ-CP",
    )

    assert cleaned == "NGHỊ ĐỊNH Về chào bán cổ phần riêng lẻ"
    assert confident is True
    assert "trimmed_trailing_issuer" in flags


def test_clean_serving_title_marks_low_confidence_when_doc_type_missing():
    cleaned, flags, confident = clean_serving_title(
        "abc",
        "Nghị định",
        "70/2015/NĐ-CP",
    )

    assert cleaned
    assert confident is False
    assert "low_confidence_title" in flags
