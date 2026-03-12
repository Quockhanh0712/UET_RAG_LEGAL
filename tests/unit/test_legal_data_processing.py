from datetime import date

from src.legal_data_processing import IngestionStateStore, normalize_document_row


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
