from src.qdrant_store import build_qdrant_payload


def test_build_qdrant_payload_lean_drops_duplicate_and_audit_fields():
    payload = {
        "chunk_id": "04/2011/QH13_D41_K3_C197_01eee94f",
        "content": "Dieu 41. Quyen va nghia vu cua to chuc xa hoi nghe nghiep",
        "text": "compat alias should be removed",
        "doc_id": "04/2011/QH13",
        "van_ban_id": "04/2011/QH13",
        "so_ky_hieu": "04/2011/QH13",
        "ten_van_ban": "LUAT Do luong",
        "loai_van_ban": "Luật",
        "doc_type": "LUAT",
        "law_family": "LUAT Do luong",
        "legal_path": "",
        "path": "Dieu 41 > Khoan 3",
        "citation_base": "Luật 04/2011/QH13 - LUAT Do luong",
        "citation_full": "Luật 04/2011/QH13 - LUAT Do luong | Dieu 41 > Khoan 3",
        "citation_short": "Dieu 41 > Khoan 3",
        "chunk_type": "ClauseChunk",
        "segment_level": "clause",
        "is_active": True,
        "hierarchy": {"dieu": "41"},
        "ten_van_ban_raw": "raw title",
        "metadata_patch_version": "metadata_patch_v1",
    }

    lean_payload = build_qdrant_payload(payload, profile="lean")

    assert lean_payload["chunk_id"] == payload["chunk_id"]
    assert lean_payload["doc_id"] == "04/2011/QH13"
    assert lean_payload["van_ban_id"] == "04/2011/QH13"
    assert lean_payload["legal_path"] == "Dieu 41 > Khoan 3"
    assert lean_payload["citation_full"] == payload["citation_full"]
    assert "text" not in lean_payload
    assert "path" not in lean_payload
    assert "citation_short" not in lean_payload
    assert "hierarchy" not in lean_payload
    assert "ten_van_ban_raw" not in lean_payload
    assert "metadata_patch_version" not in lean_payload


def test_build_qdrant_payload_lean_omits_empty_optional_fields():
    payload = {
        "chunk_id": "x1",
        "content": "Noi dung",
        "doc_id": "70/2015/NĐ-CP",
        "van_ban_id": "",
        "so_ky_hieu": "",
        "ten_van_ban": "NGHỊ ĐỊNH Quy định chi tiết",
        "loai_van_ban": "Nghị định",
        "doc_type": "NGHI_DINH",
        "law_family": "NGHỊ ĐỊNH Quy định chi tiết",
        "legal_path": "Dieu 2 > Khoan 4",
        "citation_full": "Nghị định 70/2015/NĐ-CP - NGHỊ ĐỊNH Quy định chi tiết | Dieu 2 > Khoan 4",
        "chunk_type": "ClauseChunk",
        "segment_level": "clause",
        "is_active": True,
        "fallback_reason": "",
        "khoan_so": "4",
        "diem_so": "",
        "ngay_het_hieu_luc": None,
    }

    lean_payload = build_qdrant_payload(payload, profile="lean")

    assert lean_payload["van_ban_id"] == "70/2015/NĐ-CP"
    assert lean_payload["so_ky_hieu"] == "70/2015/NĐ-CP"
    assert "fallback_reason" not in lean_payload
    assert "diem_so" not in lean_payload
    assert "ngay_het_hieu_luc" not in lean_payload


def test_build_qdrant_payload_full_keeps_compatibility_aliases():
    payload = {
        "chunk_id": "abc",
        "content": "Noi dung",
        "text": "Noi dung",
        "path": "Dieu 1",
        "citation_short": "Dieu 1",
        "hierarchy": {"dieu": "1"},
    }

    full_payload = build_qdrant_payload(payload, profile="full")

    assert full_payload["text"] == "Noi dung"
    assert full_payload["path"] == "Dieu 1"
    assert full_payload["citation_short"] == "Dieu 1"
    assert full_payload["hierarchy"] == {"dieu": "1"}


def test_build_qdrant_payload_lean_keeps_user_tenant_fields():
    payload = {
        "chunk_id": "user_chunk_1",
        "content": "Tai lieu rieng cua nguoi dung",
        "doc_id": "user_doc_1",
        "van_ban_id": "user_doc_1",
        "so_ky_hieu": "user_doc_1",
        "ten_van_ban": "Hop dong mau",
        "loai_van_ban": "PDF",
        "doc_type": "USER_UPLOAD",
        "law_family": "Hop dong mau",
        "legal_path": "Trang 1",
        "citation_full": "Hop dong mau | Trang 1",
        "chunk_type": "GenericDocumentChunk",
        "segment_level": "page_block_window",
        "user_id": "user-123",
        "session_id": "session-abc",
        "filename": "hop_dong.pdf",
        "source_type": "user_upload",
        "page_number": 1,
        "upload_time": "2026-04-04T10:00:00",
        "ingest_mode": "ocr",
        "parser_used": "pymupdf+ocrmypdf",
        "ocr_provider": "default",
    }

    lean_payload = build_qdrant_payload(payload, profile="lean")

    assert lean_payload["user_id"] == "user-123"
    assert lean_payload["session_id"] == "session-abc"
    assert lean_payload["filename"] == "hop_dong.pdf"
    assert lean_payload["source_type"] == "user_upload"
    assert lean_payload["page_number"] == 1
    assert lean_payload["upload_time"] == "2026-04-04T10:00:00"
    assert lean_payload["ingest_mode"] == "ocr"
    assert lean_payload["parser_used"] == "pymupdf+ocrmypdf"
    assert lean_payload["ocr_provider"] == "default"
