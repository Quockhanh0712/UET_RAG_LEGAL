"""Payload shaping policies for Qdrant documents."""

from __future__ import annotations

from typing import Any, Dict, Tuple

LEAN_QDRANT_PAYLOAD_FIELDS: Tuple[str, ...] = (
    "chunk_id",
    "content",
    "doc_id",
    "so_ky_hieu",
    "van_ban_id",
    "ten_van_ban",
    "loai_van_ban",
    "doc_type",
    "doc_type_rank",
    "co_quan_ban_hanh",
    "co_quan",
    "chuc_danh",
    "nguoi_ky",
    "law_family",
    "ngay_ban_hanh",
    "ngay_co_hieu_luc",
    "ngay_het_hieu_luc",
    "is_active",
    "status_label",
    "chuong",
    "ten_chuong",
    "dieu_so",
    "tieu_de_dieu",
    "khoan_so",
    "diem_so",
    "legal_path",
    "chunk_type",
    "segment_level",
    "fallback_reason",
    "provision_id",
    "parent_provision_id",
    "citation_base",
    "citation_full",
    "chunk_index",
    "total_chunks",
    "char_start",
    "char_end",
    "token_len_est",
    "token_len_model",
    "overflow_index",
    "overflow_total",
    "window_char_start",
    "window_char_end",
    "clause_spans",
    "encoder_profile",
    "user_id",
    "session_id",
    "filename",
    "source_type",
    "page_number",
    "upload_time",
    "ingest_mode",
    "parser_used",
    "ocr_provider",
)


def payload_value_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def build_qdrant_payload(chunk_or_payload: Any, profile: str = "full") -> Dict[str, Any]:
    if hasattr(chunk_or_payload, "to_dict"):
        payload = dict(chunk_or_payload.to_dict())
    else:
        payload = dict(chunk_or_payload or {})

    normalized_profile = str(profile or "full").strip().lower()
    if normalized_profile in {"full", "raw"}:
        return payload
    if normalized_profile not in {"lean", "serving"}:
        raise ValueError(f"Unsupported Qdrant payload profile: {profile}")

    doc_id = str(payload.get("doc_id") or payload.get("van_ban_id") or "").strip()
    if doc_id:
        payload["doc_id"] = doc_id
        if not str(payload.get("van_ban_id") or "").strip():
            payload["van_ban_id"] = doc_id
        if not str(payload.get("so_ky_hieu") or "").strip():
            payload["so_ky_hieu"] = doc_id

    legal_path = str(payload.get("legal_path") or payload.get("path") or "").strip()
    if legal_path:
        payload["legal_path"] = legal_path

    if not str(payload.get("co_quan") or "").strip() and str(payload.get("co_quan_ban_hanh") or "").strip():
        payload["co_quan"] = payload.get("co_quan_ban_hanh")
    if not str(payload.get("co_quan_ban_hanh") or "").strip() and str(payload.get("co_quan") or "").strip():
        payload["co_quan_ban_hanh"] = payload.get("co_quan")

    lean_payload: Dict[str, Any] = {}
    for field_name in LEAN_QDRANT_PAYLOAD_FIELDS:
        value = payload.get(field_name)
        if not payload_value_is_present(value):
            continue
        if isinstance(value, str):
            value = value.strip()
        lean_payload[field_name] = value

    return lean_payload
