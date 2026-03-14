"""
Data normalization and incremental state utilities for Vietnamese legal corpora.

This module is intentionally rule-based and deterministic to support
reproducible ingestion and graph extraction without LLM dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


DOC_TYPE_HIERARCHY: List[str] = [
    "LUAT",
    "PHAP_LENH",
    "NGHI_DINH",
    "THONG_TU",
    "QUY_CHE",
]


_DOC_TYPE_KEYWORDS: List[Tuple[str, str]] = [
    ("BO LUAT", "LUAT"),
    ("LUAT", "LUAT"),
    ("PHAP LENH", "PHAP_LENH"),
    ("NGHI DINH", "NGHI_DINH"),
    ("THONG TU", "THONG_TU"),
    ("QUY CHE", "QUY_CHE"),
]


_LAW_FAMILY_PATTERN = re.compile(
    r"(B[\u1ed9o]\s*lu[\u1eada]t[^.;,\n]*|L[u\u1ead]t[^.;,\n]*|Ph[\u00e1a]p\s*l[\u1ec7e]nh[^.;,\n]*|Ngh[\u1ecbi]\s*\u0111[\u1ecbi]nh[^.;,\n]*|Th[\u00f4o]ng\s*t[\u01b0u][^.;,\n]*|Quy\s*ch[\u1ebfe][^.;,\n]*)",
    re.IGNORECASE,
)

_DOC_TITLE_TOKEN_PATTERN = re.compile(
    r"\b(BO LUAT|LUAT|PHAP LENH|NGHI DINH|THONG TU|QUY CHE)\b"
)

_PREAMBLE_MARKERS: List[str] = [
    " CAN CU ",
    " CHUONG I",
    " DIEU 1",
    " QUOC HOI BAN HANH",
    " CHINH PHU BAN HANH",
    " UY BAN THUONG VU QUOC HOI BAN HANH",
]

_LOW_CONFIDENCE_MARKERS: List[str] = [
    "CAN CU",
    "CHUONG I",
    "DIEU 1",
    "CLIP_IMAGE",
]

_TRAILING_ISSUER_MARKERS: List[str] = [
    " CHINH PHU",
    " QUOC HOI",
    " UY BAN THUONG VU QUOC HOI",
    " THU TUONG CHINH PHU",
    " BO TRUONG",
]


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and value != value


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_legal_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def strip_accents(text: str) -> str:
    text = (text or "").replace("\u0111", "d").replace("\u0110", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_doc_id(value: Any) -> str:
    if value is None or _is_nan(value):
        return "UNKNOWN"
    text = str(value).replace("\u00a0", " ")
    text = normalize_whitespace(text).replace(" ", "")
    text = re.sub(r"/+", "/", text)
    return text.upper() if text else "UNKNOWN"


def parse_legal_date(value: Any) -> Optional[str]:
    """
    Parse common legal date formats and return ISO date string (YYYY-MM-DD).
    """
    if value is None or _is_nan(value):
        return None

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    text = normalize_whitespace(str(value))
    if not text:
        return None

    text = text.split(" ")[0]
    compact = re.sub(r"[^\d]", "", text)
    if len(compact) == 8:
        for fmt in ("%Y%m%d", "%d%m%Y", "%m%d%Y"):
            try:
                return datetime.strptime(compact, fmt).date().isoformat()
            except ValueError:
                pass

    text = text.replace(".", "/").replace("-", "/")
    text = re.sub(r"[^\d/]", "", text)

    fmts = ("%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%y")
    for fmt in fmts:
        try:
            parsed = datetime.strptime(text, fmt).date()
            return parsed.isoformat()
        except ValueError:
            continue

    return None


def is_document_active(
    ngay_het_hieu_luc: Optional[str],
    as_of_date: Optional[date] = None,
) -> bool:
    if as_of_date is None:
        as_of_date = date.today()
    if not ngay_het_hieu_luc:
        return True
    try:
        expiry = datetime.strptime(ngay_het_hieu_luc, "%Y-%m-%d").date()
    except ValueError:
        return True
    return expiry > as_of_date


def compute_expiry_days(
    ngay_het_hieu_luc: Optional[str],
    as_of_date: Optional[date] = None,
) -> Optional[int]:
    if not ngay_het_hieu_luc:
        return None
    if as_of_date is None:
        as_of_date = date.today()
    try:
        expiry = datetime.strptime(ngay_het_hieu_luc, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (expiry - as_of_date).days


def compute_status_label(
    is_active: bool,
    expires_in_days: Optional[int],
    monitor_expiring_days: int = 90,
) -> str:
    if not is_active:
        return "inactive"
    if expires_in_days is not None and 0 <= expires_in_days <= monitor_expiring_days:
        return "expiring_soon"
    return "active"


def normalize_doc_type(loai_van_ban: Any) -> Tuple[str, int]:
    text = normalize_whitespace(str(loai_van_ban or ""))
    ascii_upper = strip_accents(text).upper()

    for keyword, canonical in _DOC_TYPE_KEYWORDS:
        if keyword in ascii_upper:
            return canonical, DOC_TYPE_HIERARCHY.index(canonical) + 1

    return "OTHER", len(DOC_TYPE_HIERARCHY) + 1


def derive_law_family(
    ten_van_ban: str,
    loai_van_ban: str,
    so_ky_hieu: str,
) -> str:
    text = normalize_whitespace(ten_van_ban)
    if text:
        match = _LAW_FAMILY_PATTERN.search(text)
        if match:
            return normalize_whitespace(match.group(1))
        return text

    loai = normalize_whitespace(loai_van_ban) or "Van ban"
    so = normalize_whitespace(so_ky_hieu) or "khong ro so hieu"
    return f"{loai} {so}"


def build_citation_base(
    loai_van_ban: str,
    so_ky_hieu: str,
    ten_van_ban: str,
) -> str:
    loai = normalize_whitespace(loai_van_ban)
    so = normalize_whitespace(so_ky_hieu)
    ten = normalize_whitespace(ten_van_ban)

    parts: List[str] = []
    if loai:
        parts.append(loai)
    if so:
        parts.append(so)
    base = " ".join(parts).strip()
    if ten:
        return f"{base} - {ten}" if base else ten
    return base or "Van ban phap luat"


def build_citation_full(citation_base: str, legal_path: str) -> str:
    base = normalize_whitespace(citation_base)
    path = normalize_whitespace(legal_path)
    if path and path.lower() != "document":
        return f"{base} | {path}" if base else path
    return base or "Van ban phap luat"


def _clip_title(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars + 1]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" -_;:,./")


def _find_title_start(ascii_text: str) -> Optional[int]:
    fallback = None
    for match in _DOC_TITLE_TOKEN_PATTERN.finditer(ascii_text):
        if fallback is None:
            fallback = match.start()
        tail = ascii_text[match.start() : match.start() + 48]
        if re.match(r"(BO LUAT|LUAT|PHAP LENH|NGHI DINH|THONG TU|QUY CHE)\s+SO\b", tail):
            continue
        if re.match(r"(BO LUAT|LUAT|PHAP LENH|NGHI DINH|THONG TU|QUY CHE)\s*:\s*\d", tail):
            continue
        return match.start()
    return fallback


def clean_serving_title(
    ten_van_ban: str,
    loai_van_ban: str = "",
    so_ky_hieu: str = "",
    max_chars: int = 180,
) -> Tuple[str, List[str], bool]:
    raw = normalize_whitespace(str(ten_van_ban or ""))
    flags: List[str] = []
    if not raw:
        fallback = normalize_whitespace(" ".join([str(loai_van_ban or ""), str(so_ky_hieu or "")]))
        return fallback or "Van ban phap luat", ["empty_title"], False

    text = str(raw).replace("\u00a0", " ")
    text = re.sub(r"\d+/\d+/clip_image\d+\.\s*\w+\"?\s*(?:width|height)?=?\"?\d*\"?\s*/?>?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"clip_image\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[_]{2,}", " ", text)
    text = re.sub(r"(?:-\s*){4,}", " ", text)
    text = normalize_whitespace(text)
    if text != raw:
        flags.append("removed_ocr_noise")

    ascii_text = strip_accents(text).upper()
    start = _find_title_start(ascii_text)
    if start is not None and start > 0:
        text = text[start:].lstrip(" -_:;,./")
        ascii_text = strip_accents(text).upper()
        flags.append("dropped_header_prefix")

    cut_positions = [idx for marker in _PREAMBLE_MARKERS if (idx := ascii_text.find(marker)) >= 0]
    if cut_positions:
        text = text[: min(cut_positions)].rstrip(" -_:;,./")
        ascii_text = strip_accents(text).upper()
        flags.append("trimmed_preamble_tail")

    if ";" in text:
        semi_idx = text.find(";")
        if semi_idx >= 24:
            text = text[:semi_idx].rstrip(" -_:;,./")
            ascii_text = strip_accents(text).upper()
            flags.append("trimmed_semicolon_tail")

    for marker in _TRAILING_ISSUER_MARKERS:
        if ascii_text.endswith(marker):
            trim_chars = len(marker)
            text = text[:-trim_chars].rstrip(" -_:;,./")
            ascii_text = strip_accents(text).upper()
            flags.append("trimmed_trailing_issuer")
            break

    text = normalize_whitespace(text)
    if len(text) > max_chars:
        text = _clip_title(text, max_chars=max_chars)
        ascii_text = strip_accents(text).upper()
        flags.append("title_truncated")

    has_doc_type = bool(_DOC_TITLE_TOKEN_PATTERN.search(ascii_text))
    has_low_confidence_marker = any(marker in ascii_text for marker in _LOW_CONFIDENCE_MARKERS)
    low_confidence = (len(text) < 12) or (not has_doc_type) or has_low_confidence_marker
    if low_confidence:
        flags.append("low_confidence_title")

    return text or raw, list(dict.fromkeys(flags)), not low_confidence


def compute_document_hash(record: Dict[str, Any]) -> str:
    payload = {
        "so_ky_hieu": record.get("so_ky_hieu", ""),
        "ten_van_ban": record.get("ten_van_ban", ""),
        "loai_van_ban": record.get("loai_van_ban", ""),
        "co_quan_ban_hanh": record.get("co_quan_ban_hanh", ""),
        "nguoi_ky": record.get("nguoi_ky", ""),
        "ngay_ban_hanh": record.get("ngay_ban_hanh", ""),
        "ngay_co_hieu_luc": record.get("ngay_co_hieu_luc", ""),
        "ngay_het_hieu_luc": record.get("ngay_het_hieu_luc", ""),
        "toan_van_clean": record.get("toan_van_clean", ""),
        "mo_ta": record.get("mo_ta", ""),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def normalize_document_row(
    row: Dict[str, Any],
    as_of_date: Optional[date] = None,
    monitor_expiring_days: int = 90,
) -> Dict[str, Any]:
    so_ky_hieu_raw = row.get("so_ky_hieu") or row.get("van_ban_id")
    so_ky_hieu = normalize_whitespace(str(so_ky_hieu_raw or ""))
    doc_id = normalize_doc_id(so_ky_hieu_raw)

    ten_van_ban = normalize_whitespace(str(row.get("ten_van_ban") or ""))
    loai_van_ban = normalize_whitespace(str(row.get("loai_van_ban") or ""))
    co_quan_ban_hanh = normalize_whitespace(
        str(row.get("co_quan_ban_hanh") or row.get("co_quan") or "")
    )
    chuc_danh = normalize_whitespace(str(row.get("chuc_danh") or ""))
    nguoi_ky = normalize_whitespace(str(row.get("nguoi_ky") or ""))
    mo_ta = normalize_whitespace(str(row.get("mo_ta") or ""))

    ngay_ban_hanh = parse_legal_date(row.get("ngay_ban_hanh"))
    ngay_co_hieu_luc = parse_legal_date(row.get("ngay_co_hieu_luc"))
    ngay_het_hieu_luc = parse_legal_date(row.get("ngay_het_hieu_luc"))
    active = is_document_active(ngay_het_hieu_luc, as_of_date=as_of_date)
    expires_in_days = compute_expiry_days(ngay_het_hieu_luc, as_of_date=as_of_date)
    status_label = compute_status_label(
        is_active=active,
        expires_in_days=expires_in_days,
        monitor_expiring_days=monitor_expiring_days,
    )

    text_clean = row.get("toan_van_clean")
    if text_clean is None or _is_nan(text_clean) or not str(text_clean).strip():
        text_clean = row.get("toan_van") or row.get("clean_text") or row.get("noi_dung") or ""
    toan_van_clean = normalize_legal_text(str(text_clean))

    if not ten_van_ban and toan_van_clean:
        # Fallback title extraction for datasets without explicit title field.
        candidate = re.split(r"\b(?:\u0110i\u1ec1u|Dieu)\b", toan_van_clean, maxsplit=1)[0].strip()
        ten_van_ban = normalize_whitespace(candidate)[:300]

    cleaned_title, _, title_confident = clean_serving_title(
        ten_van_ban=ten_van_ban,
        loai_van_ban=loai_van_ban,
        so_ky_hieu=so_ky_hieu,
    )
    if title_confident and cleaned_title:
        ten_van_ban = cleaned_title

    doc_type, doc_type_rank = normalize_doc_type(loai_van_ban)
    law_family = derive_law_family(ten_van_ban, loai_van_ban, so_ky_hieu)
    citation_base = build_citation_base(loai_van_ban, so_ky_hieu, ten_van_ban)

    normalized = {
        "doc_id": doc_id,
        "so_ky_hieu": so_ky_hieu,
        "van_ban_id": doc_id,  # compatibility alias
        "ten_van_ban": ten_van_ban,
        "loai_van_ban": loai_van_ban,
        "doc_type": doc_type,
        "doc_type_rank": doc_type_rank,
        "co_quan_ban_hanh": co_quan_ban_hanh,
        "co_quan": co_quan_ban_hanh,  # compatibility alias
        "chuc_danh": chuc_danh,
        "nguoi_ky": nguoi_ky,
        "ngay_ban_hanh": ngay_ban_hanh,
        "ngay_co_hieu_luc": ngay_co_hieu_luc,
        "ngay_het_hieu_luc": ngay_het_hieu_luc,
        "expires_in_days": expires_in_days,
        "is_active": active,
        "status_label": status_label,
        "toan_van_clean": toan_van_clean,
        "toan_van": normalize_legal_text(str(row.get("toan_van") or "")),
        "mo_ta": mo_ta,
        "law_family": law_family,
        "citation_base": citation_base,
    }
    normalized["content_hash"] = compute_document_hash(normalized)
    return normalized


def normalize_documents_dataframe(df, as_of_date: Optional[date] = None, monitor_expiring_days: int = 90):
    records = [
        normalize_document_row(
            row.to_dict(),
            as_of_date=as_of_date,
            monitor_expiring_days=monitor_expiring_days,
        )
        for _, row in df.iterrows()
    ]
    try:
        import pandas as pd

        return pd.DataFrame(records)
    except Exception:
        return records


@dataclass
class IngestionStateDiff:
    to_update: List[Dict[str, Any]]
    unchanged: List[Dict[str, Any]]


class IngestionStateStore:
    """
    Lightweight JSON state tracker for incremental ingestion.
    """

    def __init__(self, state_path: str):
        self.state_path = state_path
        directory = os.path.dirname(state_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def load(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.state_path):
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def save(self, state: Dict[str, Dict[str, Any]]) -> None:
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def diff(self, records: Iterable[Dict[str, Any]]) -> IngestionStateDiff:
        state = self.load()
        to_update: List[Dict[str, Any]] = []
        unchanged: List[Dict[str, Any]] = []

        for record in records:
            doc_id = record.get("doc_id", "UNKNOWN")
            current_hash = record.get("content_hash")
            current_active = bool(record.get("is_active", True))
            current_status = record.get("status_label", "active")

            previous = state.get(doc_id)
            if previous is None:
                to_update.append(record)
                continue

            if previous.get("content_hash") != current_hash:
                to_update.append(record)
                continue

            if bool(previous.get("is_active", True)) != current_active:
                to_update.append(record)
                continue

            if previous.get("status_label", "active") != current_status:
                to_update.append(record)
                continue

            unchanged.append(record)

        return IngestionStateDiff(to_update=to_update, unchanged=unchanged)

    def update_state(
        self,
        records: Iterable[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        state = self.load()
        now = datetime.now(timezone.utc).isoformat()

        for record in records:
            doc_id = record.get("doc_id", "UNKNOWN")
            state[doc_id] = {
                "content_hash": record.get("content_hash"),
                "is_active": bool(record.get("is_active", True)),
                "status_label": record.get("status_label", "active"),
                "expires_in_days": record.get("expires_in_days"),
                "doc_type": record.get("doc_type", "OTHER"),
                "doc_type_rank": record.get("doc_type_rank"),
                "updated_at": now,
            }

        self.save(state)
        return state
