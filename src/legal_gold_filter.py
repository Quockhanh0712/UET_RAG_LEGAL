"""
Filtering and soft-ranking utilities to build a compact legal gold dataset.

The module supports two modes:
1) `filter_gold(...)` legacy hard filtering.
2) `build_soft_gold_sets(...)` profile-driven soft filtering + primary/archive split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.legal_data_processing import normalize_whitespace, strip_accents

logger = logging.getLogger(__name__)


SEED_LAW_FAMILIES = [
    "Bo luat Dan su",
    "Bo luat Hinh su",
    "Bo luat To tung hinh su",
    "Luat Doanh nghiep",
    "Luat Dau tu",
    "Luat Lao dong",
    "Bo luat Lao dong",
    "Luat Dat dai",
    "Luat Nha o",
    "Luat Kinh doanh bat dong san",
    "Luat Giao thong duong bo",
    "Luat Thue thu nhap ca nhan",
    "Luat Thue thu nhap doanh nghiep",
    "Luat Bao hiem xa hoi",
    "Luat Bao hiem y te",
    "Luat Xuat nhap canh",
    "Phap lenh Xu ly vi pham hanh chinh",
]

GUIDANCE_HINTS = [
    "huong dan",
    "quy dinh chi tiet",
    "thi hanh",
    "sua doi",
    "bo sung",
    "thay the",
    "bai bo",
]

DEFAULT_ALLOWED_DOC_TYPES = {"LUAT", "PHAP_LENH", "NGHI_DINH", "THONG_TU", "QUY_CHE"}
SEED_KEYWORDS = [strip_accents(x).upper() for x in SEED_LAW_FAMILIES]

TYPE_SCORE_MAP = {
    "LUAT": 6,
    "PHAP_LENH": 5,
    "NGHI_DINH": 3,
    "THONG_TU": 2,
    "QUY_CHE": 1,
    "OTHER": 0,
}


@dataclass
class SoftGoldConfig:
    allowed_doc_types: Sequence[str] = tuple(DEFAULT_ALLOWED_DOC_TYPES)
    min_year_soft: int = 2010
    target_min_laws: int = 40
    backfill_min_year: int = 2000
    max_primary_records: int = 2000
    include_inactive_primary: bool = False
    drop_missing_all_dates: bool = True
    today: Optional[date] = None


@dataclass
class SoftGoldResult:
    primary_records: List[Dict[str, str]]
    archive_records: List[Dict[str, str]]
    stats: Dict[str, int]


def _key(text: str) -> str:
    return strip_accents(normalize_whitespace(text or "")).upper()


def _parse_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None


def _year(date_str: str) -> Optional[int]:
    parsed = _parse_date(date_str)
    return parsed.year if parsed else None


def _has_any_date(record: Dict[str, str]) -> bool:
    return bool(record.get("ngay_ban_hanh") or record.get("ngay_co_hieu_luc") or record.get("ngay_het_hieu_luc"))


def _date_quality_score(record: Dict[str, str]) -> int:
    score = 0
    if record.get("ngay_ban_hanh"):
        score += 1
    if record.get("ngay_co_hieu_luc"):
        score += 1
    if record.get("ngay_het_hieu_luc"):
        score += 1
    return score


def is_seed_family(law_family: str) -> bool:
    family = _key(law_family)
    return any(seed in family for seed in SEED_KEYWORDS)


def _contains_seed_keywords(text: str) -> bool:
    content = _key(text)
    return any(seed in content for seed in SEED_KEYWORDS)


def _contains_guidance_hint(text: str) -> bool:
    content = _key(text).lower()
    return any(hint in content for hint in GUIDANCE_HINTS)


def is_seed_related(record: Dict[str, str]) -> bool:
    """
    Seed relevance rules (accent-insensitive):
    - law_family contains seed keyword, or
    - title/summary contains seed keyword.
    """
    if is_seed_family(record.get("law_family", "")):
        return True
    title = record.get("ten_van_ban", "")
    desc = record.get("mo_ta", "")
    return _contains_seed_keywords(title) or _contains_seed_keywords(desc)


def has_relation_signal(record: Dict[str, str]) -> bool:
    title = record.get("ten_van_ban", "")
    desc = record.get("mo_ta", "")
    return _contains_guidance_hint(title) or _contains_guidance_hint(desc)


def _score_rec(rec: Dict[str, str]) -> Tuple[int, datetime, datetime]:
    active_score = 1 if rec.get("is_active") else 0
    effective = _parse_date(rec.get("ngay_co_hieu_luc")) or datetime(1900, 1, 1)
    issued = _parse_date(rec.get("ngay_ban_hanh")) or datetime(1900, 1, 1)
    return active_score, effective, issued


def dedup_records(records: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    by_doc: Dict[str, Dict[str, str]] = {}
    for rec in records:
        doc_id = rec.get("doc_id", "")
        if not doc_id:
            continue
        candidate = by_doc.get(doc_id)
        if candidate is None or _score_rec(rec) > _score_rec(candidate):
            by_doc[doc_id] = rec

    seen_hash = set()
    unique: List[Dict[str, str]] = []
    for rec in by_doc.values():
        content_hash = rec.get("content_hash")
        if content_hash and content_hash in seen_hash:
            continue
        if content_hash:
            seen_hash.add(content_hash)
        unique.append(rec)
    return unique


def _priority_score(record: Dict[str, str], cfg: SoftGoldConfig) -> int:
    doc_type = str(record.get("doc_type", "OTHER"))
    is_law = doc_type in {"LUAT", "PHAP_LENH"}
    seed_related = bool(record.get("seed_related", False))
    relation_signal = bool(record.get("relation_signal", False))
    is_active = bool(record.get("is_active", True))

    type_score = TYPE_SCORE_MAP.get(doc_type, 0)
    seed_score = 4 if seed_related else (1 if is_law else 0)

    if doc_type in {"NGHI_DINH", "THONG_TU", "QUY_CHE"}:
        if seed_related and relation_signal:
            relation_score = 3
        elif seed_related:
            relation_score = 1
        else:
            relation_score = -2
    else:
        relation_score = 0

    year = _year(record.get("ngay_ban_hanh", ""))
    if year is None:
        recency_score = 0
    elif year >= cfg.min_year_soft + 5:
        recency_score = 3
    elif year >= cfg.min_year_soft:
        recency_score = 2
    elif year >= cfg.backfill_min_year:
        recency_score = 1
    else:
        recency_score = -2

    active_score = 2 if is_active else -2
    date_quality_score = _date_quality_score(record)
    return type_score + seed_score + relation_score + recency_score + active_score + date_quality_score


def _sort_key(record: Dict[str, str]) -> Tuple[int, int, int, int]:
    year = _year(record.get("ngay_ban_hanh", "")) or 0
    is_law_seed = 1 if (record.get("doc_type") in {"LUAT", "PHAP_LENH"} and record.get("seed_related")) else 0
    is_law = 1 if record.get("doc_type") in {"LUAT", "PHAP_LENH"} else 0
    return (is_law_seed, is_law, int(record.get("priority_score", 0)), year)


def _is_primary_candidate(record: Dict[str, str], cfg: SoftGoldConfig) -> bool:
    if not cfg.include_inactive_primary and not record.get("is_active", True):
        return False

    doc_type = str(record.get("doc_type", "OTHER"))
    year = _year(record.get("ngay_ban_hanh", ""))
    seed_related = bool(record.get("seed_related", False))
    relation_signal = bool(record.get("relation_signal", False))

    if doc_type in {"LUAT", "PHAP_LENH"}:
        if seed_related:
            return True
        return bool(year and year >= cfg.min_year_soft)

    if doc_type in {"NGHI_DINH", "THONG_TU", "QUY_CHE"}:
        if not seed_related:
            return False
        if year and year >= cfg.min_year_soft:
            return True
        return relation_signal

    return False


def _law_count(records: Sequence[Dict[str, str]]) -> int:
    return sum(1 for rec in records if rec.get("doc_type") in {"LUAT", "PHAP_LENH"})


def build_soft_gold_sets(
    records: Iterable[Dict[str, str]],
    config: SoftGoldConfig,
) -> SoftGoldResult:
    today = config.today or date.today()
    allowed_types = {x.upper() for x in config.allowed_doc_types}

    stats = {
        "input_count": 0,
        "dropped_doc_type": 0,
        "dropped_no_date": 0,
        "dropped_expired": 0,
        "candidate_count": 0,
        "dedup_count": 0,
        "primary_count": 0,
        "archive_count": 0,
        "primary_law_count": 0,
        "backfill_added_laws": 0,
    }

    candidates: List[Dict[str, str]] = []
    for raw in records:
        stats["input_count"] += 1
        record = dict(raw)
        doc_type = str(record.get("doc_type", "OTHER")).upper()

        if doc_type not in allowed_types:
            stats["dropped_doc_type"] += 1
            continue

        if config.drop_missing_all_dates and not _has_any_date(record):
            stats["dropped_no_date"] += 1
            continue

        ngay_het = record.get("ngay_het_hieu_luc")
        if ngay_het:
            parsed = _parse_date(ngay_het)
            if parsed and parsed.date() <= today:
                # Keep expired records for archive only; do not drop entirely.
                record["is_active"] = False
                stats["dropped_expired"] += 1

        record["seed_related"] = is_seed_related(record)
        record["relation_signal"] = has_relation_signal(record)
        record["priority_score"] = _priority_score(record, config)
        candidates.append(record)

    stats["candidate_count"] = len(candidates)
    deduped = dedup_records(candidates)
    stats["dedup_count"] = len(deduped)

    primary = [rec for rec in deduped if _is_primary_candidate(rec, config)]
    primary.sort(key=_sort_key, reverse=True)

    if config.max_primary_records > 0 and len(primary) > config.max_primary_records:
        primary = primary[: config.max_primary_records]

    primary_ids = {rec.get("doc_id") for rec in primary}
    law_count = _law_count(primary)

    # Soft target + auto-backfill of older law/phap lenh docs.
    if law_count < config.target_min_laws:
        law_backfill_pool = []
        for rec in deduped:
            if rec.get("doc_id") in primary_ids:
                continue
            if rec.get("doc_type") not in {"LUAT", "PHAP_LENH"}:
                continue
            if not rec.get("is_active", True) and not config.include_inactive_primary:
                continue
            year = _year(rec.get("ngay_ban_hanh", ""))
            if year is None or year < config.backfill_min_year:
                continue
            law_backfill_pool.append(rec)

        law_backfill_pool.sort(key=_sort_key, reverse=True)
        for rec in law_backfill_pool:
            if law_count >= config.target_min_laws:
                break
            if config.max_primary_records <= 0 or len(primary) < config.max_primary_records:
                primary.append(rec)
                primary_ids.add(rec.get("doc_id"))
                law_count += 1
                stats["backfill_added_laws"] += 1
                continue

            # Replace lowest-ranked non-law item when primary is full.
            non_law_indices = [
                idx for idx, item in enumerate(primary) if item.get("doc_type") not in {"LUAT", "PHAP_LENH"}
            ]
            if not non_law_indices:
                break
            replace_idx = min(non_law_indices, key=lambda idx: _sort_key(primary[idx]))
            primary[replace_idx] = rec
            primary_ids.add(rec.get("doc_id"))
            law_count += 1
            stats["backfill_added_laws"] += 1

        primary.sort(key=_sort_key, reverse=True)

    archive = [rec for rec in deduped if rec.get("doc_id") not in primary_ids]
    archive.sort(key=_sort_key, reverse=True)

    stats["primary_count"] = len(primary)
    stats["archive_count"] = len(archive)
    stats["primary_law_count"] = _law_count(primary)

    logger.info(
        "[SOFT_FILTER] input=%s candidate=%s dedup=%s primary=%s archive=%s primary_laws=%s backfill_laws=%s dropped_doc_type=%s dropped_no_date=%s",
        stats["input_count"],
        stats["candidate_count"],
        stats["dedup_count"],
        stats["primary_count"],
        stats["archive_count"],
        stats["primary_law_count"],
        stats["backfill_added_laws"],
        stats["dropped_doc_type"],
        stats["dropped_no_date"],
    )
    return SoftGoldResult(primary_records=primary, archive_records=archive, stats=stats)


def filter_gold(
    records: Iterable[Dict[str, str]],
    drop_missing_all_dates: bool = True,
    include_inactive: bool = False,
    allowed_doc_types: Optional[Sequence[str]] = None,
    min_year: int = 2015,
    today: Optional[date] = None,
) -> List[Dict[str, str]]:
    """
    Legacy hard-filter API kept for compatibility.
    """
    today = today or date.today()
    allowed_types = set(allowed_doc_types or DEFAULT_ALLOWED_DOC_TYPES)
    kept: List[Dict[str, str]] = []
    input_count = 0
    dropped_inactive = 0
    dropped_no_date = 0
    dropped_non_seed = 0
    dropped_doc_type = 0
    dropped_old = 0
    dropped_expired = 0

    for rec in records:
        input_count += 1
        if not include_inactive and not rec.get("is_active", True):
            dropped_inactive += 1
            continue

        ngay_het = rec.get("ngay_het_hieu_luc")
        if ngay_het:
            try:
                if datetime.strptime(ngay_het, "%Y-%m-%d").date() <= today:
                    dropped_expired += 1
                    continue
            except Exception:
                pass

        if drop_missing_all_dates and not _has_any_date(rec):
            dropped_no_date += 1
            continue

        ngay_ban_hanh = rec.get("ngay_ban_hanh")
        if ngay_ban_hanh:
            try:
                year = datetime.strptime(ngay_ban_hanh, "%Y-%m-%d").year
                if year < min_year:
                    dropped_old += 1
                    continue
            except Exception:
                pass

        doc_type = str(rec.get("doc_type", "OTHER"))
        if doc_type not in allowed_types:
            dropped_doc_type += 1
            continue

        if not is_seed_related(rec):
            dropped_non_seed += 1
            continue

        kept.append(dict(rec))

    deduped = dedup_records(kept)
    logger.info(
        "[FILTER] input=%s kept=%s deduped=%s dropped_inactive=%s dropped_expired=%s dropped_old=%s dropped_no_date=%s dropped_non_seed=%s dropped_doc_type=%s",
        input_count,
        len(kept),
        len(deduped),
        dropped_inactive,
        dropped_expired,
        dropped_old,
        dropped_no_date,
        dropped_non_seed,
        dropped_doc_type,
    )
    return deduped
