from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from dotenv import load_dotenv


LAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_DIR.parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "gold" / "vbpl_gold.parquet"

MODEL_ID = "Quockhanh05/Vietnam_legal_embeddings"
COLLECTION_BASE = "legal_documents_lab"
COLLECTION_NAME = "legal_documents_lab__quockhanh05_vietnam_legal_embeddings__d768"
DENSE_DIM = 768
MAX_SEQ_LENGTH = 256
TARGET_CHUNK_TOKENS = 220
HARD_CAP_TOKENS = 256
OVERLAP_TOKENS = 32
NORMALIZE_EMBEDDINGS = True

BM25_PATH = (
    PROJECT_ROOT
    / "data"
    / "qdrant_db"
    / f"bm25__{COLLECTION_NAME}.json"
)

DOMAIN_ORDER: Tuple[str, ...] = (
    "hinh_su",
    "giao_thong",
    "moi_truong",
    "bien_hang_hai",
    "lao_dong",
)

DOMAIN_LABELS: Dict[str, str] = {
    "hinh_su": "Hinh su",
    "giao_thong": "Giao thong",
    "moi_truong": "Moi truong",
    "bien_hang_hai": "Bien / hang hai",
    "lao_dong": "Lao dong",
}

DOMAIN_RULES: Dict[str, Tuple[str, ...]] = {
    "hinh_su": (
        "bo luat hinh su",
        "to tung hinh su",
        "thi hanh an hinh su",
        "dieu tra hinh su",
    ),
    "giao_thong": (
        "giao thong duong bo",
        "trat tu an toan giao thong duong bo",
        "duong sat",
        "van tai duong bo",
    ),
    "moi_truong": (
        "bao ve moi truong",
        "moi truong",
    ),
    "bien_hang_hai": (
        "luat bien",
        "hang hai",
        "canh sat bien",
        "bien phong",
        "tai nguyen moi truong bien",
        "thuy san",
    ),
    "lao_dong": (
        "bo luat lao dong",
        "luat lao dong",
        "nguoi lao dong",
        "viec lam",
        "bao hiem xa hoi",
    ),
}


def normalize_for_match(value: object) -> str:
    text = str(value or "").lower()
    text = text.replace("đ", "d").replace("Đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


def matched_domains(text: str) -> List[str]:
    normalized = normalize_for_match(text)
    matches: List[str] = []
    for domain in DOMAIN_ORDER:
        if any(pattern in normalized for pattern in DOMAIN_RULES[domain]):
            matches.append(domain)
    return matches


def apply_lab_environment(
    *,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    require_qdrant: bool = True,
) -> tuple[str, str]:
    load_dotenv(PROJECT_ROOT / ".env")

    resolved_url = (
        qdrant_url
        or os.getenv("QDRANT_CLOUD_URL")
        or os.getenv("QDRANT_URL")
        or ""
    ).strip()
    resolved_key = (qdrant_api_key or os.getenv("QDRANT_API_KEY") or "").strip()

    if require_qdrant and not resolved_url:
        raise RuntimeError("Missing QDRANT_CLOUD_URL or QDRANT_URL for the lab.")
    if require_qdrant and not resolved_key:
        raise RuntimeError("Missing QDRANT_API_KEY for the lab.")

    if resolved_url:
        os.environ["QDRANT_CLOUD_URL"] = resolved_url
    if resolved_key:
        os.environ["QDRANT_API_KEY"] = resolved_key

    os.environ["QDRANT_USE_CLOUD"] = "true"
    os.environ["QDRANT_COLLECTION"] = COLLECTION_NAME
    os.environ["QDRANT_COLLECTION_BY_MODEL"] = "false"
    os.environ["QDRANT_DENSE_DIM"] = str(DENSE_DIM)
    os.environ["EMBEDDING_MODEL"] = MODEL_ID
    os.environ["EMBEDDING_NORMALIZE"] = "true" if NORMALIZE_EMBEDDINGS else "false"

    return resolved_url, resolved_key


def domain_counts(domains_by_row: Iterable[Sequence[str]]) -> Dict[str, int]:
    counts = {domain: 0 for domain in DOMAIN_ORDER}
    for domains in domains_by_row:
        for domain in domains:
            if domain in counts:
                counts[domain] += 1
    return counts
