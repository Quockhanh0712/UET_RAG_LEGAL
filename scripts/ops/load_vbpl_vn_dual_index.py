"""
Ingest VietTung04/vbpl-vn into:
1) Gold dataset (document-level normalized + filtered)
2) Qdrant (hybrid search chunks, optional)
3) Neo4j (rule-based legal graph, optional)

This script is deterministic and does not use LLM for extraction.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

WEB_SOURCE_SEEDS = [
    {"name": "VBPL", "url": "https://vbpl.vn", "purpose": "official legal lookup"},
    {"name": "Cong Bao", "url": "https://congbao.chinhphu.vn", "purpose": "gazette verification"},
    {"name": "Chinh Phu", "url": "https://chinhphu.vn", "purpose": "government decrees and circulars"},
    {"name": "BTP", "url": "https://moj.gov.vn", "purpose": "legal ministry references"},
    {"name": "Thu Vien Phap Luat", "url": "https://thuvienphapluat.vn", "purpose": "cross-reference titles and status"},
]

DEFAULT_GOLD_OUTPUT = PROJECT_ROOT / "data" / "gold" / "vbpl_gold.parquet"
DEFAULT_WEB_SOURCES_OUTPUT = PROJECT_ROOT / "data" / "gold" / "web_sources_seed.json"
DEFAULT_ALLOWED_DOC_TYPES = ("LUAT", "PHAP_LENH", "NGHI_DINH", "THONG_TU", "QUY_CHE")

PROFILE_SETTINGS = {
    "strict": {
        "min_year_soft": 2015,
        "target_min_laws": 30,
        "backfill_min_year": 2005,
        "max_primary_records": 1200,
    },
    "balanced": {
        "min_year_soft": 2010,
        "target_min_laws": 40,
        "backfill_min_year": 2000,
        "max_primary_records": 2000,
    },
    "broad": {
        "min_year_soft": 2005,
        "target_min_laws": 50,
        "backfill_min_year": 1995,
        "max_primary_records": 3500,
    },
}


def _parse_as_of_date(value: str) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("--as-of-date must be YYYY-MM-DD")


def _load_dataset(dataset_name: str, split: str):
    from datasets import load_dataset

    logger.info("[DATASET] Loading %s (%s)", dataset_name, split)
    dataset = load_dataset(dataset_name, split=split)
    df = dataset.to_pandas()
    logger.info("[DATASET] Loaded %s rows", len(df))
    return df


def _normalize_rows(df, as_of_date=None, monitor_expiring_days: int = 90) -> List[Dict]:
    from src.legal_data_processing import normalize_document_row

    records: List[Dict] = []
    for _, row in df.iterrows():
        normalized = normalize_document_row(
            row.to_dict(),
            as_of_date=as_of_date,
            monitor_expiring_days=monitor_expiring_days,
        )
        normalized["source"] = "VietTung04/vbpl-vn"
        if normalized.get("toan_van_clean"):
            records.append(normalized)
    logger.info("[NORMALIZE] Kept %s records with non-empty text", len(records))
    return records


def _filter_gold_records(
    normalized_records: Sequence[Dict],
    drop_missing_all_dates: bool,
    allowed_doc_types: Sequence[str],
    include_inactive_primary: bool,
    min_year_soft: int,
    target_min_laws: int,
    backfill_min_year: int,
    max_primary_records: int,
    today: Optional[date],
) -> Dict[str, object]:
    from src.legal_gold_filter import SoftGoldConfig, build_soft_gold_sets

    cfg = SoftGoldConfig(
        allowed_doc_types=allowed_doc_types,
        min_year_soft=min_year_soft,
        target_min_laws=target_min_laws,
        backfill_min_year=backfill_min_year,
        max_primary_records=max_primary_records,
        include_inactive_primary=include_inactive_primary,
        drop_missing_all_dates=drop_missing_all_dates,
        today=today,
    )
    result = build_soft_gold_sets(normalized_records, config=cfg)
    logger.info(
        "[GOLD] primary=%s archive=%s from normalized=%s",
        len(result.primary_records),
        len(result.archive_records),
        len(normalized_records),
    )
    return {
        "primary": result.primary_records,
        "archive": result.archive_records,
        "stats": result.stats,
    }


def _save_gold_records(records: Sequence[Dict], output_path: Path) -> None:
    if not records:
        logger.warning("[GOLD] No records to export")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame(records)
    ext = output_path.suffix.lower()
    if ext == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8")
    else:
        # Default to parquet. If parquet dependencies are missing, fall back to CSV.
        try:
            df.to_parquet(output_path, index=False)
        except Exception as exc:
            fallback = output_path.with_suffix(".csv")
            logger.warning("[GOLD] Could not write parquet (%s). Writing CSV: %s", exc, fallback)
            df.to_csv(fallback, index=False, encoding="utf-8")
            output_path = fallback
    logger.info("[GOLD] Exported document-level dataset: %s", output_path)


def _derive_archive_output(primary_output: Path) -> Path:
    suffix = primary_output.suffix.lower() or ".parquet"
    if suffix not in {".parquet", ".csv"}:
        suffix = ".parquet"
    return primary_output.with_name(f"{primary_output.stem}_archive{suffix}")


def _log_distribution(records: Sequence[Dict], label: str) -> None:
    doc_type_counts: Dict[str, int] = {}
    year_counts: Dict[int, int] = {}
    for rec in records:
        dt = rec.get("doc_type", "OTHER")
        doc_type_counts[dt] = doc_type_counts.get(dt, 0) + 1
        try:
            y = datetime.strptime(rec.get("ngay_ban_hanh", ""), "%Y-%m-%d").year
            year_counts[y] = year_counts.get(y, 0) + 1
        except Exception:
            pass
    logger.info("[GOLD] %s doc_type distribution: %s", label, doc_type_counts)
    if year_counts:
        logger.info("[GOLD] %s year tail: %s", label, dict(sorted(year_counts.items())[-5:]))


def _validate_primary_quality(
    records: Sequence[Dict],
    target_min_laws: int,
    recent_year_threshold: int = 2015,
    recent_ratio_threshold: float = 0.55,
) -> Dict[str, float]:
    total = len(records)
    law_count = sum(1 for rec in records if rec.get("doc_type") in {"LUAT", "PHAP_LENH"})
    law_ratio = (law_count / total) if total else 0.0

    recent_count = 0
    dated_count = 0
    for rec in records:
        try:
            y = datetime.strptime(rec.get("ngay_ban_hanh", ""), "%Y-%m-%d").year
            dated_count += 1
            if y >= recent_year_threshold:
                recent_count += 1
        except Exception:
            continue
    recent_ratio = (recent_count / dated_count) if dated_count else 0.0

    if law_count < target_min_laws:
        logger.warning("[QUALITY] primary law/phap lenh count=%s below target=%s", law_count, target_min_laws)
    if not (law_ratio >= 0.25 or law_count >= target_min_laws):
        logger.warning("[QUALITY] primary law ratio is low: %.2f%%", law_ratio * 100.0)
    if recent_ratio < recent_ratio_threshold:
        logger.warning(
            "[QUALITY] primary recent ratio %.2f%% below threshold %.2f%% (year>=%s)",
            recent_ratio * 100.0,
            recent_ratio_threshold * 100.0,
            recent_year_threshold,
        )

    return {
        "primary_total": total,
        "primary_law_count": law_count,
        "primary_law_ratio": law_ratio,
        "primary_recent_ratio": recent_ratio,
        "recent_year_threshold": recent_year_threshold,
    }


def _resolve_profile_settings(args) -> Dict[str, int]:
    profile = PROFILE_SETTINGS[args.profile]
    return {
        "min_year_soft": args.min_year_soft if args.min_year_soft is not None else profile["min_year_soft"],
        "target_min_laws": args.target_min_laws if args.target_min_laws is not None else profile["target_min_laws"],
        "backfill_min_year": args.backfill_min_year if args.backfill_min_year is not None else profile["backfill_min_year"],
        "max_primary_records": args.max_primary_records if args.max_primary_records is not None else profile["max_primary_records"],
    }


def _save_web_sources(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(WEB_SOURCE_SEEDS, f, ensure_ascii=False, indent=2)
    logger.info("[GOLD] Exported web-source seed list: %s", output_path)


def _chunk_records(records: Sequence[Dict], encoder_profile: str = "default") -> List:
    from src.legal_chunker_v2 import LegalChunkerV2

    chunker = LegalChunkerV2()
    all_chunks = []
    for idx, record in enumerate(records):
        record_with_profile = dict(record)
        record_with_profile["encoder_profile"] = encoder_profile
        chunks = chunker.chunk_document(record_with_profile, doc_index=idx)
        all_chunks.extend(chunks)
        if (idx + 1) % 100 == 0:
            logger.info("[CHUNK] Processed %s docs -> %s chunks", idx + 1, len(all_chunks))
    logger.info("[CHUNK] Total chunks: %s", len(all_chunks))
    return all_chunks


def _embed_chunks(
    chunks: Sequence,
    batch_size: int,
    model_name: Optional[str] = None,
    normalize_embeddings: Optional[bool] = None,
) -> List[List[float]]:
    from src.embeddings import embed_texts

    contents = [chunk.content for chunk in chunks]
    logger.info("[EMBED] Generating embeddings for %s chunks", len(contents))
    return embed_texts(
        contents,
        batch_size=batch_size,
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
        text_type="passage",
        add_prefix=True,
    )


def _index_qdrant(
    records: Sequence[Dict],
    chunks: Sequence,
    embeddings: Sequence[Sequence[float]],
    force_recreate: bool,
    collection_name: str | None = None,
    dense_dim: Optional[int] = None,
):
    from src.qdrant_store import QdrantLegalStore

    kwargs = {}
    if collection_name:
        kwargs["collection_name"] = collection_name
    if dense_dim is not None:
        kwargs["dense_dim"] = int(dense_dim)

    store = QdrantLegalStore(**kwargs)
    store.create_collection(force_recreate=force_recreate)

    if not force_recreate:
        for record in records:
            doc_id = record.get("doc_id")
            if not doc_id:
                continue
            try:
                store.delete_by_doc_id(doc_id)
            except Exception as exc:
                logger.warning("[QDRANT] Could not delete old chunks for %s: %s", doc_id, exc)

    store.add_documents(chunks, embeddings, batch_size=100)
    logger.info("[QDRANT] Stats: %s", store.get_stats())


def _index_graph(records: Sequence[Dict], chunks: Sequence, snapshot_date: Optional[str] = None):
    from src.legal_graph_extractor import LegalGraphExtractor
    from src.neo4j_store import get_graph_store

    graph_store = get_graph_store()
    if not graph_store.enabled:
        logger.info("[NEO4J] Graph indexing skipped (store disabled)")
        return

    graph_store.create_constraints()
    graph_store.upsert_doc_type_hierarchy()
    graph_store.upsert_documents(records)
    graph_store.upsert_provisions([chunk.to_dict() for chunk in chunks])

    extractor = LegalGraphExtractor()
    relations = extractor.extract_relations_for_documents(records, snapshot_date=snapshot_date)
    graph_store.upsert_amendment_relations(relations)
    logger.info(
        "[NEO4J] Indexed %s docs, %s provisions, %s relations",
        len(records),
        len(chunks),
        len(relations),
    )


def main():
    from src.legal_data_processing import IngestionStateStore
    from src.embeddings import DEFAULT_MODEL, get_embedding_dimension
    from src.qdrant_store import COLLECTION_NAME, collection_name_for_model, model_slug

    parser = argparse.ArgumentParser(description="Load vbpl-vn dataset into Gold set + Qdrant + Neo4j")
    parser.add_argument("--dataset", default="VietTung04/vbpl-vn", help="HuggingFace dataset path")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of documents (0 = all)")
    parser.add_argument("--as-of-date", default="", help="Date for active status check (YYYY-MM-DD)")
    parser.add_argument("--monitor-expiring-days", type=int, default=90, help="Mark records as expiring_soon within this many days")
    parser.add_argument("--profile", choices=tuple(PROFILE_SETTINGS.keys()), default="balanced", help="Soft filtering profile")
    parser.add_argument("--min-year-soft", type=int, default=None, help="Soft recency year for primary set")
    parser.add_argument("--target-min-laws", type=int, default=None, help="Soft target LUAT+PHAP_LENH count in primary")
    parser.add_argument("--backfill-min-year", type=int, default=None, help="Backfill floor year for LUAT/PHAP_LENH")
    parser.add_argument("--max-primary-records", type=int, default=None, help="Primary set cap (profile default when omitted)")
    parser.add_argument("--abort-on-low-laws", action="store_true", help="Abort when primary law count is below target")

    # Backward-compatible aliases
    parser.add_argument("--min-year", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--require-min-laws", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-records", type=int, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--gold-output", default=str(DEFAULT_GOLD_OUTPUT), help="Document-level gold dataset output (.parquet or .csv)")
    parser.add_argument("--gold-archive-output", default="", help="Archive dataset output (.parquet or .csv)")
    parser.add_argument(
        "--web-sources-output",
        default=str(DEFAULT_WEB_SOURCES_OUTPUT),
        help="Output JSON file with web-source seeds for crawl verification",
    )
    parser.add_argument("--allowed-doc-types", default=",".join(DEFAULT_ALLOWED_DOC_TYPES), help="Comma-separated allowed doc types")
    parser.add_argument("--keep-inactive", action="store_true", help="Allow inactive docs in primary set")
    parser.add_argument("--allow-missing-all-dates", action="store_true", help="Keep records even when all date fields are missing")
    parser.add_argument("--data-only", action="store_true", help="Only build and export gold dataset (skip embeddings/indexing)")

    parser.add_argument("--force-recreate", action="store_true", help="Recreate Qdrant collection and ignore incremental state")
    parser.add_argument("--skip-qdrant", action="store_true", help="Skip Qdrant indexing")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip Neo4j indexing")
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--embedding-model", default="", help="SentenceTransformer model id override")
    parser.add_argument(
        "--embedding-normalize",
        default="auto",
        choices=("auto", "true", "false"),
        help="Normalize embeddings before indexing",
    )
    parser.add_argument(
        "--collection-by-model",
        action="store_true",
        help="Auto-name Qdrant collection by model+dimension",
    )
    parser.add_argument("--qdrant-collection", default="", help="Override Qdrant collection name")
    parser.add_argument(
        "--state-path",
        default=str(PROJECT_ROOT / "data" / "cache" / "vbpl_ingest_state.json"),
        help="Incremental state file path",
    )
    args = parser.parse_args()

    embedding_model_name = args.embedding_model.strip() or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    normalize_embeddings: Optional[bool] = None
    if args.embedding_normalize == "true":
        normalize_embeddings = True
    elif args.embedding_normalize == "false":
        normalize_embeddings = False
    embedding_dim = int(get_embedding_dimension(embedding_model_name))

    if args.qdrant_collection:
        qdrant_collection = args.qdrant_collection
    elif args.collection_by_model:
        qdrant_collection = collection_name_for_model(COLLECTION_NAME, embedding_model_name, embedding_dim)
    else:
        qdrant_collection = ""

    normalize_label = "auto" if normalize_embeddings is None else str(normalize_embeddings).lower()
    encoder_profile = f"{model_slug(embedding_model_name)}|norm={normalize_label}"
    logger.info(
        "[EMBED] model=%s dim=%s normalize=%s collection=%s",
        embedding_model_name,
        embedding_dim,
        normalize_label,
        qdrant_collection or COLLECTION_NAME,
    )

    as_of_date = _parse_as_of_date(args.as_of_date) if args.as_of_date else date.today()
    allowed_doc_types = [x.strip().upper() for x in args.allowed_doc_types.split(",") if x.strip()]
    if not allowed_doc_types:
        allowed_doc_types = list(DEFAULT_ALLOWED_DOC_TYPES)

    # Honor backward-compatible aliases if present.
    if args.min_year is not None:
        args.min_year_soft = args.min_year
    if args.require_min_laws is not None:
        args.target_min_laws = args.require_min_laws
    if args.max_records is not None:
        args.max_primary_records = args.max_records

    resolved = _resolve_profile_settings(args)
    archive_output = Path(args.gold_archive_output) if args.gold_archive_output else _derive_archive_output(Path(args.gold_output))
    logger.info(
        "[CONFIG] profile=%s min_year_soft=%s target_min_laws=%s backfill_min_year=%s max_primary_records=%s",
        args.profile,
        resolved["min_year_soft"],
        resolved["target_min_laws"],
        resolved["backfill_min_year"],
        resolved["max_primary_records"],
    )

    start_time = time.time()
    df = _load_dataset(args.dataset, args.split)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
        logger.info("[DATASET] Limited to %s rows", len(df))

    normalized_records = _normalize_rows(
        df,
        as_of_date=as_of_date,
        monitor_expiring_days=args.monitor_expiring_days,
    )
    gold_sets = _filter_gold_records(
        normalized_records,
        drop_missing_all_dates=not args.allow_missing_all_dates,
        include_inactive_primary=args.keep_inactive,
        allowed_doc_types=allowed_doc_types,
        min_year_soft=resolved["min_year_soft"],
        target_min_laws=resolved["target_min_laws"],
        backfill_min_year=resolved["backfill_min_year"],
        max_primary_records=resolved["max_primary_records"],
        today=as_of_date,
    )
    gold_primary = gold_sets["primary"]
    gold_archive = gold_sets["archive"]
    gold_stats = gold_sets["stats"]
    quality_stats = _validate_primary_quality(gold_primary, target_min_laws=resolved["target_min_laws"])

    law_count = sum(1 for r in gold_primary if r.get("doc_type") in {"LUAT", "PHAP_LENH"})
    if law_count < resolved["target_min_laws"]:
        message = (
            "[GOLD] Primary law/phap lenh count %s below target %s "
            "(profile=%s). Pipeline continues with warning."
        )
        if args.abort_on_low_laws:
            logger.warning(message + " Aborting due to --abort-on-low-laws.", law_count, resolved["target_min_laws"], args.profile)
            return
        logger.warning(message, law_count, resolved["target_min_laws"], args.profile)

    _save_gold_records(gold_primary, Path(args.gold_output))
    _save_gold_records(gold_archive, archive_output)
    _save_web_sources(Path(args.web_sources_output))

    if args.data_only:
        logger.info("[GOLD] Stats: %s", gold_stats)
        logger.info("[QUALITY] Stats: %s", quality_stats)
        _log_distribution(gold_primary, label="primary")
        _log_distribution(gold_archive, label="archive")
        elapsed = time.time() - start_time
        logger.info(
            "[DONE] Data-only mode completed with primary=%s archive=%s in %.2fs",
            len(gold_primary),
            len(gold_archive),
            elapsed,
        )
        return

    state_store = IngestionStateStore(args.state_path)

    if args.force_recreate:
        records_to_update = gold_primary
        logger.info("[STATE] Force mode: ingest all %s records", len(records_to_update))
    else:
        diff = state_store.diff(gold_primary)
        records_to_update = diff.to_update
        logger.info("[STATE] Incremental diff: %s updates, %s unchanged", len(diff.to_update), len(diff.unchanged))

    if not records_to_update:
        logger.info("[DONE] No updates detected")
        return

    chunks = _chunk_records(records_to_update, encoder_profile=encoder_profile)
    if not chunks:
        logger.warning("[DONE] No chunks generated")
        return

    embeddings = _embed_chunks(
        chunks,
        batch_size=args.embedding_batch_size,
        model_name=embedding_model_name,
        normalize_embeddings=normalize_embeddings,
    )

    if not args.skip_qdrant:
        _index_qdrant(
            records=records_to_update,
            chunks=chunks,
            embeddings=embeddings,
            force_recreate=args.force_recreate,
            collection_name=qdrant_collection or None,
            dense_dim=embedding_dim,
        )

    if not args.skip_neo4j:
        _index_graph(records_to_update, chunks, snapshot_date=as_of_date.isoformat() if as_of_date else None)

    state_store.update_state(records_to_update)
    elapsed = time.time() - start_time
    logger.info(
        "[DONE] Updated %s docs -> %s chunks in %.2fs",
        len(records_to_update),
        len(chunks),
        elapsed,
    )


if __name__ == "__main__":
    main()
