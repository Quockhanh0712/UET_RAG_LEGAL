"""
Patch serving metadata in-place on Qdrant without rebuilding the index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.legal_data_processing import (
    build_citation_base,
    build_citation_full,
    clean_serving_title,
    derive_law_family,
    normalize_whitespace,
)

PATCH_VERSION = "metadata_patch_v1_20260313"
DEFAULT_COLLECTION = "legal_documents__truro7_vn_law_embedding__d768"
DEFAULT_SCROLL_RETRIES = 5
DEFAULT_BATCH_RETRIES = 5
LEAN_DELETE_FIELDS = (
    "ten_van_ban_raw",
    "law_family_raw",
    "metadata_patch_version",
    "metadata_patch_flags",
    "metadata_patched_at",
    "citation_short",
    "text",
    "path",
    "hierarchy",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_env(name: str) -> str:
    value = str(os.getenv(name, "")).strip()
    if not value:
        raise RuntimeError(f"Missing required env: {name}")
    return value


def _load_client(timeout: int) -> QdrantClient:
    return QdrantClient(
        url=_require_env("QDRANT_CLOUD_URL"),
        api_key=_require_env("QDRANT_API_KEY"),
        timeout=timeout,
    )


def _with_retries(fn, *, retries: int, delay_s: float, action: str):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            sleep_s = delay_s * attempt
            print(f"[WARN] {action} failed on attempt {attempt}/{retries}: {exc}. retry_in={sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise last_error


def _scroll_collection(
    client: QdrantClient,
    collection_name: str,
    scroll_limit: int,
    scroll_retries: int,
) -> Tuple[Dict[str, Dict], int]:
    offset = None
    total_points = 0
    doc_map: Dict[str, Dict] = {}
    payload_selector = [
        "doc_id",
        "van_ban_id",
        "so_ky_hieu",
        "loai_van_ban",
        "ten_van_ban",
        "ten_van_ban_raw",
        "law_family",
        "law_family_raw",
        "citation_base",
        "citation_full",
        "citation_short",
        "legal_path",
        "path",
    ]

    while True:
        points, offset = _with_retries(
            lambda: client.scroll(
                collection_name=collection_name,
                limit=scroll_limit,
                offset=offset,
                with_payload=payload_selector,
                with_vectors=False,
            ),
            retries=scroll_retries,
            delay_s=2.0,
            action=f"scroll offset={offset}",
        )
        if not points:
            break

        total_points += len(points)
        for point in points:
            payload = point.payload or {}
            doc_id = normalize_whitespace(str(payload.get("doc_id") or payload.get("van_ban_id") or "")).upper()
            if not doc_id:
                continue

            entry = doc_map.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "payload": payload,
                    "points": [],
                    "point_count": 0,
                },
            )
            entry["point_count"] += 1
            if not entry.get("payload"):
                entry["payload"] = payload
            entry["points"].append(
                {
                    "id": point.id,
                    "path": normalize_whitespace(str(payload.get("path") or "")),
                    "legal_path": normalize_whitespace(str(payload.get("legal_path") or "")),
                    "citation_full": normalize_whitespace(str(payload.get("citation_full") or "")),
                    "citation_short": normalize_whitespace(str(payload.get("citation_short") or "")),
                }
            )

        if offset is None:
            break

    return doc_map, total_points


def _build_doc_patch(entry: Dict, patched_at: str, lean: bool) -> Dict:
    payload = entry["payload"]
    raw_title = normalize_whitespace(str(payload.get("ten_van_ban_raw") or payload.get("ten_van_ban") or ""))
    raw_family = normalize_whitespace(str(payload.get("law_family_raw") or payload.get("law_family") or ""))
    loai_van_ban = normalize_whitespace(str(payload.get("loai_van_ban") or ""))
    so_ky_hieu = normalize_whitespace(str(payload.get("so_ky_hieu") or entry["doc_id"]))

    cleaned_title, flags, confident = clean_serving_title(
        ten_van_ban=raw_title,
        loai_van_ban=loai_van_ban,
        so_ky_hieu=so_ky_hieu,
    )
    target_title = cleaned_title if confident else raw_title
    target_family = derive_law_family(target_title, loai_van_ban, so_ky_hieu) if confident else raw_family
    target_base = build_citation_base(loai_van_ban, so_ky_hieu, target_title) if confident else normalize_whitespace(
        str(payload.get("citation_base") or "")
    )

    patch_flags = list(flags)
    if not confident:
        patch_flags.append("kept_raw_title")
    patch_flags = list(dict.fromkeys(patch_flags))

    common_payload = {}
    if not lean:
        common_payload.update(
            {
                "ten_van_ban_raw": raw_title,
                "law_family_raw": raw_family,
                "metadata_patch_version": PATCH_VERSION,
                "metadata_patch_flags": patch_flags,
                "metadata_patched_at": patched_at,
            }
        )
    if confident:
        common_payload.update(
            {
                "ten_van_ban": target_title,
                "law_family": target_family,
                "citation_base": target_base,
            }
        )

    citation_groups: Dict[Tuple[str, str], List] = defaultdict(list)
    citation_changed_points = 0
    if confident:
        for point in entry["points"]:
            path = point["path"] or ""
            legal_path = point["legal_path"] or path
            target_short = path or "Nguon phap ly"
            target_full = build_citation_full(target_base, legal_path)
            if target_full != point["citation_full"] or ((not lean) and target_short != point["citation_short"]):
                citation_groups[(target_full, target_short)].append(point["id"])
                citation_changed_points += 1

    title_changed = confident and target_title != raw_title
    family_changed = confident and target_family != raw_family
    base_changed = confident and target_base != normalize_whitespace(str(payload.get("citation_base") or ""))

    return {
        "doc_id": entry["doc_id"],
        "so_ky_hieu": so_ky_hieu,
        "loai_van_ban": loai_van_ban,
        "raw_title": raw_title,
        "target_title": target_title,
        "raw_family": raw_family,
        "target_family": target_family,
        "target_base": target_base,
        "common_payload": common_payload,
        "citation_groups": citation_groups,
        "point_count": entry["point_count"],
        "citation_changed_points": citation_changed_points,
        "title_changed": title_changed,
        "family_changed": family_changed,
        "base_changed": base_changed,
        "confident": confident,
        "flags": patch_flags,
    }


def _write_report(output_dir: Path, payload: Dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metadata_patch_report.json"
    md_path = output_dir / "metadata_patch_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Qdrant Metadata Patch Report",
        "",
        f"- run_id: `{payload.get('run_id', '')}`",
        f"- collection: `{payload.get('collection', '')}`",
        f"- dry_run: `{payload.get('dry_run', True)}`",
        f"- lean: `{payload.get('lean', False)}`",
        f"- patched_at: `{payload.get('patched_at', '')}`",
        f"- docs_seen: `{payload.get('docs_seen', 0)}`",
        f"- points_seen: `{payload.get('points_seen', 0)}`",
        f"- docs_confident: `{payload.get('docs_confident', 0)}`",
        f"- docs_low_confidence: `{payload.get('docs_low_confidence', 0)}`",
        f"- docs_title_changed: `{payload.get('docs_title_changed', 0)}`",
        f"- docs_family_changed: `{payload.get('docs_family_changed', 0)}`",
        f"- points_citation_changed: `{payload.get('points_citation_changed', 0)}`",
        f"- common_operations: `{payload.get('common_operations', 0)}`",
        f"- citation_operations: `{payload.get('citation_operations', 0)}`",
        f"- delete_operations: `{payload.get('delete_operations', 0)}`",
        "",
        "## Sample Diffs",
        "",
    ]
    for item in payload.get("sample_diffs", []):
        lines.append(f"### {item.get('doc_id', '')}")
        lines.append(f"- confident: `{item.get('confident', False)}`")
        lines.append(f"- flags: `{item.get('flags', [])}`")
        lines.append(f"- before_title: `{item.get('before_title', '')}`")
        lines.append(f"- after_title: `{item.get('after_title', '')}`")
        lines.append(f"- before_family: `{item.get('before_family', '')}`")
        lines.append(f"- after_family: `{item.get('after_family', '')}`")
        lines.append(f"- citation_changed_points: `{item.get('citation_changed_points', 0)}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _batched(items: List, size: int) -> List[List]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _make_doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(
        should=[
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id),
            ),
            models.FieldCondition(
                key="van_ban_id",
                match=models.MatchValue(value=doc_id),
            ),
        ]
    )


def _build_operations(doc_patches: List[Dict], lean: bool) -> Tuple[List, List, List]:
    common_ops = []
    citation_ops = []
    delete_ops = []
    for patch in doc_patches:
        if patch["common_payload"]:
            common_ops.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload=patch["common_payload"],
                        filter=_make_doc_filter(patch["doc_id"]),
                    )
                )
            )
        if lean:
            delete_ops.append(
                models.DeletePayloadOperation(
                    delete_payload=models.DeletePayload(
                        keys=list(LEAN_DELETE_FIELDS),
                        filter=_make_doc_filter(patch["doc_id"]),
                    )
                )
            )
        for (citation_full, citation_short), point_ids in patch["citation_groups"].items():
            citation_payload = {"citation_full": citation_full}
            if not lean:
                citation_payload["citation_short"] = citation_short
            citation_ops.append(
                models.SetPayloadOperation(
                    set_payload=models.SetPayload(
                        payload=citation_payload,
                        points=point_ids,
                    )
                )
            )
    return common_ops, citation_ops, delete_ops


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch Qdrant metadata in-place for serving collection.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data" / "gold" / "research" / "qdrant_metadata_patch"))
    parser.add_argument("--scroll-limit", type=int, default=512)
    parser.add_argument("--batch-ops", type=int, default=64)
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--scroll-retries", type=int, default=DEFAULT_SCROLL_RETRIES)
    parser.add_argument("--batch-retries", type=int, default=DEFAULT_BATCH_RETRIES)
    parser.add_argument("--lean", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    client = _load_client(timeout=args.timeout)
    patched_at = _utc_now()

    doc_map, points_seen = _scroll_collection(
        client=client,
        collection_name=args.collection,
        scroll_limit=args.scroll_limit,
        scroll_retries=args.scroll_retries,
    )
    doc_patches = [_build_doc_patch(entry, patched_at=patched_at, lean=args.lean) for entry in doc_map.values()]
    doc_patches.sort(key=lambda item: item["doc_id"])
    if args.limit_docs > 0:
        doc_patches = doc_patches[: args.limit_docs]

    docs_confident = sum(1 for item in doc_patches if item["confident"])
    docs_low_confidence = sum(1 for item in doc_patches if not item["confident"])
    docs_title_changed = sum(1 for item in doc_patches if item["title_changed"])
    docs_family_changed = sum(1 for item in doc_patches if item["family_changed"])
    points_citation_changed = sum(int(item["citation_changed_points"]) for item in doc_patches)

    common_ops, citation_ops, delete_ops = _build_operations(doc_patches, lean=args.lean)

    sample_diffs = []
    interesting = [
        item
        for item in doc_patches
        if item["title_changed"] or item["family_changed"] or not item["confident"]
    ]
    for item in interesting[: args.sample_limit]:
        sample_diffs.append(
            {
                "doc_id": item["doc_id"],
                "confident": item["confident"],
                "flags": item["flags"],
                "before_title": item["raw_title"][:220],
                "after_title": item["target_title"][:220],
                "before_family": item["raw_family"][:220],
                "after_family": item["target_family"][:220],
                "citation_changed_points": item["citation_changed_points"],
            }
        )

    report = {
        "run_id": run_id,
        "collection": args.collection,
        "dry_run": not args.apply,
        "lean": bool(args.lean),
        "patched_at": patched_at,
        "docs_seen": len(doc_patches),
        "points_seen": points_seen,
        "docs_confident": docs_confident,
        "docs_low_confidence": docs_low_confidence,
        "docs_title_changed": docs_title_changed,
        "docs_family_changed": docs_family_changed,
        "points_citation_changed": points_citation_changed,
        "common_operations": len(common_ops),
        "citation_operations": len(citation_ops),
        "delete_operations": len(delete_ops),
        "sample_diffs": sample_diffs,
    }
    _write_report(output_dir=output_dir, payload=report)

    if not args.apply:
        print(output_dir / "metadata_patch_report.json")
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "collection": args.collection,
                    "dry_run": True,
                    "lean": bool(args.lean),
                    "docs_seen": len(doc_patches),
                    "docs_confident": docs_confident,
                    "docs_low_confidence": docs_low_confidence,
                    "docs_title_changed": docs_title_changed,
                    "docs_family_changed": docs_family_changed,
                    "points_citation_changed": points_citation_changed,
                },
                ensure_ascii=True,
            )
        )
        return 0

    for batch in _batched(common_ops, args.batch_ops):
        _with_retries(
            lambda current_batch=batch: client.batch_update_points(
                collection_name=args.collection,
                update_operations=current_batch,
                wait=True,
                timeout=args.timeout,
            ),
            retries=args.batch_retries,
            delay_s=3.0,
            action=f"batch_update common_ops size={len(batch)}",
        )
    for batch in _batched(citation_ops, args.batch_ops):
        _with_retries(
            lambda current_batch=batch: client.batch_update_points(
                collection_name=args.collection,
                update_operations=current_batch,
                wait=True,
                timeout=args.timeout,
            ),
            retries=args.batch_retries,
            delay_s=3.0,
            action=f"batch_update citation_ops size={len(batch)}",
        )
    for batch in _batched(delete_ops, args.batch_ops):
        _with_retries(
            lambda current_batch=batch: client.batch_update_points(
                collection_name=args.collection,
                update_operations=current_batch,
                wait=True,
                timeout=args.timeout,
            ),
            retries=args.batch_retries,
            delay_s=3.0,
            action=f"batch_update delete_ops size={len(batch)}",
        )

    points_after = int(getattr(client.get_collection(args.collection), "points_count", 0) or 0)
    report["points_after"] = points_after
    report["dry_run"] = False
    _write_report(output_dir=output_dir, payload=report)
    print(output_dir / "metadata_patch_report.json")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "collection": args.collection,
                "dry_run": False,
                "lean": bool(args.lean),
                "docs_seen": len(doc_patches),
                "points_seen": points_seen,
                "points_after": points_after,
                "docs_confident": docs_confident,
                "docs_low_confidence": docs_low_confidence,
                "docs_title_changed": docs_title_changed,
                "docs_family_changed": docs_family_changed,
                "points_citation_changed": points_citation_changed,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
