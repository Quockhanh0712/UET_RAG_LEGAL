"""
Prepare/evaluate rule-based legal graph relations for research v2.

Usage:
1) Create review template:
   python scripts/evaluate_graph_relations_v2.py --input-path data/gold/vbpl_gold.parquet
2) Fill `is_correct` in exported CSV, then re-run with --labels-path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.legal_graph_extractor import (
    REL_ABROGATES,
    REL_AMENDS,
    REL_REPLACES,
    LegalGraphExtractor,
)


CORE_RELATIONS = {REL_AMENDS, REL_REPLACES, REL_ABROGATES}


def _load_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _extract_sample_rows(df: pd.DataFrame, sample_size: int) -> List[Dict[str, str]]:
    extractor = LegalGraphExtractor()
    rows: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        doc = row.to_dict()
        doc_id = str(doc.get("doc_id") or doc.get("so_ky_hieu") or doc.get("van_ban_id") or "")
        mo_ta = str(doc.get("mo_ta") or "")
        if not doc_id or not mo_ta:
            continue
        rels = extractor.extract_amendment_relations(doc_id=doc_id, mo_ta=mo_ta)
        for rel in rels:
            rows.append(
                {
                    "source_doc_id": rel.source_doc_id,
                    "target_doc_id": rel.target_doc_id,
                    "relation_type": rel.relation_type,
                    "evidence_text": rel.evidence_text,
                    "evidence_source": rel.evidence_source,
                    "confidence_rule": rel.confidence_rule,
                    "snapshot_date": rel.snapshot_date,
                    "is_correct": "",
                    "reviewer_note": "",
                }
            )
            if len(rows) >= sample_size:
                return rows
    return rows


def _duplicate_count(rows: List[Dict[str, str]]) -> int:
    seen = set()
    dup = 0
    for r in rows:
        key = (r["source_doc_id"], r["target_doc_id"], r["relation_type"])
        if key in seen:
            dup += 1
        else:
            seen.add(key)
    return dup


def _evaluate_labels(labels_df: pd.DataFrame) -> Dict[str, float]:
    subset = labels_df[labels_df["relation_type"].isin(list(CORE_RELATIONS))].copy()
    subset["is_correct_num"] = pd.to_numeric(subset["is_correct"], errors="coerce")
    judged = subset.dropna(subset=["is_correct_num"])
    if len(judged) == 0:
        return {"precision_core": 0.0, "judged_samples": 0}
    precision = float((judged["is_correct_num"] >= 1).mean())
    return {"precision_core": precision, "judged_samples": int(len(judged))}


def _to_markdown(summary: Dict[str, float]) -> str:
    return (
        "# Graph Relation Evaluation Report (v2)\n\n"
        f"- Sample size: `{int(summary.get('sample_size', 0))}`\n"
        f"- Duplicate edge count: `{int(summary.get('duplicate_edges', 0))}`\n"
        f"- Judged samples: `{int(summary.get('judged_samples', 0))}`\n"
        f"- Precision (AMENDS/REPLACES/ABROGATES): `{summary.get('precision_core', 0.0):.2%}` (target >= 85%)\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate legal graph relations for research v2")
    parser.add_argument("--input-path", default="data/gold/vbpl_gold.parquet")
    parser.add_argument("--output-dir", default="data/gold/research")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument(
        "--labels-path",
        default="",
        help="Optional reviewed CSV path with is_correct labels",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = output_dir / "graph_relation_labels_template.csv"
    df = _load_df(input_path)
    rows = _extract_sample_rows(df, sample_size=args.sample_size)
    pd.DataFrame(rows).to_csv(sample_path, index=False, encoding="utf-8")

    summary = {
        "sample_size": len(rows),
        "duplicate_edges": _duplicate_count(rows),
        "precision_core": 0.0,
        "judged_samples": 0,
    }

    labels_path = Path(args.labels_path) if args.labels_path else sample_path
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path, dtype=str)
        summary.update(_evaluate_labels(labels_df))

    (output_dir / "graph_relation_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "graph_relation_eval_summary.md").write_text(
        _to_markdown(summary),
        encoding="utf-8",
    )

    print(f"[DONE] Relation template: {sample_path}")
    print(f"[DONE] Summary exported to: {output_dir}")


if __name__ == "__main__":
    main()
