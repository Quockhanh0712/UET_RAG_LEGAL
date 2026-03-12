"""
Evaluate chunking quality gates for research v2.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.legal_chunker_v2 import LegalChunkerV2


ARTICLE_PATTERN = re.compile(
    r"^\s*(?:\u0110i\u1ec1u|Dieu)\s+(\d+[A-Za-z0-9\-]*)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _cross_dieu_violation(chunk) -> bool:
    own_dieu = str(getattr(chunk, "dieu_so", "") or "").strip().lower()
    if not own_dieu:
        return False
    # Only count article headings at line starts to avoid false positives
    # from cross-references in normal prose.
    matches = [m.lower() for m in ARTICLE_PATTERN.findall(str(getattr(chunk, "content", "") or ""))]
    if not matches:
        return False
    # Invariant for chunk boundary: the leading article heading must match
    # the chunk's own article. Additional article headings may appear in
    # quoted amendment text and are not considered boundary violations.
    return matches[0] != own_dieu


def _to_rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0


def evaluate(df: pd.DataFrame, max_docs: int = 0) -> Dict[str, float]:
    if max_docs > 0:
        df = df.head(max_docs)
    evaluated_docs = len(df)

    chunker = LegalChunkerV2()
    chunks: List = []
    for _, row in df.iterrows():
        chunks.extend(chunker.chunk_document(row.to_dict()))

    total = len(chunks)
    if total == 0:
        return {
            "evaluated_docs": evaluated_docs,
            "total_chunks": 0,
            "provision_id_valid_rate": 0.0,
            "structured_chunk_rate": 0.0,
            "citation_full_rate": 0.0,
            "non_cross_dieu_rate": 0.0,
            "fallback_chunk_rate": 0.0,
        }

    provision_id_valid = sum(1 for c in chunks if str(getattr(c, "provision_id", "") or "").strip() != "")
    structured = sum(1 for c in chunks if getattr(c, "chunk_type", "") in {"ProvisionChunk", "ClauseChunk"})
    citation_full = sum(1 for c in chunks if str(getattr(c, "citation_full", "") or "").strip() != "")
    no_cross_dieu = sum(1 for c in chunks if not _cross_dieu_violation(c))
    fallback_count = sum(1 for c in chunks if getattr(c, "chunk_type", "") == "FallbackChunk")

    return {
        "evaluated_docs": evaluated_docs,
        "total_chunks": total,
        "provision_id_valid_rate": _to_rate(provision_id_valid, total),
        "structured_chunk_rate": _to_rate(structured, total),
        "citation_full_rate": _to_rate(citation_full, total),
        "non_cross_dieu_rate": _to_rate(no_cross_dieu, total),
        "fallback_chunk_rate": _to_rate(fallback_count, total),
    }


def _render_markdown(metrics: Dict[str, float]) -> str:
    return (
        "# Chunking Quality Report (v2)\n\n"
        f"- Evaluated docs: `{int(metrics['evaluated_docs'])}`\n"
        f"- Total chunks: `{int(metrics['total_chunks'])}`\n"
        f"- Provision ID valid rate: `{metrics['provision_id_valid_rate']:.2%}` (target >= 90%)\n"
        f"- Structured chunk rate: `{metrics['structured_chunk_rate']:.2%}` (target >= 85%)\n"
        f"- Citation full rate: `{metrics['citation_full_rate']:.2%}` (target = 100%)\n"
        f"- Non cross-Dieu rate: `{metrics['non_cross_dieu_rate']:.2%}` (target = 100%)\n"
        f"- Fallback chunk rate: `{metrics['fallback_chunk_rate']:.2%}`\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate legal chunking quality for research v2")
    parser.add_argument("--input-path", default="data/gold/vbpl_gold.parquet")
    parser.add_argument("--output-dir", default="data/gold/research")
    parser.add_argument("--max-docs", type=int, default=0, help="Optional doc cap for quick checks")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataframe(input_path)
    metrics = evaluate(df, max_docs=args.max_docs)

    (output_dir / "chunk_quality_report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "chunk_quality_report.md").write_text(
        _render_markdown(metrics),
        encoding="utf-8",
    )
    print(f"[DONE] Chunking report exported to: {output_dir}")


if __name__ == "__main__":
    main()
