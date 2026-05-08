"""
Data mining report for filtered legal gold datasets.

Outputs:
- CSV summaries (distribution, laws list, quality checks)
- Markdown report with quality evaluation
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.legal_data_processing import normalize_whitespace, strip_accents
from src.legal_gold_filter import SEED_LAW_FAMILIES


LAW_DOC_TYPES = {"LUAT", "PHAP_LENH"}


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _to_year(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.year


def _distribution(series: pd.Series, name: str) -> pd.DataFrame:
    counts = series.fillna("UNKNOWN").astype(str).value_counts(dropna=False).rename_axis(name).reset_index(name="count")
    total = int(counts["count"].sum()) if len(counts) else 0
    counts["ratio"] = counts["count"] / total if total else 0.0
    return counts


def _numeric_summary(series: pd.Series) -> Dict[str, float]:
    data = pd.to_numeric(series, errors="coerce").dropna()
    if len(data) == 0:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(data.min()),
        "p50": float(data.quantile(0.5)),
        "p90": float(data.quantile(0.9)),
        "max": float(data.max()),
        "mean": float(data.mean()),
    }


def _contains_seed(record_text: str, seed_key: str) -> bool:
    return seed_key in strip_accents(normalize_whitespace(record_text or "")).upper()


def _seed_coverage(primary_df: pd.DataFrame, archive_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    primary_text = (
        primary_df["law_family"].fillna("")
        + " "
        + primary_df["ten_van_ban"].fillna("")
        + " "
        + primary_df["mo_ta"].fillna("")
    )
    archive_text = (
        archive_df["law_family"].fillna("")
        + " "
        + archive_df["ten_van_ban"].fillna("")
        + " "
        + archive_df["mo_ta"].fillna("")
    )

    for seed in SEED_LAW_FAMILIES:
        seed_key = strip_accents(seed).upper()
        primary_hits = int(primary_text.apply(lambda x: _contains_seed(x, seed_key)).sum())
        archive_hits = int(archive_text.apply(lambda x: _contains_seed(x, seed_key)).sum())
        rows.append(
            {
                "seed_law_family": seed,
                "primary_hits": primary_hits,
                "archive_hits": archive_hits,
                "covered_in_primary": primary_hits > 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["covered_in_primary", "primary_hits"], ascending=[False, False])


def _quality_checks(
    primary_df: pd.DataFrame,
    recent_year_threshold: int,
    target_min_laws: int,
    target_law_ratio: float,
    target_recent_ratio: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    total = len(primary_df)
    law_count = int(primary_df["doc_type"].isin(list(LAW_DOC_TYPES)).sum()) if total else 0
    law_ratio = law_count / total if total else 0.0

    years = _to_year(primary_df["ngay_ban_hanh"]) if "ngay_ban_hanh" in primary_df.columns else pd.Series(dtype=float)
    dated = years.dropna()
    recent_count = int((dated >= recent_year_threshold).sum())
    recent_ratio = recent_count / len(dated) if len(dated) else 0.0

    active_ratio = float(primary_df["is_active"].fillna(False).mean()) if "is_active" in primary_df.columns else 0.0
    duplicate_doc_id = int(primary_df.duplicated(subset=["doc_id"]).sum()) if "doc_id" in primary_df.columns else 0
    duplicate_hash = int(primary_df.duplicated(subset=["content_hash"]).sum()) if "content_hash" in primary_df.columns else 0
    missing_issue_date = int(primary_df["ngay_ban_hanh"].isna().sum()) if "ngay_ban_hanh" in primary_df.columns else total
    missing_issue_ratio = missing_issue_date / total if total else 0.0

    checks = [
        {
            "check_name": "law_count_or_ratio",
            "value": f"count={law_count}, ratio={law_ratio:.4f}",
            "target": f"count>={target_min_laws} OR ratio>={target_law_ratio:.2f}",
            "status": "PASS" if (law_count >= target_min_laws or law_ratio >= target_law_ratio) else "WARN",
        },
        {
            "check_name": "recent_ratio",
            "value": f"{recent_ratio:.4f}",
            "target": f">={target_recent_ratio:.2f} (year>={recent_year_threshold})",
            "status": "PASS" if recent_ratio >= target_recent_ratio else "WARN",
        },
        {
            "check_name": "active_ratio",
            "value": f"{active_ratio:.4f}",
            "target": ">=0.95",
            "status": "PASS" if active_ratio >= 0.95 else "WARN",
        },
        {
            "check_name": "duplicate_doc_id",
            "value": str(duplicate_doc_id),
            "target": "==0",
            "status": "PASS" if duplicate_doc_id == 0 else "FAIL",
        },
        {
            "check_name": "duplicate_content_hash",
            "value": str(duplicate_hash),
            "target": "==0",
            "status": "PASS" if duplicate_hash == 0 else "FAIL",
        },
        {
            "check_name": "missing_issue_date_ratio",
            "value": f"{missing_issue_ratio:.4f}",
            "target": "<=0.05",
            "status": "PASS" if missing_issue_ratio <= 0.05 else "WARN",
        },
    ]
    checks_df = pd.DataFrame(checks)

    score = 100
    status_map = {"PASS": 0, "WARN": 8, "FAIL": 25}
    for check in checks:
        score -= status_map[check["status"]]
    score = max(0, score)
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 55:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"

    summary = {
        "quality_score": float(score),
        "quality_grade": grade,
        "primary_total": float(total),
        "law_count": float(law_count),
        "law_ratio": float(law_ratio),
        "recent_ratio": float(recent_ratio),
        "active_ratio": float(active_ratio),
    }
    return checks_df, summary


def _build_law_list(primary_df: pd.DataFrame, archive_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "doc_id",
        "doc_type",
        "so_ky_hieu",
        "ten_van_ban",
        "law_family",
        "ngay_ban_hanh",
        "ngay_co_hieu_luc",
        "ngay_het_hieu_luc",
        "is_active",
        "priority_score",
        "co_quan_ban_hanh",
    ]
    p = primary_df.reindex(columns=required_columns).copy()
    p["source_set"] = "primary"
    a = archive_df.reindex(columns=required_columns).copy()
    a["source_set"] = "archive"
    combined = pd.concat([p, a], ignore_index=True)
    laws = combined[combined["doc_type"].isin(list(LAW_DOC_TYPES))].copy()

    if len(laws) == 0:
        return laws

    laws["source_priority"] = laws["source_set"].map({"primary": 1, "archive": 0}).fillna(0)
    laws["issue_year"] = _to_year(laws["ngay_ban_hanh"])
    laws = laws.sort_values(["source_priority", "issue_year", "priority_score"], ascending=[False, False, False])
    laws = laws.drop_duplicates(subset=["doc_id"], keep="first")
    columns = ["doc_id", "source_set", *required_columns[1:]]
    return laws[[col for col in columns if col in laws.columns]]


def _render_markdown(
    report_path: Path,
    primary_df: pd.DataFrame,
    archive_df: pd.DataFrame,
    quality_summary: Dict[str, float],
    checks_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    dist_primary_doc_type: pd.DataFrame,
    dist_archive_doc_type: pd.DataFrame,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    covered_seed = int(seed_df["covered_in_primary"].sum()) if len(seed_df) else 0
    total_seed = int(len(seed_df))

    primary_text_len = _numeric_summary(primary_df["toan_van_clean"].fillna("").str.len()) if "toan_van_clean" in primary_df.columns else {}
    archive_text_len = _numeric_summary(archive_df["toan_van_clean"].fillna("").str.len()) if "toan_van_clean" in archive_df.columns else {}

    content = []
    content.append("# Gold Dataset Mining Report")
    content.append("")
    content.append(f"- Generated at: `{now}`")
    content.append(f"- Primary rows: `{len(primary_df)}`")
    content.append(f"- Archive rows: `{len(archive_df)}`")
    content.append("")
    content.append("## Quality Summary")
    content.append("")
    content.append(f"- Quality grade: **{quality_summary['quality_grade']}**")
    content.append(f"- Quality score: **{quality_summary['quality_score']:.1f}/100**")
    content.append(f"- Law count (primary): `{int(quality_summary['law_count'])}`")
    content.append(f"- Law ratio (primary): `{quality_summary['law_ratio']:.2%}`")
    content.append(f"- Recent ratio (primary): `{quality_summary['recent_ratio']:.2%}`")
    content.append(f"- Active ratio (primary): `{quality_summary['active_ratio']:.2%}`")
    content.append("")
    content.append("## Seed Coverage")
    content.append("")
    content.append(f"- Covered seed families in primary: `{covered_seed}/{total_seed}`")
    if total_seed > 0:
        uncovered = seed_df[~seed_df["covered_in_primary"]]["seed_law_family"].tolist()
        content.append(f"- Uncovered seed families: `{len(uncovered)}`")
        if uncovered:
            preview = ", ".join(uncovered[:8])
            content.append(f"- Uncovered preview: {preview}")
    content.append("")
    content.append("## Doc Type Distribution")
    content.append("")
    content.append("### Primary (top 10)")
    content.append("")
    content.append(dist_primary_doc_type.head(10).to_markdown(index=False))
    content.append("")
    content.append("### Archive (top 10)")
    content.append("")
    content.append(dist_archive_doc_type.head(10).to_markdown(index=False))
    content.append("")
    content.append("## Text Length")
    content.append("")
    content.append(f"- Primary chars p50/p90: `{primary_text_len.get('p50', 0):.0f}/{primary_text_len.get('p90', 0):.0f}`")
    content.append(f"- Archive chars p50/p90: `{archive_text_len.get('p50', 0):.0f}/{archive_text_len.get('p90', 0):.0f}`")
    content.append("")
    content.append("## Quality Checks")
    content.append("")
    content.append(checks_df.to_markdown(index=False))
    content.append("")
    content.append("## Notes")
    content.append("")
    content.append("- `gold_primary` is retrieval default corpus.")
    content.append("- `gold_archive` is long-tail corpus for history/graph expansion.")

    report_path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mining report for gold legal datasets")
    parser.add_argument("--primary-path", default="data/gold/vbpl_gold.parquet", help="Path to gold primary dataset")
    parser.add_argument("--archive-path", default="data/gold/vbpl_gold_archive.parquet", help="Path to gold archive dataset")
    parser.add_argument("--output-dir", default="data/gold/reports", help="Output directory for report artifacts")
    parser.add_argument("--recent-year-threshold", type=int, default=2015, help="Recent year threshold for quality checks")
    parser.add_argument("--target-min-laws", type=int, default=40, help="Target minimum LUAT+PHAP_LENH in primary")
    parser.add_argument("--target-law-ratio", type=float, default=0.25, help="Target law ratio in primary")
    parser.add_argument("--target-recent-ratio", type=float, default=0.55, help="Target recent ratio in primary")
    args = parser.parse_args()

    primary_path = Path(args.primary_path)
    archive_path = Path(args.archive_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_df = _load_frame(primary_path)
    archive_df = _load_frame(archive_path)

    primary_doc_type = _distribution(primary_df["doc_type"], "doc_type")
    archive_doc_type = _distribution(archive_df["doc_type"], "doc_type")
    primary_year = _distribution(_to_year(primary_df["ngay_ban_hanh"]).fillna(-1).astype(int), "issue_year")
    archive_year = _distribution(_to_year(archive_df["ngay_ban_hanh"]).fillna(-1).astype(int), "issue_year")
    primary_family = _distribution(primary_df["law_family"], "law_family")
    primary_agency = _distribution(primary_df["co_quan_ban_hanh"], "co_quan_ban_hanh")

    law_list = _build_law_list(primary_df, archive_df)
    seed_df = _seed_coverage(primary_df, archive_df)
    checks_df, quality_summary = _quality_checks(
        primary_df=primary_df,
        recent_year_threshold=args.recent_year_threshold,
        target_min_laws=args.target_min_laws,
        target_law_ratio=args.target_law_ratio,
        target_recent_ratio=args.target_recent_ratio,
    )

    # Save structured outputs.
    primary_doc_type.to_csv(output_dir / "primary_doc_type_distribution.csv", index=False, encoding="utf-8")
    archive_doc_type.to_csv(output_dir / "archive_doc_type_distribution.csv", index=False, encoding="utf-8")
    primary_year.to_csv(output_dir / "primary_year_distribution.csv", index=False, encoding="utf-8")
    archive_year.to_csv(output_dir / "archive_year_distribution.csv", index=False, encoding="utf-8")
    primary_family.head(200).to_csv(output_dir / "primary_law_family_top.csv", index=False, encoding="utf-8")
    primary_agency.head(200).to_csv(output_dir / "primary_agency_top.csv", index=False, encoding="utf-8")
    law_list.to_csv(output_dir / "law_documents_list.csv", index=False, encoding="utf-8")
    seed_df.to_csv(output_dir / "seed_coverage.csv", index=False, encoding="utf-8")
    checks_df.to_csv(output_dir / "quality_checks.csv", index=False, encoding="utf-8")
    pd.DataFrame([quality_summary]).to_csv(output_dir / "quality_summary.csv", index=False, encoding="utf-8")

    _render_markdown(
        report_path=output_dir / "gold_dataset_report.md",
        primary_df=primary_df,
        archive_df=archive_df,
        quality_summary=quality_summary,
        checks_df=checks_df,
        seed_df=seed_df,
        dist_primary_doc_type=primary_doc_type,
        dist_archive_doc_type=archive_doc_type,
    )

    print(f"[DONE] Report exported to: {output_dir}")
    print(f"[DONE] Quality grade: {quality_summary['quality_grade']} score={quality_summary['quality_score']:.1f}")


if __name__ == "__main__":
    main()
