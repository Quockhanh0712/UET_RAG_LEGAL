"""
Bootstrap research v2 artifacts for legal chunking + hybrid search + graph.

This script generates deterministic artifacts under data/gold/research:
- evidence_matrix.csv
- graph_schema_spec.md
- chunking_spec.md
- experiment_plan.yaml
- research_survey_report.md
- decision_log.md
- benchmark_results.csv (header only if not exists)
- query_set_v1.csv (60 queries: factual/amendment/temporal)
- eval_pairs_template.csv (300 query-doc rows: 200 dev, 100 test)
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "data" / "gold" / "research"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_evidence_matrix():
    rows = [
        {
            "paper_id": "graphrag_2024",
            "venue": "arXiv",
            "task": "GraphRAG architecture",
            "dataset": "multi-domain",
            "retrieval_stack": "graph + vector",
            "graph_role": "context expansion and organization",
            "chunking_role": "entity-centric",
            "metrics": "qualitative + retrieval quality",
            "key_finding": "Graph context helps multi-hop and thematic queries",
            "risk": "cost blow-up if expansion is unconstrained",
            "tier": "A",
        },
        {
            "paper_id": "coliee_task3",
            "venue": "COLIEE",
            "task": "Legal case/statute retrieval",
            "dataset": "COLIEE corpora",
            "retrieval_stack": "BM25 + neural",
            "graph_role": "optional legal relation priors",
            "chunking_role": "statute/article units",
            "metrics": "F1, recall, ranking metrics",
            "key_finding": "Hybrid retrieval outperforms single retriever",
            "risk": "benchmark overfitting",
            "tier": "A",
        },
        {
            "paper_id": "sar_eacl_2023",
            "venue": "EACL 2023",
            "task": "Statute retrieval with structure",
            "dataset": "statute retrieval benchmark",
            "retrieval_stack": "retriever + structure-aware graph",
            "graph_role": "legal structure signal",
            "chunking_role": "article/section aware",
            "metrics": "MRR, recall",
            "key_finding": "Structure-aware retrieval improves ranking quality",
            "risk": "domain transfer gap",
            "tier": "A",
        },
        {
            "paper_id": "drill_vlsp_2025_17",
            "venue": "VLSP 2025",
            "task": "Vietnamese legal retrieval",
            "dataset": "DRiLL",
            "retrieval_stack": "hybrid retrieval",
            "graph_role": "not central",
            "chunking_role": "legal structure helps precision",
            "metrics": "macro-F2, ranking metrics",
            "key_finding": "Vietnamese legal retrieval benefits from domain adaptation",
            "risk": "dataset-specific assumptions",
            "tier": "A",
        },
        {
            "paper_id": "drill_vlsp_2025_20",
            "venue": "VLSP 2025",
            "task": "two-stage retrieval + rerank",
            "dataset": "DRiLL",
            "retrieval_stack": "retrieve then rerank",
            "graph_role": "optional",
            "chunking_role": "retrieval granularity impacts rerank quality",
            "metrics": "macro-F2, nDCG",
            "key_finding": "Two-stage pipeline improves citation-oriented retrieval",
            "risk": "latency increase",
            "tier": "A",
        },
        {
            "paper_id": "justnlp_2025_19",
            "venue": "JusTNLP 2025",
            "task": "structure-aware legal chunking",
            "dataset": "legal corpora",
            "retrieval_stack": "chunking-focused",
            "graph_role": "not required",
            "chunking_role": "core contribution",
            "metrics": "retrieval and citation metrics",
            "key_finding": "Structure-aware chunking improves citation precision",
            "risk": "parser brittleness on noisy text",
            "tier": "A",
        },
        {
            "paper_id": "bm25_legal_2021",
            "venue": "arXiv",
            "task": "BM25 legal baseline",
            "dataset": "legal IR",
            "retrieval_stack": "BM25",
            "graph_role": "none",
            "chunking_role": "token-based retrieval",
            "metrics": "MAP, recall",
            "key_finding": "Sparse retrieval remains strong legal baseline",
            "risk": "semantic mismatch queries",
            "tier": "A",
        },
        {
            "paper_id": "graphrag_docs",
            "venue": "Docs",
            "task": "implementation flow",
            "dataset": "n/a",
            "retrieval_stack": "graph pipeline",
            "graph_role": "entity/community indexing",
            "chunking_role": "input segmentation guidance",
            "metrics": "n/a",
            "key_finding": "Clear dataflow for graph construction and query modes",
            "risk": "doc != legal-domain guarantee",
            "tier": "C",
        },
        {
            "paper_id": "coliee_2023_overview",
            "venue": "COLIEE 2023 overview",
            "task": "shared task analysis",
            "dataset": "COLIEE",
            "retrieval_stack": "hybrid + rerank patterns",
            "graph_role": "varies by team",
            "chunking_role": "statute/case granularity critical",
            "metrics": "task metrics",
            "key_finding": "Ensemble and staged retrieval are robust in legal IR",
            "risk": "cross-lingual portability",
            "tier": "A",
        },
        {
            "paper_id": "temporal_kg_preprint",
            "venue": "preprint",
            "task": "temporal legal KG",
            "dataset": "legal corpora",
            "retrieval_stack": "temporal graph",
            "graph_role": "versioned relation reasoning",
            "chunking_role": "document-level",
            "metrics": "temporal QA quality",
            "key_finding": "Temporal edge metadata improves as-of-date answers",
            "risk": "weaker evidence quality than peer-reviewed papers",
            "tier": "B",
        },
    ]
    fields = [
        "paper_id",
        "venue",
        "task",
        "dataset",
        "retrieval_stack",
        "graph_role",
        "chunking_role",
        "metrics",
        "key_finding",
        "risk",
        "tier",
    ]
    _write_csv(RESEARCH_DIR / "evidence_matrix.csv", rows, fields)


def _build_query_set():
    factual_queries = [
        "Điều kiện cấp giấy phép kinh doanh vận tải là gì?",
        "Thủ tục đăng ký hộ kinh doanh gồm những bước nào?",
        "Điều kiện thành lập công ty cổ phần theo luật hiện hành?",
        "Mức xử phạt khi không đăng ký tạm trú là bao nhiêu?",
        "Quy định về nghỉ hưu trước tuổi trong khu vực nhà nước?",
        "Điều kiện cấp sổ đỏ cho đất ở hiện nay?",
        "Quy trình cấp đổi căn cước công dân thực hiện ra sao?",
        "Thời hạn giải quyết hồ sơ đăng ký kết hôn là bao lâu?",
        "Doanh nghiệp phải nộp những loại thuế cơ bản nào?",
        "Điều kiện cấp phép xây dựng nhà ở riêng lẻ?",
        "Quy định về bảo hiểm xã hội bắt buộc cho người lao động?",
        "Thủ tục khai sinh cho trẻ em thực hiện ở đâu?",
        "Điều kiện vay vốn ưu đãi cho doanh nghiệp nhỏ và vừa?",
        "Mức phạt vi phạm nồng độ cồn đối với ô tô?",
        "Quy định về hợp đồng lao động xác định thời hạn?",
        "Thủ tục tách thửa đất tại khu vực đô thị?",
        "Điều kiện cấp giấy phép lao động cho người nước ngoài?",
        "Mức xử phạt hành vi quảng cáo sai sự thật?",
        "Quy trình đăng ký nhãn hiệu tại Việt Nam?",
        "Điều kiện hoàn thuế giá trị gia tăng đối với doanh nghiệp?",
    ]
    amendment_queries = [
        "Văn bản nào sửa đổi Nghị định 12/2020/NĐ-CP?",
        "Những văn bản đã bãi bỏ Thông tư 99/2015/TT-BTC?",
        "Nghị định 34/2018/NĐ-CP thay thế quy định nào trước đó?",
        "Văn bản nào bổ sung cho Luật Doanh nghiệp 2020?",
        "Những điều khoản bị sửa đổi trong Nghị định 45/2019/NĐ-CP?",
        "Thông tư nào thay thế Thông tư 22/2016/TT-BYT?",
        "Quy định nào đã hết hiệu lực do văn bản mới?",
        "Có văn bản nào bãi bỏ một phần Nghị định 10/2021/NĐ-CP không?",
        "Luật nào điều chỉnh bổ sung cho Luật Đất đai hiện hành?",
        "Nghị định nào cập nhật mức xử phạt trong lĩnh vực giao thông?",
        "Văn bản nào sửa đổi cơ chế thuế thu nhập doanh nghiệp?",
        "Thông tư thay thế hướng dẫn cũ về hóa đơn điện tử là gì?",
        "Những văn bản liên quan theo quan hệ sửa đổi đối với Luật Nhà ở?",
        "Nghị định nào bãi bỏ quy định cũ về phòng cháy chữa cháy?",
        "Văn bản mới nào thay thế quy định về đăng kiểm phương tiện?",
        "Luật nào sửa đổi, bổ sung Bộ luật Lao động gần đây?",
        "Thông tư nào điều chỉnh lại mức học phí theo khung mới?",
        "Nghị định nào thay thế quy định về xử phạt xây dựng?",
        "Văn bản nào bãi bỏ toàn bộ văn bản 15/2014/TT-BXD?",
        "Quan hệ sửa đổi giữa các văn bản về bảo hiểm y tế gồm những gì?",
    ]
    temporal_queries = [
        "Nghị định 12/2020/NĐ-CP còn hiệu lực đến ngày nào?",
        "Tại thời điểm 2021-01-01 văn bản này có hiệu lực không?",
        "Quy định này áp dụng từ khi nào?",
        "Điều khoản này có bị hết hiệu lực trong năm 2023 không?",
        "Văn bản nào có hiệu lực tại ngày 2024-06-01 về hóa đơn điện tử?",
        "Tại ngày 2020-12-31 quy định xử phạt cũ còn hiệu lực không?",
        "Văn bản thay thế bắt đầu có hiệu lực từ ngày nào?",
        "Trong giai đoạn 2019-2021 quy định nào được áp dụng?",
        "Đến ngày 2025-01-01 văn bản này còn hiệu lực không?",
        "Ngày có hiệu lực và ngày hết hiệu lực của Thông tư này là gì?",
        "Văn bản nào có hiệu lực tại thời điểm doanh nghiệp thành lập năm 2022?",
        "Tại tháng 03/2024 quy định về đăng kiểm áp dụng theo văn bản nào?",
        "Quy định này hết hiệu lực trước hay sau ngày 2024-12-31?",
        "Văn bản nào đang có hiệu lực hiện tại về bảo hiểm xã hội?",
        "Năm 2026 quy định xử phạt giao thông áp dụng theo văn bản nào?",
        "Tại ngày 2018-05-01 luật nào điều chỉnh thủ tục đầu tư?",
        "Văn bản có hiệu lực hồi tố trong lĩnh vực thuế không?",
        "As of 2023-07-01, regulation on e-invoice applies from which decree?",
        "Tại thời điểm ký hợp đồng năm 2020 áp dụng quy định nào?",
        "Văn bản nào có trạng thái expiring_soon trong 90 ngày tới?",
    ]

    rows = []
    idx = 1
    for q in factual_queries:
        rows.append({"query_id": f"Q{idx:03d}", "query_type": "factual", "query_text": q})
        idx += 1
    for q in amendment_queries:
        rows.append({"query_id": f"Q{idx:03d}", "query_type": "amendment", "query_text": q})
        idx += 1
    for q in temporal_queries:
        rows.append({"query_id": f"Q{idx:03d}", "query_type": "temporal", "query_text": q})
        idx += 1

    _write_csv(
        RESEARCH_DIR / "query_set_v1.csv",
        rows,
        ["query_id", "query_type", "query_text"],
    )

    eval_rows = []
    pair_id = 1
    for q_idx, row in enumerate(rows):
        for candidate_rank in range(1, 6):
            split = "dev" if pair_id <= 200 else "test"
            eval_rows.append(
                {
                    "pair_id": f"P{pair_id:04d}",
                    "query_id": row["query_id"],
                    "query_type": row["query_type"],
                    "query_text": row["query_text"],
                    "candidate_rank": candidate_rank,
                    "doc_id": "",
                    "relevance": "",
                    "citation_valid": "",
                    "notes": "",
                    "split": split,
                }
            )
            pair_id += 1

    _write_csv(
        RESEARCH_DIR / "eval_pairs_template.csv",
        eval_rows,
        [
            "pair_id",
            "query_id",
            "query_type",
            "query_text",
            "candidate_rank",
            "doc_id",
            "relevance",
            "citation_valid",
            "notes",
            "split",
        ],
    )


def _write_specs():
    graph_spec = """
# Graph Schema Spec (V1)

## Scope
- Default mode: document-level graph for legal impact retrieval.
- Upgrade target: provision-level graph after quality/latency gate passes.

## Node Types
- `Document`: canonical legal document (`doc_id` as unique key).
- `DocType`, `Agency`, `Signer`, `LawFamily`: metadata nodes.
- `Provision` remains available for phase-2 expansion.

## Edge Types
- Core legal-impact edges: `AMENDS`, `REPLACES`, `ABROGATES`, `CITES`.
- Metadata edges: `ISSUED_BY`, `SIGNED_BY`, `BELONGS_TO_FAMILY`, `OF_TYPE`, `TYPE_SUPERIOR_TO`.

## Mandatory Edge Properties (AMENDS/REPLACES/ABROGATES/CITES)
- `evidence_text`
- `evidence_source` (default: `mo_ta`)
- `confidence_rule`
- `snapshot_date`
- `updated_at`

## Expansion Rules
- Graph expansion happens after hybrid shortlist.
- V1 cap: `hops=1`.
- Modes:
  - `local`: includes `CITES + legal-impact edges` and same-type expansion.
  - `legal_impact`: only `AMENDS/REPLACES/ABROGATES`.
  - `temporal`: legal-impact edges with active/as-of filtering in retrieval layer.
"""
    _write_text(RESEARCH_DIR / "graph_schema_spec.md", graph_spec)

    chunking_spec = """
# Chunking Spec (V1)

## Objective
- Improve citation precision by deterministic legal structure parsing.
- Keep fallback deterministic and constrained to avoid cross-article leakage.

## Preprocessing
- Stabilize inline markers into line boundaries before regex parsing:
  - `Chuong/Chương`
  - `Dieu/Điều`
  - `Khoan/Khoản`
  - `Diem/Điểm`

## Chunk Types
- `ProvisionChunk`: article-level (`Dieu`).
- `ClauseChunk`: clause/point-level (`Khoan/Diem`).
- `FallbackChunk`: parser fallback when structure is missing or hard split is required.

## Mandatory Chunk Metadata
- `chunk_type`
- `legal_path`
- `segment_level`
- `fallback_reason`
- `token_len_est`
- `encoder_profile`
- `provision_id`
- `citation_full`
- `is_active`
- `status_label`
- `law_family`

## Invariants
- Do not split across `Dieu`.
- If long content requires split, split only within same provision.
- `citation_full` must always be populated.
"""
    _write_text(RESEARCH_DIR / "chunking_spec.md", chunking_spec)

    experiment_plan = """
title: "Legal Retrieval Research v2"
defaults:
  corpus:
    primary: "data/gold/vbpl_gold.parquet"
    archive: "data/gold/vbpl_gold_archive.parquet"
  priority: "citation_precision_first"
  deterministic_only: true
  graph_phase: "document_level_first"
variants:
  baseline:
    - id: "B0"
      name: "BM25 only"
      dense_weight: 0.0
      sparse_weight: 1.0
      reranker: false
      graph_mode: "none"
    - id: "B1"
      name: "Dense only"
      dense_weight: 1.0
      sparse_weight: 0.0
      reranker: false
      graph_mode: "none"
    - id: "B2"
      name: "Hybrid"
      dense_weight: 0.6
      sparse_weight: 0.4
      reranker: false
      graph_mode: "none"
    - id: "B3"
      name: "Hybrid + Reranker"
      dense_weight: 0.6
      sparse_weight: 0.4
      reranker: true
      graph_mode: "none"
  graph:
    - id: "G1"
      name: "B3 + local graph expansion"
      dense_weight: 0.6
      sparse_weight: 0.4
      reranker: true
      graph_mode: "local"
    - id: "G2"
      name: "B3 + legal-impact graph expansion"
      dense_weight: 0.6
      sparse_weight: 0.4
      reranker: true
      graph_mode: "legal_impact"
    - id: "G3"
      name: "B3 + temporal graph expansion"
      dense_weight: 0.6
      sparse_weight: 0.4
      reranker: true
      graph_mode: "temporal"
  chunking:
    - id: "C1"
      name: "fixed-size fallback split"
    - id: "C2"
      name: "article-only chunking"
    - id: "C3"
      name: "article + clause/point chunking"
    - id: "C4"
      name: "C3 + context header metadata"
metrics:
  retrieval:
    - "Recall@20"
    - "nDCG@10"
    - "MRR@10"
  citation:
    - "citation_valid_rate"
  latency:
    - "latency_p95_ms"
acceptance_gate:
  target_variant: "G2 + C3"
  compared_to: "B3"
  pass_if:
    - "Recall@20 improves >= 5%"
    - "OR nDCG@10 improves >= 3%"
  mandatory:
    - "citation_valid_rate >= 0.95"
    - "latency_p95 increase <= 35%"
backlog:
  sprint_1:
    name: "Data/Chunk"
    owner: "data_engineer"
    input: "gold_primary + gold_archive"
    output: "stable chunk payload with new metadata"
    done_criteria: ">=85% structured chunks, 100% citation_full"
  sprint_2:
    name: "Graph/Expansion"
    owner: "graph_engineer"
    input: "document metadata + relation extraction"
    output: "graph edges with evidence metadata"
    done_criteria: "relation precision >=0.85 on labeled set"
  sprint_3:
    name: "Benchmark/Report"
    owner: "ml_engineer"
    input: "query_set_v1 + eval_pairs labels"
    output: "benchmark_results + decision_log update"
    done_criteria: "all variants evaluated or marked graph-pending"
"""
    _write_text(RESEARCH_DIR / "experiment_plan.yaml", experiment_plan)


def _write_report_and_log():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""
# Research Survey Report (v2)

Generated: `{now}`

## Skills Applied
- `brainstorming-research-ideas`: tension hunting, adjacent possible, two-sentence test.
- `creative-thinking-for-research`: constraint manipulation, inversion/negation, reformulation.

## Hypotheses (12)

| ID | Framework | Hypothesis |
|---|---|---|
| H1 | Tension hunting | Citation precision and recall can both improve if graph expansion is gated by intent. |
| H2 | Tension hunting | Latency increase from graph expansion can be bounded by shortlist-first design. |
| H3 | Tension hunting | Deterministic parsing can match LLM extraction reliability for legal chunk boundaries. |
| H4 | Constraint manipulation | Forcing no cross-Dieu split increases citation validity. |
| H5 | Constraint manipulation | Restricting graph hops to 1 preserves relevance and controls noise. |
| H6 | Constraint manipulation | Document-level graph first reduces infra risk versus immediate provision-level graph. |
| H7 | Problem reformulation | Reframe retrieval from \"find text\" to \"find legally citable provision path\". |
| H8 | Problem reformulation | Treat temporal queries as filtering problem first, generation problem second. |
| H9 | Problem reformulation | Use chunk type as retrieval control signal instead of only text similarity. |
| H10 | Adjacent possible | Existing qdrant hybrid + reranker stack can evaluate B0..G3 without new infra. |
| H11 | Adjacent possible | Existing gold corpus quality is sufficient for v1 benchmark if labels are curated. |
| H12 | Adjacent possible | Edge metadata (`evidence/confidence/snapshot`) enables auditable legal-impact graph. |

## Candidate Options (Top 4)

| Option | Description | Citation Precision Impact | Feasibility | Infra Cost | Decision |
|---|---|---:|---:|---:|---|
| O1 | Hybrid + reranker + intent-gated graph + structure chunking | High | High | Medium | **Primary** |
| O2 | Hybrid + reranker + structure chunking, no graph | Medium-High | Very High | Low | Backup |
| O3 | Dense-only + graph expansion | Medium | Medium | Medium | Rejected |
| O4 | Sparse-only + legal-impact graph | Medium | High | Low-Med | Rejected |

## Two-Sentence Test
- Problem: Vietnamese legal retrieval struggles to maintain both recall and citation validity, especially for amendment and temporal questions where document relationships matter.
- Insight: We improve this by deterministic structure-aware chunking plus intent-gated document-level graph expansion after hybrid shortlist, which works because it constrains noisy expansion while preserving legal-impact signals.

## Risk Table

| Risk | Severity | Mitigation |
|---|---|---|
| Inline marker noise breaks parser | High | Add marker stabilization preprocessing and fallback reason tracking |
| Graph expansion adds irrelevant docs | High | Intent-gated modes + hop=1 + shortlist-first |
| Neo4j unavailable in dev | Medium | Mark graph variants as `graph-pending`, still run B0..B3 |
| Labeling bottleneck for 300 pairs | Medium | Use template with split dev/test and prioritize amendment/temporal first |
| Latency regression after rerank+graph | Medium | Track p95 in benchmark gate and enforce <=35% increase |
"""
    _write_text(RESEARCH_DIR / "research_survey_report.md", report)

    decision_log = f"""
# Decision Log

Updated: `{now}`

## Locked Defaults
- Data scope: `gold_primary + gold_archive` only.
- Priority: citation precision first.
- Extraction policy: deterministic/rule-based only (no LLM extraction for chunking/graph).
- Graph rollout: document-level first, provision-level in phase 2 after gate pass.
- Fallback policy: if Neo4j unavailable, run baseline/hybrid variants and mark graph variants `graph-pending`.

## Active Recommendation
- Retrieval profile: `B3` baseline + `G2 + C3` target.
- Graph mode for legal-impact queries: `legal_impact`.
- Chunking mode target: `ProvisionChunk + ClauseChunk` with strict citation metadata.

## Promotion Conditions
- `Recall@20` +5% or `nDCG@10` +3% over `B3`.
- `citation_valid_rate >= 95%`.
- `latency_p95_ms` increase <= 35%.
"""
    _write_text(RESEARCH_DIR / "decision_log.md", decision_log)


def _ensure_benchmark_contract():
    benchmark_path = RESEARCH_DIR / "benchmark_results.csv"
    if benchmark_path.exists():
        return
    _write_csv(
        benchmark_path,
        [],
        [
            "run_id",
            "variant_id",
            "embedding_model_id",
            "embedding_dim",
            "normalize_embeddings",
            "status",
            "num_queries",
            "Recall@20",
            "nDCG@10",
            "MRR@10",
            "citation_valid_rate",
            "latency_p95_ms",
            "encode_p95_ms",
            "notes",
            "created_at",
        ],
    )


def main() -> None:
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    _build_evidence_matrix()
    _build_query_set()
    _write_specs()
    _write_report_and_log()
    _ensure_benchmark_contract()
    print(f"[DONE] Research artifacts bootstrapped at: {RESEARCH_DIR}")


if __name__ == "__main__":
    main()
