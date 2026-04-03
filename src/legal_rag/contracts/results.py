"""Result DTO contracts for Legal RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LegalAnswer:
    answer: str
    citations: List[Dict]
    intent: str
    retrieval_time: float
    generation_time: float
    rerank_time: float = 0.0
    used_reranker: bool = False
    graph_expansion_used: bool = False
    graph_mode: str = "none"
    rewrite_time: float = 0.0
    rewrite_used: bool = False
    rewrite_status: str = "disabled"
    rewrite_fallback_reason: str = ""
    rewritten_queries: List[str] = field(default_factory=list)
    missing_numbers: List[str] = field(default_factory=list)
    per_subquery_hits: List[Dict[str, object]] = field(default_factory=list)
    retrieval_union_count: int = 0
    rewrite_mode: str = "original"
    rewrite_original_query: str = ""
    rewrite_queries_compact: List[str] = field(default_factory=list)
    rewrite_strategy: str = "structured_synthesis"
    rewrite_quality_passed: bool = False
    retrieval_query_set: List[str] = field(default_factory=list)
    retrieval_query_plan: List[Dict[str, object]] = field(default_factory=list)
    query_facets: Dict[str, object] = field(default_factory=dict)
    rewrite_debug: Dict[str, object] = field(default_factory=dict)


@dataclass
class LegalSearchResult:
    citations: List[Dict]
    intent: str
    retrieval_time: float
    rerank_time: float = 0.0
    used_reranker: bool = False
    graph_expansion_used: bool = False
    graph_mode: str = "none"
    rewrite_time: float = 0.0
    rewrite_used: bool = False
    rewrite_status: str = "disabled"
    rewrite_fallback_reason: str = ""
    rewritten_queries: List[str] = field(default_factory=list)
    missing_numbers: List[str] = field(default_factory=list)
    per_subquery_hits: List[Dict[str, object]] = field(default_factory=list)
    retrieval_union_count: int = 0
    rewrite_mode: str = "original"
    rewrite_original_query: str = ""
    rewrite_queries_compact: List[str] = field(default_factory=list)
    rewrite_strategy: str = "structured_synthesis"
    rewrite_quality_passed: bool = False
    retrieval_query_set: List[str] = field(default_factory=list)
    retrieval_query_plan: List[Dict[str, object]] = field(default_factory=list)
    query_facets: Dict[str, object] = field(default_factory=dict)
    rewrite_debug: Dict[str, object] = field(default_factory=dict)

    def format_with_citations(self) -> str:
        answer_text = str(getattr(self, "answer", "") or "")
        if not self.citations:
            return answer_text

        citation_text = "\n\n---\n\n**CAN CU PHAP LY:**\n\n"
        seen = set()
        visible_index = 1

        for cite in self.citations:
            key = (
                cite.get("doc_id", cite.get("van_ban_id", "")),
                cite.get("dieu_so", ""),
                cite.get("khoan_so", ""),
                cite.get("diem_so", ""),
            )
            if key in seen:
                continue
            seen.add(key)

            citation_full = cite.get("citation_full", "")
            if citation_full:
                citation_text += f"{visible_index}. {citation_full}\n"
                visible_index += 1
                continue

            ten_vb = cite.get("ten_van_ban", "Van ban phap luat")
            loai_vb = cite.get("loai_van_ban", "")
            co_quan = cite.get("co_quan", "")
            dieu = cite.get("dieu_so", "")
            khoan = cite.get("khoan_so", "")
            diem = cite.get("diem_so", "")

            detail_parts = []
            if dieu:
                detail_parts.append(f"Dieu {dieu}")
            if khoan and khoan != "0":
                detail_parts.append(f"Khoan {khoan}")
            if diem:
                detail_parts.append(f"Diem {diem})")

            source = f"[{loai_vb}] {ten_vb}" if loai_vb else ten_vb
            if co_quan:
                source = f"{source} - {co_quan}"

            if detail_parts:
                citation_text += f"{visible_index}. {' - '.join(detail_parts)} | {source}\n"
            else:
                citation_text += f"{visible_index}. {source}\n"
            visible_index += 1

        return answer_text + citation_text

