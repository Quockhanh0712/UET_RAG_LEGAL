"""Context and citation formatting helpers for Legal RAG pipeline."""

from __future__ import annotations

from typing import Dict, List, Tuple


def build_rewrite_context_block(
    *,
    original_query: str,
    rewrite_mode: str,
    rewrite_status: str,
    rewrite_quality_passed: bool,
    rewrite_queries_compact: List[str],
    rewrite_fallback_reason: str,
    query_facets: Dict[str, object],
    rewrite_debug: Dict[str, object],
) -> str:
    lines = [
        f"ORIGINAL_QUERY: {str(original_query or '').strip()}",
        f"REWRITE_MODE: {str(rewrite_mode or 'original').strip()}",
        f"REWRITE_STATUS: {str(rewrite_status or 'disabled').strip()}",
        f"REWRITE_QUALITY_PASSED: {str(bool(rewrite_quality_passed)).lower()}",
    ]
    facets = dict(query_facets or {})
    debug = dict(rewrite_debug or {})
    actor_values = ", ".join(
        str(item) for item in list(facets.get("actors", []) or []) if str(item or "").strip()
    ) or "<none>"
    issue_values = ", ".join(
        str(item) for item in list(facets.get("issues", []) or []) if str(item or "").strip()
    ) or "<none>"
    focus_value = str(facets.get("focus") or "<none>")
    main_act_value = str(facets.get("main_act") or debug.get("main_act") or "<none>")
    condition_values = ", ".join(
        str(item) for item in list(facets.get("conditions", []) or []) if str(item or "").strip()
    ) or "<none>"
    lines.append(f"QUERY_FACETS_ACTORS: {actor_values}")
    lines.append(f"QUERY_FACETS_ISSUES: {issue_values}")
    lines.append(f"QUERY_FOCUS: {focus_value}")
    lines.append(f"MAIN_ACT: {main_act_value}")
    lines.append(f"KEY_CONDITIONS: {condition_values}")
    named_law_hints = ", ".join(
        str(item) for item in list(debug.get("named_law_hints", []) or []) if str(item or "").strip()
    ) or "<none>"
    lines.append(f"NAMED_LAW_HINTS: {named_law_hints}")
    question_units = list(debug.get("question_units", []) or [])
    if question_units:
        for idx, unit in enumerate(question_units, start=1):
            lines.append(f"QUESTION_UNIT_{idx}_LABEL: {str(unit.get('question_label') or '<none>')}")
            lines.append(f"QUESTION_UNIT_{idx}_ASKED: {str(unit.get('user_asked_text') or '<none>')}")
            lines.append(f"QUESTION_UNIT_{idx}_INTENT: {str(unit.get('intent') or '<none>')}")
    if rewrite_queries_compact:
        for idx, query in enumerate(rewrite_queries_compact, start=1):
            lines.append(f"RETRIEVAL_SUBQUERY_{idx}: {query}")
    else:
        lines.append("RETRIEVAL_SUBQUERY_1: <none>")
    if rewrite_fallback_reason:
        lines.append(f"REWRITE_FALLBACK_REASON: {rewrite_fallback_reason}")
    lines.append(
        "REWRITE_GUARDRAIL: Use rewrite only to understand intent; answer must follow ORIGINAL_QUERY and EVIDENCE_CONTEXT."
    )
    return "\n".join(lines)


def format_context(results: List) -> Tuple[str, List[Dict]]:
    if not results:
        return "", []

    context_parts = []
    citations = []

    for index, result in enumerate(results):
        meta = result.metadata
        header_parts = [f"[{index + 1}]"]

        if meta.get("citation_full"):
            header_parts.append(meta["citation_full"])
        else:
            if meta.get("loai_van_ban"):
                header_parts.append(f"[{meta['loai_van_ban']}]")
            if meta.get("ten_van_ban"):
                header_parts.append(meta["ten_van_ban"])
            if meta.get("dieu_so"):
                header_parts.append(f"Dieu {meta['dieu_so']}")
            if meta.get("khoan_so"):
                header_parts.append(f"Khoan {meta['khoan_so']}")
            if meta.get("diem_so"):
                header_parts.append(f"Diem {meta['diem_so']})")

        context_parts.append(f"{' | '.join(header_parts)}\n{result.content}")
        citations.append(
            {
                "content": result.content,
                "doc_id": meta.get("doc_id", meta.get("van_ban_id", "")),
                "van_ban_id": meta.get("van_ban_id", ""),
                "so_ky_hieu": meta.get("so_ky_hieu", ""),
                "ten_van_ban": meta.get("ten_van_ban", ""),
                "loai_van_ban": meta.get("loai_van_ban", ""),
                "co_quan": meta.get("co_quan", meta.get("co_quan_ban_hanh", "")),
                "law_family": meta.get("law_family", ""),
                "chuong": meta.get("chuong", ""),
                "ten_chuong": meta.get("ten_chuong", ""),
                "dieu_so": meta.get("dieu_so", ""),
                "tieu_de_dieu": meta.get("tieu_de_dieu", ""),
                "khoan_so": meta.get("khoan_so", ""),
                "diem_so": meta.get("diem_so", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "segment_level": meta.get("segment_level", ""),
                "legal_path": meta.get("legal_path", meta.get("path", "")),
                "is_active": meta.get("is_active", True),
                "citation_base": meta.get("citation_base", ""),
                "citation_full": meta.get("citation_full", ""),
                "token_len_est": meta.get("token_len_est", 0),
                "encoder_profile": meta.get("encoder_profile", ""),
                "score": result.score,
            }
        )

    return "\n\n---\n\n".join(context_parts), citations

