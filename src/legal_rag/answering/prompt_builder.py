"""Prompt construction helpers for Legal RAG answering."""

from __future__ import annotations


def build_answer_prompt(query: str, context) -> str:
    if isinstance(context, dict):
        query_context = str(context.get("query_context") or "").strip()
        evidence_context = str(context.get("evidence_context") or "").strip()
        return (
            "TRA LOI DUNG DINH DANG SAU:\n"
            "KET_LUAN:\n"
            "CAN_CU:\n"
            "PHAN_TICH_NGAN:\n"
            "CHUA_DU_CAN_CU:\n\n"
            f"QUERY_CONTEXT:\n{query_context}\n\n"
            f"EVIDENCE_CONTEXT:\n{evidence_context}\n\n"
            f"CAU HOI_GOC: {query}\n\nTRA LOI:"
        )
    return (
        "TRA LOI DUNG DINH DANG SAU:\n"
        "KET_LUAN:\n"
        "CAN_CU:\n"
        "PHAN_TICH_NGAN:\n"
        "CHUA_DU_CAN_CU:\n\n"
        f"CONTEXT:\n{context}\n\nCAU HOI: {query}\n\nTRA LOI:"
    )

