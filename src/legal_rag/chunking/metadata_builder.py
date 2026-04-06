"""Chunk metadata helpers for ingestion chunking."""

from __future__ import annotations

import hashlib


def build_provision_id(doc_id: str, dieu_so: str = "", suffix: str = "") -> str:
    parts = [doc_id or "UNKNOWN"]
    if dieu_so:
        parts.append(f"D{dieu_so}")
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def generate_chunk_id(
    *,
    doc_id: str,
    dieu_so: str,
    chunk_index: int,
    text: str,
    overflow_index: int = 1,
) -> str:
    digest = hashlib.md5(str(text or "").encode("utf-8")).hexdigest()[:8]
    base = build_provision_id(doc_id, dieu_so=dieu_so)
    if overflow_index > 1:
        base = f"{base}_W{overflow_index}"
    return f"{base}_C{chunk_index}_{digest}"

