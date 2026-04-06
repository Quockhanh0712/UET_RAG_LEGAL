"""Chunking policy definitions (ingestion-side boundary)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingPolicy:
    min_chars: int = 1200
    max_chars: int = 2800
    overlap_chars: int = 120
    target_chunk_tokens: int = 420
    hard_cap_tokens: int = 448
    overlap_tokens: int = 64


def build_chunking_policy(
    *,
    min_chars: int,
    max_chars: int,
    overlap_chars: int,
    target_chunk_tokens: int,
    hard_cap_tokens: int,
    overlap_tokens: int,
) -> ChunkingPolicy:
    return ChunkingPolicy(
        min_chars=int(min_chars),
        max_chars=int(max_chars),
        overlap_chars=int(overlap_chars),
        target_chunk_tokens=int(target_chunk_tokens),
        hard_cap_tokens=int(hard_cap_tokens),
        overlap_tokens=int(overlap_tokens),
    )

