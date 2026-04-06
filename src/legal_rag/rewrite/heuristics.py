"""Heuristic helpers for legal query rewrite."""

from __future__ import annotations

import re
from typing import Callable, Iterable, List, Sequence

from src.legal_data_processing import strip_accents


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().strip('"“”')


def normalize_key(text: str) -> str:
    return strip_accents(normalize_text(text)).lower()


def extract_named_law_hints(
    query: str,
    *,
    law_hint_patterns: Sequence[str],
    normalize_text_fn: Callable[[str], str] = normalize_text,
    normalize_key_fn: Callable[[str], str] = normalize_key,
    limit: int = 4,
) -> List[str]:
    hints: List[str] = []
    seen = set()
    for pattern in law_hint_patterns:
        for match in re.finditer(pattern, str(query or "")):
            value = normalize_text_fn(match.group(1)).rstrip(" ,.;:?!")
            key = normalize_key_fn(value)
            if not key or key in seen:
                continue
            seen.add(key)
            hints.append(value)
            if len(hints) >= limit:
                return hints
    return hints


def dedupe_values(
    values: Iterable[str],
    *,
    normalize_key_fn: Callable[[str], str] = normalize_key,
    limit: int = 10,
) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        text = normalize_text(value)
        key = normalize_key_fn(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= limit:
            break
    return output

