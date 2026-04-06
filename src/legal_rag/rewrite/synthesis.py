"""Query synthesis helpers for rewrite planning."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence


def deduplicate_query_plan(
    plan: Sequence[Dict[str, object]],
    *,
    normalize_key_fn: Callable[[str], str],
    max_subqueries: int,
) -> List[Dict[str, object]]:
    seen = set()
    deduped: List[Dict[str, object]] = []
    for item in plan:
        key = normalize_key_fn(str(item.get("query", "")))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(dict(item))
        if len(deduped) >= max_subqueries:
            break
    return deduped


def synthesize_query_set(
    *,
    original_query: str,
    query_plan: Sequence[Dict[str, object]],
    max_queries: int,
) -> List[str]:
    queries: List[str] = []
    for item in query_plan:
        query = str(item.get("query", "")).strip()
        if query:
            queries.append(query)
    if original_query not in queries:
        queries.append(original_query)
    return list(dict.fromkeys(queries))[:max_queries]

