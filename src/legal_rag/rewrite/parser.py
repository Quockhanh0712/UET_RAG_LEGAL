"""Parser/merge utilities for structured rewrite extraction."""

from __future__ import annotations

from typing import Callable, List, Sequence


def merge_question_units(
    preferred: Sequence[object],
    fallback: Sequence[object],
    *,
    normalize_key_fn: Callable[[str], str],
    max_units: int = 3,
) -> List[object]:
    merged: List[object] = []
    seen = set()

    for group in (preferred, fallback):
        for unit in group:
            intent = normalize_key_fn(getattr(unit, "intent", ""))
            legal_axis = normalize_key_fn(getattr(unit, "legal_axis", ""))
            act_span = normalize_key_fn(getattr(unit, "act_span", ""))
            actor_refs = ",".join(
                sorted(normalize_key_fn(actor) for actor in getattr(unit, "actor_refs", []))
            )
            condition_refs = ",".join(
                sorted(normalize_key_fn(condition) for condition in getattr(unit, "condition_refs", []))
            )
            key = "|".join([intent, legal_axis, act_span, actor_refs, condition_refs])
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(unit)
            if len(merged) >= max_units:
                return merged

    return merged

