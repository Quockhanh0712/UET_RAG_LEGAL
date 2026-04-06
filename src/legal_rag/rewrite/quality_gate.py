"""Quality gate checks for rewrite query plans."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

GENERIC_MAIN_ACT_PATTERNS = (
    "hành vi được nêu",
    "hanh vi duoc neu",
    "hành vi chính",
    "hanh vi chinh",
    "tình huống được nêu",
    "tinh huong duoc neu",
)


def is_generic_main_act(value: str, *, normalize_key_fn: Callable[[str], str]) -> bool:
    normalized = normalize_key_fn(value)
    if not normalized:
        return True
    return any(token in normalized for token in GENERIC_MAIN_ACT_PATTERNS)


def evaluate_quality_gate(
    *,
    extraction: object,
    query_plan: Sequence[Dict[str, object]],
    original_query: str,
    normalize_key_fn: Callable[[str], str],
    normalize_text_fn: Callable[[str], str],
    is_generic_main_act_fn: Callable[[str], bool],
    drift_tokens: Sequence[str],
) -> List[str]:
    reasons: List[str] = []
    main_act = str(getattr(extraction, "main_act", "") or "")
    surface_actors = list(getattr(extraction, "surface_actors", []) or [])
    question_units = list(getattr(extraction, "question_units", []) or [])
    conditions = list(getattr(extraction, "conditions", []) or [])
    named_law_hints = list(getattr(extraction, "named_law_hints", []) or [])

    if not main_act:
        if surface_actors:
            reasons.append("missing_main_act_for_scenario")
    elif is_generic_main_act_fn(main_act):
        reasons.append("generic_main_act")

    if not query_plan:
        reasons.append("empty_query_plan")
        return sorted(set(reasons))

    original_key = normalize_key_fn(original_query)
    main_act_key = normalize_key_fn(main_act)
    actor_keys = [normalize_key_fn(getattr(actor, "name", "")) for actor in surface_actors]
    condition_values = [
        getattr(condition, "value", "")
        for condition in conditions
        if getattr(condition, "kind", "") in {"tuoi", "so_tien", "so_lan", "quan_he_lao_dong"}
    ]
    matched_units = set()

    for item in query_plan:
        query = normalize_text_fn(str(item.get("query", "")))
        query_key = normalize_key_fn(query)
        if not query_key:
            reasons.append("blank_query")
            continue
        if main_act_key and main_act_key not in query_key:
            reasons.append("query_missing_main_act")
        if "hanh vi duoc neu" in query_key or "tinh huong duoc neu" in query_key:
            reasons.append("generic_query_template")
        if any(token in query_key for token in drift_tokens):
            reasons.append("topic_drift")
        if len(query_key.split()) < 5:
            reasons.append("query_too_short")
        if actor_keys and all(token not in query_key for token in actor_keys if token):
            reasons.append("query_missing_actor_anchor")
        if named_law_hints:
            hint_hits = sum(1 for hint in named_law_hints if normalize_key_fn(hint) in query_key)
            main_hits = 1 if main_act_key and main_act_key in query_key else 0
            if hint_hits and not main_hits:
                reasons.append("named_law_override")
        if any(token in query_key for token in ("co toi", "pham toi", "vo toi", "nan nhan", "duoc mien trach nhiem hinh su")):
            if not any(token in query_key for token in ("dieu kien", "xem xet", "co duoc", "co phai")):
                reasons.append("premature_legal_conclusion")
        source_label = normalize_key_fn(str(item.get("source_question_unit", "")))
        if any(source_label == normalize_key_fn(getattr(unit, "question_label", "")) for unit in question_units):
            matched_units.add(source_label)

    for value in condition_values:
        value_key = normalize_key_fn(value)
        if value_key and not any(value_key in normalize_key_fn(str(item.get("query", ""))) for item in query_plan):
            question_sensitive = any(
                value_key in normalize_key_fn(getattr(unit, "user_asked_text", ""))
                or value_key in normalize_key_fn(" ".join(getattr(unit, "condition_refs", []) or []))
                for unit in question_units
            )
            if question_sensitive:
                reasons.append("condition_not_preserved")
                break

    if question_units and len(matched_units) < len(question_units):
        reasons.append("question_unit_coverage_failed")
    if main_act and normalize_key_fn(main_act) not in original_key:
        reasons.append("main_act_not_grounded_in_original_query")

    return sorted(set(reasons))

