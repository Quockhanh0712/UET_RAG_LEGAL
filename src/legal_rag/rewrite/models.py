"""Public DTO models for legal query rewrite workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

DEFAULT_QUESTION_FOCUS = "chu_the_chiu_trach_nhiem"


@dataclass
class SurfaceActor:
    name: str
    actor_type: str = "khong_ro"
    source_span: str = ""

    def to_public_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class KeyCondition:
    kind: str
    value: str
    source_span: str = ""

    def to_public_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class ClauseUnit:
    clause_label: str
    actor_refs: List[str] = field(default_factory=list)
    act_span: str = ""
    condition_refs: List[str] = field(default_factory=list)
    intent: str = "xac_dinh_chu_the"

    def to_public_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class QuestionUnit:
    question_label: str
    actor_refs: List[str] = field(default_factory=list)
    act_span: str = ""
    condition_refs: List[str] = field(default_factory=list)
    legal_axis: str = ""
    user_asked_text: str = ""
    intent: str = "xac_dinh_chu_the"

    def to_public_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class StructuredRewriteExtraction:
    needs_decomposition: bool = False
    main_act: str = ""
    surface_actors: List[SurfaceActor] = field(default_factory=list)
    conditions: List[KeyCondition] = field(default_factory=list)
    explicit_legal_axes: List[str] = field(default_factory=list)
    question_focus: str = DEFAULT_QUESTION_FOCUS
    clause_units: List[ClauseUnit] = field(default_factory=list)
    named_law_hints: List[str] = field(default_factory=list)
    question_units: List[QuestionUnit] = field(default_factory=list)
    main_act_candidate_reasons: List[str] = field(default_factory=list)


@dataclass
class QueryRewriteResult:
    original_query: str
    queries: List[str]
    status: str
    raw_response: str = ""
    missing_numbers: List[str] = field(default_factory=list)
    fallback_reason: str = ""
    latency_ms: float = 0.0
    rewrite_strategy: str = "structured_synthesis"
    quality_passed: bool = False
    retrieval_query_set: List[str] = field(default_factory=list)
    actor_mentions: List[str] = field(default_factory=list)
    issue_tags: List[str] = field(default_factory=list)
    question_focus: str = DEFAULT_QUESTION_FOCUS
    needs_decomposition: bool = False
    rejection_reasons: List[str] = field(default_factory=list)
    main_act: str = ""
    surface_actors: List[Dict[str, str]] = field(default_factory=list)
    conditions: List[Dict[str, str]] = field(default_factory=list)
    clause_units: List[Dict[str, object]] = field(default_factory=list)
    named_law_hints: List[str] = field(default_factory=list)
    question_units: List[Dict[str, object]] = field(default_factory=list)
    retrieval_query_plan: List[Dict[str, object]] = field(default_factory=list)
    main_act_candidate_reasons: List[str] = field(default_factory=list)
    decomposition_policy: str = "objective_question_unit"

    @property
    def rewrite_used(self) -> bool:
        effective_queries = self.retrieval_query_set or self.queries
        return self.status == "ok" and any(q != self.original_query for q in effective_queries)

    @property
    def query_facets(self) -> Dict[str, object]:
        return {
            "actors": list(self.actor_mentions),
            "issues": list(self.issue_tags),
            "focus": self.question_focus,
            "main_act": self.main_act,
            "conditions": [item.get("value", "") for item in self.conditions if str(item.get("value", "")).strip()],
        }

    @property
    def debug_payload(self) -> Dict[str, object]:
        return {
            "needs_decomposition": bool(self.needs_decomposition),
            "main_act": self.main_act,
            "main_act_candidate_reasons": list(self.main_act_candidate_reasons),
            "actor_mentions": list(self.actor_mentions),
            "surface_actors": list(self.surface_actors),
            "conditions": list(self.conditions),
            "issue_tags": list(self.issue_tags),
            "explicit_legal_axes": list(self.issue_tags),
            "question_focus": self.question_focus,
            "clause_units": list(self.clause_units),
            "named_law_hints": list(self.named_law_hints),
            "question_units": list(self.question_units),
            "retrieval_query_plan": list(self.retrieval_query_plan),
            "decomposition_policy": self.decomposition_policy,
            "rejection_reasons": list(self.rejection_reasons),
            "quality_fail_reasons": list(self.rejection_reasons),
        }

