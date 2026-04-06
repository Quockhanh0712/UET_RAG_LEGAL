"""Query rewrite submodules used by the legal query rewriter facade."""

from .gateway import call_rewrite_gateway
from .heuristics import extract_named_law_hints
from .parser import merge_question_units
from .quality_gate import evaluate_quality_gate, is_generic_main_act
from .synthesis import deduplicate_query_plan, synthesize_query_set

__all__ = [
    "call_rewrite_gateway",
    "extract_named_law_hints",
    "merge_question_units",
    "deduplicate_query_plan",
    "synthesize_query_set",
    "evaluate_quality_gate",
    "is_generic_main_act",
]
