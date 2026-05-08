# Task 06 - Query Rewrite Modularization

## Muc tieu

Tach `src/legal_query_rewriter.py` thanh workflow ro buoc:

- gateway
- heuristic extraction
- parser/merge
- synthesis
- quality gate

## Scope

Trong scope:

- public DTO models
- heuristic utilities
- extraction parse/merge
- retrieval query synthesis
- quality gate

Ngoai scope:

- khong doi public trace field names
- khong doi `QueryRewriteService` contract

## Proposed split

- `src/legal_rag/rewrite/models.py`
- `src/legal_rag/rewrite/gateway.py`
- `src/legal_rag/rewrite/heuristics.py`
- `src/legal_rag/rewrite/parser.py`
- `src/legal_rag/rewrite/synthesis.py`
- `src/legal_rag/rewrite/quality_gate.py`

Facade compatibility:

- `src/legal_query_rewriter.py` van expose `LegalQueryRewriter` va cac DTO public

## RED

Them/cap nhat tests cho:

1. heuristic extraction cho actor/condition/law hint
2. parse va merge extraction output
3. question unit synthesis
4. quality gate fallback
5. service wrapper fallback khi rewrite fail

Recommended tests:

- `test_heuristic_named_law_hints_extracts_expected_values`
- `test_merge_question_units_deduplicates_and_preserves_axis`
- `test_synthesize_query_plan_keeps_original_query`
- `test_quality_gate_marks_invalid_main_act`
- `test_query_rewrite_service_fallbacks_on_exception`

## GREEN

Implementation steps:

1. move DTO sang `models.py`
2. move heuristic methods sang `heuristics.py`
3. move parse/merge sang `parser.py`
4. move query synthesis + quality gate sang module rieng
5. giu `LegalQueryRewriter.rewrite_query()` la orchestration facade

## REFACTOR

- giam reliance vao private mutable state
- su dung pure function toi da cho heuristics/parser/synthesis
- dat test theo behavior cua tung submodule

## Exit criteria

- query rewrite co test theo tung lop logic
- trace output giu nguyen
- `QueryRewriteService` khong can sua public contract

## Dependency

- Task 01
- Task 04
