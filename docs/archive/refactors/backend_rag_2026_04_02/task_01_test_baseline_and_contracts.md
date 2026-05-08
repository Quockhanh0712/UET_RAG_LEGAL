# Task 01 - Test Baseline And Contracts

## Muc tieu

Dong bang behavior hien tai cua backend va shared legal pipeline truoc khi refactor module.

## Tai sao task nay phai lam truoc

Hien tai da co drift giua test va implementation, the hien ro nhat o:

- `tests/unit/test_legal_rag_pipeline_search.py`

Neu bo qua task nay, cac task sau se khong biet minh dang giu contract nao.

## Scope

Trong scope:

- cap nhat/bo sung test unit cho contract metadata cua `LegalRAGPipeline`
- them contract tests cho `build_search_payload()` va `build_chat_payload()`
- review va phan loai test cu: active, stale, transitional
- tao helper fake result/meta thay vi lap dict thu cong nhieu noi

Ngoai scope:

- khong doi architecture runtime
- khong doi route implementation

## File du kien dong vao

- `tests/unit/test_legal_rag_pipeline_search.py`
- `tests/unit/test_backend_adapter.py` hoac file moi tuong duong
- `tests/unit/test_query_rewrite_service.py`
- `tests/integration/test_api.py` (chi neu can cap nhat contract that su dang dung)
- file helper test moi neu can

## RED

Viet/cap nhat test de cover:

1. `LegalRAGPipeline.search()` phai tolerate metadata day du theo contract hien tai
2. neu metadata thieu key optional, behavior fallback phai ro rang
3. `build_search_payload()` tra ve day du cac field trace
4. `build_chat_payload()` tra ve `fallback_used`, `fallback_error`, `collection`

Recommended first failing tests:

- `test_search_returns_full_retrieval_metadata_contract`
- `test_search_tolerates_missing_optional_rewrite_fields`
- `test_build_search_payload_exposes_trace_contract`
- `test_build_chat_payload_exposes_trace_contract`

## GREEN

Implementation duoc phep:

- bo sung helper metadata default
- neu can, sua `LegalRAGPipeline.search()` de dung `meta.get(...)` cho field optional
- bo sung test fixture chung

Rule:

- khong doi ten field public
- khong doi response schema

## REFACTOR

- tach test helper vao mot cho
- loai bo duplicate fake citation/meta setup
- danh dau test stale ro ly do

## Exit criteria

- subset test contract xanh
- failing test baseline ban dau da duoc xu ly ro rang
- co danh sach test stale can sua o task 08

## Dependency

- none
