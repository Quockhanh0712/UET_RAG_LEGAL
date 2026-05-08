# Task 08 - Runtime Wiring And Cleanup

## Muc tieu

Don dep facade, wiring, docs, va stale tests sau khi cac task truoc da xong.

## Scope

Trong scope:

- cleanup import/facade
- review stale tests
- bo sung docs migration noi bo
- final regression pass

Ngoai scope:

- khong them feature moi
- khong doi public API contract neu khong bat buoc

## Cleanup targets

- `web_app/backend/main.py`
- `src/legal_rag_pipeline.py`
- `src/qdrant_store.py`
- `src/legal_query_rewriter.py`
- test stale nhu `tests/unit/test_core.py`

## RED

Truoc khi cleanup, can co regression suite pass cho:

- route contract
- service contract
- pipeline search/query contract
- qdrant runtime/payload contract
- rewrite contract
- chunking contract

## GREEN

Implementation steps:

1. xoa logic duplicate khoi facade cu
2. de lai wrapper public can thiet
3. deprecate test stale hoac viet lai theo structure moi
4. update README/docs neu entrypoint thay doi noi bo

## REFACTOR

- flatten import path neu bi long vong
- remove helper khong con duoc dung
- cap nhat cross-reference trong folder plan nay

## Exit criteria

- worktree sach, khong con module "nua cu nua moi" khong ro role
- regression suite then chot xanh
- README va docs khong noi sai structure moi

## Dependency

- Task 02
- Task 03
- Task 04
- Task 05
- Task 06
- Task 07
