# Backend + RAG Refactor Plan Index

Updated: `2026-04-03`

## Muc tieu

Refactor backend va RAG workflow theo huong:

- tach ro module theo responsibility
- giu nguyen API contract va tracing quan trong
- them safety net bang test truoc khi doi wiring
- thuc hien theo task nho, co TDD cycle ro rang

## Skills da ap dung

- `find-skills`
  - xac nhan khong can bo sung skill ngoai danh sach user da chi dinh
- `brainstorming-research-ideas`
  - dung cho framing problem-first, tension analysis, dependency sequencing
- `reference-builder`
  - dung de tao bo tai lieu ke hoach co cross-reference va exit criteria ro rang
- `tdd-workflow`
  - moi task deu co RED -> GREEN -> REFACTOR
- `backend-architect`
  - dung de chia module theo boundary va contract backend thuc te

## Folder convention

Tat ca file planning cho dot refactor nay nam trong cung mot folder:

- `docs/refactor_backend_rag_2026_04_02/`

Moi task co mot file markdown rieng de de chi dinh thi cong.

## File map

- [01_current_state_audit.md](./01_current_state_audit.md)
- [02_target_architecture.md](./02_target_architecture.md)
- [03_master_plan.md](./03_master_plan.md)
- [04_completion_report_2026_04_03.md](./04_completion_report_2026_04_03.md)
- [05_refactor_cleanup_inventory_2026_04_03.md](./05_refactor_cleanup_inventory_2026_04_03.md)
- [task_01_test_baseline_and_contracts.md](./task_01_test_baseline_and_contracts.md)
- [task_02_backend_api_modularization.md](./task_02_backend_api_modularization.md)
- [task_03_backend_application_services.md](./task_03_backend_application_services.md)
- [task_04_rag_pipeline_orchestrator_split.md](./task_04_rag_pipeline_orchestrator_split.md)
- [task_05_qdrant_store_modularization.md](./task_05_qdrant_store_modularization.md)
- [task_06_query_rewrite_modularization.md](./task_06_query_rewrite_modularization.md)
- [task_07_chunking_and_ingestion_boundaries.md](./task_07_chunking_and_ingestion_boundaries.md)
- [task_08_runtime_wiring_and_cleanup.md](./task_08_runtime_wiring_and_cleanup.md)

## Current status

Roadmap completed: `8/8 tasks done`.

Regression summary:

- focused roadmap suite: `44 passed, 20 skipped`
- wide check (`tests/unit` + `tests/integration`): `78 passed, 20 skipped`

## Thu tu khuyen nghi

1. Task 01
2. Task 02
3. Task 03
4. Task 04
5. Task 05
6. Task 06
7. Task 07
8. Task 08

## Baseline history

Baseline fail (`5 passed, 1 failed`) in early phase is now closed.

- root cause was metadata drift between `LegalRAGPipeline.search()` and test doubles
- fixed during task 01 contract freeze and verified by full regression in completion report

## Cach su dung bo plan nay

Khi bat dau implement, user co the chi dinh theo kieu:

- `lam task 01`
- `bat dau tu task 02`
- `lam task 04 truoc`

Moi task file da co:

- scope
- dependency
- file du kien tac dong
- RED tests
- GREEN implementation
- REFACTOR cleanup
- exit criteria
