# Master Plan

Updated: `2026-04-03`

## North star

Sau dot refactor nay, backend va RAG workflow phai dat 4 tieu chi:

1. route mong
2. workflow ro boundary
3. contract on dinh
4. test regression du bao duoc refactor tiep theo

## Phase map

| Task | Ten task | Dependency | Muc tieu chinh |
| --- | --- | --- | --- |
| 01 | test baseline and contracts | none | dong bang behavior hien tai |
| 02 | backend api modularization | 01 | tach schemas, routes, app factory |
| 03 | backend application services | 02 | route handler khong con business logic lon |
| 04 | rag pipeline orchestrator split | 01,03 | `LegalRAGPipeline` tro thanh facade/orchestrator |
| 05 | qdrant store modularization | 01,04 | tach payload, bm25, client, repository |
| 06 | query rewrite modularization | 01,04 | tach heuristic/parser/synthesis/quality gate |
| 07 | chunking and ingestion boundaries | 01 | tach ingestion policy khoi online workflow |
| 08 | runtime wiring and cleanup | 02-07 | facade cleanup, docs, final regression |

## Why thu tu nay

### Problem-first ly do

- neu chua co contract test, moi task refactor deu la "blind surgery"
- neu chua tach API/service layer, backend main se tiep tuc la nut co chai
- neu chua co service boundary, tach orchestrator ben duoi se khong giam coupling o tang tren

### Risk-first ly do

- `main.py` la diem vao cua toan he thong, nen can tach som nhung phai co test guard truoc
- `legal_rag_pipeline.py`, `qdrant_store.py`, `legal_query_rewriter.py` deu la core workflow, nen refactor sau khi service layer da ro
- `legal_chunker_v2.py` la ingestion-heavy; co the song song sau khi contract online da duoc dong bang

## Definition of done cho moi task

Moi task chi duoc xem la xong khi:

1. co test RED duoc viet hoac cap nhat truoc implementation
2. test GREEN
3. facade cu van pass contract
4. khong mo rong scope ngoai task file
5. da cap nhat tai lieu lien quan trong folder nay neu dependency thay doi

## Risk register

| Risk | Muc do | Giam thieu |
| --- | --- | --- |
| drift giua fake meta va meta thuc | high | task 01 bo sung builder/helper cho test metadata |
| route refactor gay pha response schema | high | contract tests cho `/api/chat`, `/api/search` |
| tach qdrant store gay loi runtime cache | high | giu `get_qdrant_store()` facade + runtime signature tests |
| tach rewrite logic gay mat trace | high | lock `rewrite_context`, `rewrite_debug`, `query_facets` |
| tach chunking qua som anh huong indexing | medium | de task 07 sau, va khong doi retrieval contract online |
| xoa dead code qua som | medium | facade deprecation, xoa o task 08 |

## Suggested implementation cadence

### Sprint A

- Task 01
- Task 02
- Task 03

### Sprint B

- Task 04
- Task 05
- Task 06

### Sprint C

- Task 07
- Task 08

## Execution status (2026-04-03)

- task 01: completed
- task 02: completed
- task 03: completed
- task 04: completed
- task 05: completed
- task 06: completed
- task 07: completed
- task 08: completed

Validation snapshot:

- roadmap regression suite: `44 passed, 20 skipped`
- broad system check (`tests/unit` + `tests/integration`): `78 passed, 20 skipped`

## Decision rules trong qua trinh lam

- neu task dang dong vao API + runtime + retrieval cung luc, task do qua to, phai tach nho them
- neu can doi public contract, phai sua task 01 truoc
- neu test moi can mock qua sau implementation detail, can them interface/facade thay vi mock sau hon nua
