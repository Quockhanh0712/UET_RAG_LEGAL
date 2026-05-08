# Task 03 - Backend Application Services

## Muc tieu

Tach business logic khoi route handler, dua backend sang model:

- route -> application service -> shared pipeline / legacy infrastructure

## Scope

Trong scope:

- shared legal search service
- shared legal chat service
- legacy hybrid/legal/user search service
- document/session/metrics services neu route dang chua logic lon

Ngoai scope:

- chua doi sau vao `src/legal_rag_pipeline.py`
- chua tach qdrant/rewrite/chunker

## Service candidates

- `SharedLegalSearchService`
- `SharedLegalChatService`
- `LegacySearchService`
- `DocumentService`
- `MetricsService`
- `SessionService`

## RED

Them test service-level cho:

1. route handler goi service, khong build payload phuc tap tai route
2. `SharedLegalSearchService` goi `get_legal_pipeline()` + `build_search_payload()`
3. `SharedLegalChatService` luu chat history dung cho shared path
4. fallback behavior duoc test o service thay vi route

Recommended tests:

- `test_shared_legal_search_service_builds_contract_response`
- `test_shared_legal_chat_service_persists_history`
- `test_chat_route_delegates_to_service`
- `test_search_route_delegates_to_service`

## GREEN

Implementation steps:

1. extract `_run_shared_legal_search()` thanh service
2. extract `_run_shared_legal_chat()` thanh service
3. move save-history logic vao service/helper
4. route chi parse request va return response

## REFACTOR

- giam utility leakage trong route
- gom dependency wiring vao constructor/factory
- dat ten service theo use case, khong theo tool

## Exit criteria

- route handler ngan
- co test service doc lap
- shared legal va legacy path co boundary ro

## Dependency

- Task 02
