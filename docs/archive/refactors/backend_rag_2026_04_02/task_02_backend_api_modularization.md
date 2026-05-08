# Task 02 - Backend API Modularization

## Muc tieu

Bien `web_app/backend/main.py` thanh entrypoint mong, tach route/schema/bootstrap ra module rieng.

## Scope

Trong scope:

- tach pydantic models ra `api/schemas.py`
- tach routers theo nhom endpoint
- tao `app_factory.py` hoac module tuong duong
- giu `main.py` de chay `uvicorn web_app.backend.main:app`

Ngoai scope:

- chua tach service logic sau route
- chua doi shared pipeline internals

## File du kien tao moi

- `web_app/backend/app_factory.py`
- `web_app/backend/api/schemas.py`
- `web_app/backend/api/routes/health.py`
- `web_app/backend/api/routes/legal_chat.py`
- `web_app/backend/api/routes/legal_search.py`
- `web_app/backend/api/routes/documents.py`
- `web_app/backend/api/routes/metrics.py`
- `web_app/backend/api/routes/settings.py`
- `web_app/backend/api/routes/ocr.py`

## RED

Them/cap nhat test cho:

1. app van expose dung endpoints chinh
2. import `app` tu `web_app.backend.main` van thanh cong
3. `GET /api/status`, `POST /api/search`, `POST /api/chat` van tra schema cu

Recommended tests:

- `test_main_exports_fastapi_app`
- `test_status_route_contract`
- `test_search_route_contract`
- `test_chat_route_contract`

## GREEN

Implementation steps:

1. move schemas ra `api/schemas.py`
2. move route functions vao router files
3. tao `create_app()` va register routers
4. de `main.py` chi load env, build app, giu `__main__` neu can

## REFACTOR

- remove import cycle
- move config-only helpers khoi route modules
- giu route module theo domain, khong theo HTTP method

## Exit criteria

- `main.py` khong con chua pydantic schema va route logic day dac
- route files co the doc lap
- public app entrypoint giu nguyen

## Dependency

- Task 01
