# Legal RAG Web App

Web UI and FastAPI backend for the shared legal runtime.

## Source of truth

- frontend: `web_app/frontend/`
- backend: `web_app/backend/`
- shared runtime: `src/`

The production backend path is the shared runtime controlled by:

```env
BACKEND_USE_SHARED_LEGAL_PIPELINE=true
```

With that flag enabled:

- `search_mode=legal|user|hybrid` goes through the shared backend orchestration
- backend startup does not preload the legacy embedding model
- embedding defaults come from the contract resolver and default to `truro7/vn-law-embedding`

## Run backend

```powershell
$env:BACKEND_USE_SHARED_LEGAL_PIPELINE='true'
.\.venv\Scripts\python.exe -m uvicorn web_app.backend.main:app --host 127.0.0.1 --port 8080
```

## Run frontend

```powershell
cd web_app\frontend
npm install
npm run dev
```

## Main endpoints

- `GET /api/status`
- `POST /api/search`
- `POST /api/chat`
- `POST /api/upload`
- `GET /api/documents`

## Notes

- `web_app/frontend/node_modules/` and `web_app/frontend/dist/` are local artifacts, not repo source.
- Historical demo docs were moved under `docs/archive/`.
