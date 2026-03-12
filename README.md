# UET RAG Legal

Legal Retrieval-Augmented Generation system for Vietnamese legal use cases.

## What Is In This Repo

- `src/`: core RAG pipeline, embeddings, retrieval, reranker, graph extraction.
- `scripts/`: data loading, benchmarking, migration, and research execution scripts.
- `web_app/`: FastAPI backend + React frontend.
- `tests/`: unit and integration tests.
- `docs/`: system documentation and handover materials.
- `data/gold/research` and `data/gold/reports`: lightweight research artifacts.

## Quick Start

### 1. Create environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional Chainlit demo:

```powershell
pip install -r requirements-chainlit.txt
```

### 2. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Then update `.env` values for your local setup (LLM provider, Qdrant, API keys).

### 3. Run backend

```powershell
python web_app\backend\main.py
```

### 4. Run frontend

```powershell
cd web_app\frontend
npm install
npm run dev
```

### 5. Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit -q
```

## Team Workflow (2 People)

- `main` is protected. No direct push for feature work.
- One task = one branch = one PR.
- Branch naming:
  - `feat/a-<topic>` for member A
  - `feat/b-<topic>` for member B
- PR requires 1 reviewer approval from the other member.
- Use squash merge.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for full commit and PR rules.

## Security Rules

- Never commit `.env` or raw secrets.
- Rotate any leaked key immediately (for example `QDRANT_API_KEY`).
- Do not commit local virtual env, build artifacts, node_modules, or large data files.
