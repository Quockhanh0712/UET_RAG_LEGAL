# Quockhanh05 Embedding Lab

Isolated quick lab for testing `Quockhanh05/Vietnam_legal_embeddings` without changing the production backend, frontend, or Qdrant collection.

## What It Uses

- Embedding model: `Quockhanh05/Vietnam_legal_embeddings`
- Model max sequence length: `256`
- Chunk budget: target `220`, hard cap `256`
- Qdrant collection: `legal_documents_lab__quockhanh05_vietnam_legal_embeddings__d768`
- BM25 artifact: `data/qdrant_db/bm25__legal_documents_lab__quockhanh05_vietnam_legal_embeddings__d768.json`

## Environment

Use the same Qdrant credentials as the main project:

```powershell
$env:QDRANT_CLOUD_URL="https://<cluster-url>"
$env:QDRANT_API_KEY="<api-key>"
```

`QDRANT_URL` is also accepted as a fallback for the URL.

## Index The Lab Collection

```powershell
python experiments\quockhanh05_embedding_lab\index_lab_collection.py --force-recreate --max-docs 200
```

Useful options:

```powershell
python experiments\quockhanh05_embedding_lab\index_lab_collection.py --max-docs 50 --skip-smoke
python experiments\quockhanh05_embedding_lab\index_lab_collection.py --max-docs 200 --max-chunks 1000
```

The script indexes a balanced subset from:

- hinh su
- giao thong
- moi truong
- bien / hang hai
- lao dong

## Run The Mini Chat UI

```powershell
python -m uvicorn experiments.quockhanh05_embedding_lab.app:app --host 127.0.0.1 --port 8099
```

Open:

```text
http://127.0.0.1:8099
```

## Verification

```powershell
python -m py_compile experiments\quockhanh05_embedding_lab\index_lab_collection.py experiments\quockhanh05_embedding_lab\app.py
pytest tests/unit/test_qdrant_store_runtime.py tests/unit/test_backend_adapter.py
```
