from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.document_ingestion.models import IngestOutcome, IngestedChunk


class _Row(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)


class _FakeIndexer:
    def ensure_collection(self):
        return None


class _FakeProcessor:
    def __init__(self):
        self.indexer = _FakeIndexer()
        self.deleted = []

    def ingest_saved_file(self, request):
        return IngestOutcome(
            status="ready",
            chunks=[
                IngestedChunk(
                    chunk_id=f"{request.doc_id}_chunk_1",
                    content="Noi dung da duoc index",
                    doc_id=request.doc_id,
                )
            ],
            parser_used="plain-text",
            ocr_provider=request.ocr_provider,
            page_count=1,
            text_chars=24,
        )

    def delete_document(self, *, doc_id, user_id, artifact_dir):
        self.deleted.append((doc_id, user_id, Path(artifact_dir)))


def _install_fake_persistence(monkeypatch, service_module):
    documents = {}
    jobs = {}

    def _save_uploaded_document(**kwargs):
        existing = documents.get(kwargs["doc_id"])
        payload = existing.to_dict() if existing is not None else {}
        payload.update(
            {
                "id": kwargs["doc_id"],
                "user_id": kwargs["user_id"],
                "session_id": kwargs["session_id"],
                "filename": kwargs["filename"],
                "file_type": kwargs["file_type"],
                "file_size": kwargs["file_size"],
                "chunk_count": kwargs.get("chunk_count", 0),
                "job_id": kwargs.get("job_id"),
                "ingest_mode": kwargs.get("ingest_mode"),
                "page_count": kwargs.get("page_count"),
                "text_chars": kwargs.get("text_chars"),
                "parser_used": kwargs.get("parser_used"),
                "ocr_provider": kwargs.get("ocr_provider"),
                "error_message": kwargs.get("error_message"),
                "upload_time": payload.get("upload_time", "2026-04-04T10:00:00"),
                "indexed_at": kwargs.get("indexed_at"),
                "updated_at": "2026-04-04T10:00:01",
                "status": kwargs.get("status", "queued"),
            }
        )
        row = _Row(**payload)
        documents[kwargs["doc_id"]] = row
        return row

    def _save_document_ingest_job(**kwargs):
        existing = jobs.get(kwargs["job_id"])
        payload = existing.to_dict() if existing is not None else {}
        payload.update(
            {
                "id": kwargs["job_id"],
                "job_id": kwargs["job_id"],
                "doc_id": kwargs["doc_id"],
                "user_id": kwargs["user_id"],
                "session_id": kwargs["session_id"],
                "filename": kwargs["filename"],
                "content_mode": kwargs["content_mode"],
                "ocr_provider": kwargs.get("ocr_provider"),
                "parser_used": kwargs.get("parser_used"),
                "status": kwargs["status"],
                "error_message": kwargs.get("error_message"),
                "created_at": payload.get("created_at", "2026-04-04T10:00:00"),
                "updated_at": "2026-04-04T10:00:01",
                "started_at": kwargs.get("started_at"),
                "completed_at": kwargs.get("completed_at"),
            }
        )
        row = _Row(**payload)
        jobs[kwargs["job_id"]] = row
        return row

    monkeypatch.setattr(service_module, "init_db", lambda: None)
    monkeypatch.setattr(service_module, "save_uploaded_document", _save_uploaded_document)
    monkeypatch.setattr(service_module, "save_document_ingest_job", _save_document_ingest_job)
    monkeypatch.setattr(
        service_module,
        "get_uploaded_document",
        lambda doc_id, user_id=None: next(
            (
                row
                for row in documents.values()
                if row.id == doc_id and (user_id is None or row.user_id == user_id)
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        service_module,
        "get_document_ingest_job_for_doc",
        lambda doc_id: next((row for row in jobs.values() if row.doc_id == doc_id), None),
    )
    monkeypatch.setattr(
        service_module,
        "list_uploaded_documents",
        lambda user_id, session_id=None: [
            row
            for row in documents.values()
            if row.user_id == user_id and (session_id is None or row.session_id == session_id)
        ],
    )
    monkeypatch.setattr(service_module, "delete_uploaded_document", lambda doc_id, user_id: documents.pop(doc_id, None))

    def _delete_document_ingest_jobs(doc_id):
        to_delete = [job_id for job_id, row in jobs.items() if row.doc_id == doc_id]
        for job_id in to_delete:
            jobs.pop(job_id, None)
        return len(to_delete)

    monkeypatch.setattr(service_module, "delete_document_ingest_jobs", _delete_document_ingest_jobs)

    return documents, jobs


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_route_returns_job_and_status_then_reaches_ready(monkeypatch, tmp_path):
    from backend.api.routes import documents as documents_routes
    from backend import document_ingestion_service as service_module

    monkeypatch.setattr(service_module.settings, "UPLOAD_DIR", tmp_path / "uploads")
    documents, jobs = _install_fake_persistence(monkeypatch, service_module)

    service = service_module.DocumentIngestionService(processor=_FakeProcessor())
    service._initialized = False

    monkeypatch.setattr(documents_routes, "get_document_ingestion_service", lambda: service)

    app = FastAPI()
    app.include_router(documents_routes.router)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/upload",
            files={"file": ("note.txt", b"Tai lieu can xu ly bat dong bo", "text/plain")},
            data={
                "user_id": "user-1",
                "session_id": "session-1",
                "content_mode": "text",
                "ocr_provider": "default",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["status"] == "queued"
        assert payload["job_id"]
        assert payload["chunk_count"] == 0

        status_payload = None
        for _ in range(20):
            poll = await client.get(f"/api/documents/{payload['doc_id']}/status?user_id=user-1")
            assert poll.status_code == 200
            status_payload = poll.json()
            if status_payload["status"] == "ready":
                break
            await asyncio.sleep(0.01)

        assert status_payload is not None
        assert status_payload["status"] == "ready"
        assert status_payload["job_id"] == payload["job_id"]
        assert status_payload["parser_used"] == "plain-text"
        assert status_payload["chunk_count"] == 1

        list_response = await client.get("/api/documents?user_id=user-1")
        assert list_response.status_code == 200
        documents_payload = list_response.json()["documents"]
        assert len(documents_payload) == 1
        assert documents_payload[0]["status"] == "ready"
        assert documents_payload[0]["ingest_mode"] == "text"

        delete_response = await client.delete(f"/api/documents/{payload['doc_id']}?user_id=user-1")
        assert delete_response.status_code == 200
        assert payload["doc_id"] not in documents
        assert all(job.doc_id != payload["doc_id"] for job in jobs.values())
        assert service.processor.deleted[0][0] == payload["doc_id"]
