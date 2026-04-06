"""
User document store facade backed by the unified ingestion module.

This keeps the legacy `get_user_store()` surface stable for chat search paths,
while removing the old in-memory registry and smart chunker upload path.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .config import settings
    from .database import (
        delete_document_ingest_jobs,
        delete_uploaded_document,
        list_uploaded_documents,
        save_document_ingest_job,
        save_uploaded_document,
    )
    from .document_ingestion_service import get_document_ingestion_service
except ImportError:
    from config import settings
    from database import (
        delete_document_ingest_jobs,
        delete_uploaded_document,
        list_uploaded_documents,
        save_document_ingest_job,
        save_uploaded_document,
    )
    from document_ingestion_service import get_document_ingestion_service

from src.document_ingestion.models import IngestRequest

logger = logging.getLogger(__name__)


class UserDocumentStore:
    COLLECTION_NAME = "user_docs_private"

    def __init__(self):
        self._service = get_document_ingestion_service()
        self._indexer = self._service.processor.indexer
        self._artifacts = self._service.artifact_store

    def ensure_collection(self):
        self._service.initialize()
        return True

    async def upload_file(
        self,
        file_path: str,
        user_id: str,
        session_id: str,
        filename: Optional[str] = None,
        content_mode: str = "text",
        ocr_provider: str = "default",
    ) -> Tuple[bool, str, int]:
        """
        Legacy synchronous upload fallback.

        New API routes use the background ingestion service directly. This
        method exists to avoid breaking older direct callers inside the repo.
        """

        self._service.initialize()
        path = Path(file_path)
        safe_filename = Path(filename or path.name).name
        file_size = path.stat().st_size
        file_type = path.suffix.lower().lstrip(".")
        doc_id = f"user_{user_id}_{uuid.uuid4().hex[:12]}"
        job_id = uuid.uuid4().hex
        artifact_dir = self._artifacts.document_dir(user_id, session_id, doc_id)
        original_path = self._artifacts.original_file_path(user_id, session_id, doc_id, safe_filename)
        original_path.write_bytes(path.read_bytes())
        upload_time = datetime.now().isoformat()

        save_uploaded_document(
            doc_id=doc_id,
            user_id=user_id,
            session_id=session_id,
            filename=safe_filename,
            file_type=file_type,
            file_size=file_size,
            chunk_count=0,
            job_id=job_id,
            ingest_mode=content_mode,
            ocr_provider=ocr_provider,
            status="running",
        )
        save_document_ingest_job(
            job_id=job_id,
            doc_id=doc_id,
            user_id=user_id,
            session_id=session_id,
            filename=safe_filename,
            content_mode=content_mode,
            status="running",
            ocr_provider=ocr_provider,
            started_at=datetime.now(),
        )

        request = IngestRequest(
            doc_id=doc_id,
            user_id=user_id,
            session_id=session_id,
            filename=safe_filename,
            file_type=file_type,
            file_size=file_size,
            saved_file_path=original_path,
            artifact_dir=artifact_dir,
            upload_time=upload_time,
            content_mode=content_mode,
            ocr_provider=ocr_provider,
        )
        outcome = self._service.processor.ingest_saved_file(request)
        save_uploaded_document(
            doc_id=doc_id,
            user_id=user_id,
            session_id=session_id,
            filename=safe_filename,
            file_type=file_type,
            file_size=file_size,
            chunk_count=outcome.chunk_count,
            job_id=job_id,
            ingest_mode=content_mode,
            page_count=outcome.page_count,
            text_chars=outcome.text_chars,
            parser_used=outcome.parser_used,
            ocr_provider=outcome.ocr_provider or ocr_provider,
            error_message=outcome.error_message,
            status=outcome.status,
            indexed_at=datetime.now() if outcome.status == "ready" else None,
        )
        save_document_ingest_job(
            job_id=job_id,
            doc_id=doc_id,
            user_id=user_id,
            session_id=session_id,
            filename=safe_filename,
            content_mode=content_mode,
            status=outcome.status,
            ocr_provider=outcome.ocr_provider or ocr_provider,
            parser_used=outcome.parser_used,
            error_message=outcome.error_message,
            completed_at=datetime.now(),
        )
        return outcome.status == "ready", doc_id, outcome.chunk_count

    def search(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        use_hybrid: bool = True,
    ) -> List[Dict]:
        return self._indexer.search(
            query=query,
            user_id=user_id,
            session_id=session_id,
            top_k=top_k,
            use_hybrid=use_hybrid,
        )

    def get_user_documents(self, user_id: str, session_id: Optional[str] = None):
        return list_uploaded_documents(user_id=user_id, session_id=session_id)

    def delete_document(self, doc_id: str, user_id: str) -> bool:
        documents = list_uploaded_documents(user_id=user_id, include_deleted=True)
        target = next((document for document in documents if document.id == doc_id), None)
        if target is None:
            return False

        artifact_dir = self._artifacts.document_dir(user_id, target.session_id or "default", doc_id)
        self._service.processor.delete_document(doc_id=doc_id, user_id=user_id, artifact_dir=artifact_dir)
        delete_document_ingest_jobs(doc_id)
        delete_uploaded_document(doc_id, user_id)
        return True

    def cleanup_session(self, session_id: str) -> int:
        logger.info("[USER_STORE] cleanup_session is not implemented for DB-scoped store")
        return 0


_user_store: Optional[UserDocumentStore] = None


def get_user_store() -> UserDocumentStore:
    global _user_store
    if _user_store is None:
        _user_store = UserDocumentStore()
        _user_store.ensure_collection()
    return _user_store
