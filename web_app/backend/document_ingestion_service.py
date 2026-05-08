from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import HTTPException, UploadFile

try:
    from .config import settings
    from .database import (
        delete_document_ingest_jobs,
        delete_uploaded_document,
        get_document_ingest_job_for_doc,
        get_uploaded_document,
        init_db,
        list_uploaded_documents,
        save_document_ingest_job,
        save_uploaded_document,
    )
except ImportError:
    from config import settings
    from database import (
        delete_document_ingest_jobs,
        delete_uploaded_document,
        get_document_ingest_job_for_doc,
        get_uploaded_document,
        init_db,
        list_uploaded_documents,
        save_document_ingest_job,
        save_uploaded_document,
    )
from src.document_ingestion.extractors import UnsupportedDocumentError
from src.document_ingestion.models import IngestRequest
from src.document_ingestion.storage import ArtifactStore

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class DocumentIngestionService:
    def __init__(self, processor=None, artifact_store: Optional[ArtifactStore] = None):
        self.artifact_store = artifact_store or ArtifactStore(settings.UPLOAD_DIR)
        self._processor = processor
        self._tasks: Dict[str, asyncio.Task] = {}
        self._initialized = False

    @property
    def processor(self):
        if self._processor is None:
            from src.document_ingestion.service import DocumentIngestionProcessor

            self._processor = DocumentIngestionProcessor(artifact_store=self.artifact_store)
        return self._processor

    def initialize(self) -> None:
        if self._initialized:
            return
        init_db()
        self._initialized = True

    @staticmethod
    def _normalize_content_mode(content_mode: str) -> str:
        normalized = str(content_mode or "text").strip().lower()
        if normalized not in {"text", "ocr"}:
            raise HTTPException(status_code=400, detail="content_mode phai la 'text' hoac 'ocr'.")
        return normalized

    @staticmethod
    def _safe_filename(filename: str) -> str:
        basename = Path(filename or "upload.bin").name
        return basename or "upload.bin"

    @staticmethod
    def _validate_magic(file_bytes: bytes, filename: str) -> str:
        if not file_bytes:
            raise HTTPException(status_code=400, detail="File rong.")
        if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File vuot gioi han {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB.",
            )

        extension = Path(filename).suffix.lower()
        if extension == ".doc":
            raise HTTPException(
                status_code=400,
                detail="Khong ho tro file .doc trong V1. Vui long chuyen sang .docx.",
            )
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Chi ho tro PDF, DOCX, TXT.")
        if extension == ".pdf" and not file_bytes.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="File PDF khong hop le.")
        if extension == ".docx" and not file_bytes.startswith(b"PK"):
            raise HTTPException(status_code=400, detail="File DOCX khong hop le.")
        if extension == ".txt" and b"\x00" in file_bytes:
            raise HTTPException(status_code=400, detail="File TXT co dau hieu du lieu nhi phan.")
        return extension.lstrip(".")

    async def submit_upload(
        self,
        *,
        upload: UploadFile,
        user_id: str,
        session_id: Optional[str],
        content_mode: str = "text",
        ocr_provider: str = "default",
    ) -> Dict:
        self.initialize()
        normalized_mode = self._normalize_content_mode(content_mode)
        effective_session_id = session_id or str(uuid.uuid4())
        safe_filename = self._safe_filename(upload.filename or "upload.bin")
        file_bytes = await upload.read()
        file_type = self._validate_magic(file_bytes, safe_filename)

        doc_id = f"user_{user_id}_{uuid.uuid4().hex[:12]}"
        job_id = uuid.uuid4().hex
        upload_time = datetime.now().isoformat()
        artifact_dir = self.artifact_store.document_dir(user_id, effective_session_id, doc_id)
        saved_file_path = self.artifact_store.save_original(
            user_id=user_id,
            session_id=effective_session_id,
            doc_id=doc_id,
            filename=safe_filename,
            file_bytes=file_bytes,
        )

        save_uploaded_document(
            doc_id=doc_id,
            user_id=user_id,
            session_id=effective_session_id,
            filename=safe_filename,
            file_type=file_type,
            file_size=len(file_bytes),
            chunk_count=0,
            job_id=job_id,
            ingest_mode=normalized_mode,
            parser_used=None,
            ocr_provider=ocr_provider,
            error_message=None,
            status="queued",
        )
        save_document_ingest_job(
            job_id=job_id,
            doc_id=doc_id,
            user_id=user_id,
            session_id=effective_session_id,
            filename=safe_filename,
            content_mode=normalized_mode,
            status="queued",
            ocr_provider=ocr_provider,
            parser_used=None,
            error_message=None,
        )

        request = IngestRequest(
            doc_id=doc_id,
            user_id=user_id,
            session_id=effective_session_id,
            filename=safe_filename,
            file_type=file_type,
            file_size=len(file_bytes),
            saved_file_path=saved_file_path,
            artifact_dir=artifact_dir,
            upload_time=upload_time,
            content_mode=normalized_mode,
            ocr_provider=ocr_provider or "default",
        )

        task = asyncio.create_task(self._run_job(job_id=job_id, request=request))
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": safe_filename,
            "chunk_count": 0,
            "message": "Da tiep nhan tai lieu va dua vao hang doi xu ly.",
            "job_id": job_id,
            "status": "queued",
        }

    async def _run_job(self, *, job_id: str, request: IngestRequest) -> None:
        started_at = datetime.now()
        try:
            save_uploaded_document(
                doc_id=request.doc_id,
                user_id=request.user_id,
                session_id=request.session_id,
                filename=request.filename,
                file_type=request.file_type,
                file_size=request.file_size,
                chunk_count=0,
                job_id=job_id,
                ingest_mode=request.content_mode,
                parser_used=None,
                ocr_provider=request.ocr_provider,
                error_message=None,
                status="running",
            )
            save_document_ingest_job(
                job_id=job_id,
                doc_id=request.doc_id,
                user_id=request.user_id,
                session_id=request.session_id,
                filename=request.filename,
                content_mode=request.content_mode,
                status="running",
                ocr_provider=request.ocr_provider,
                parser_used=None,
                error_message=None,
                started_at=started_at,
            )

            outcome = await asyncio.to_thread(self.processor.ingest_saved_file, request)
            completed_at = datetime.now()

            save_uploaded_document(
                doc_id=request.doc_id,
                user_id=request.user_id,
                session_id=request.session_id,
                filename=request.filename,
                file_type=request.file_type,
                file_size=request.file_size,
                chunk_count=outcome.chunk_count,
                job_id=job_id,
                ingest_mode=request.content_mode,
                page_count=outcome.page_count,
                text_chars=outcome.text_chars,
                parser_used=outcome.parser_used or None,
                ocr_provider=outcome.ocr_provider or request.ocr_provider,
                error_message=outcome.error_message,
                status=outcome.status,
                indexed_at=completed_at if outcome.status == "ready" else None,
            )
            save_document_ingest_job(
                job_id=job_id,
                doc_id=request.doc_id,
                user_id=request.user_id,
                session_id=request.session_id,
                filename=request.filename,
                content_mode=request.content_mode,
                status=outcome.status,
                ocr_provider=outcome.ocr_provider or request.ocr_provider,
                parser_used=outcome.parser_used or None,
                error_message=outcome.error_message,
                started_at=started_at,
                completed_at=completed_at,
            )
        except asyncio.CancelledError:
            logger.info("[INGEST] Job cancelled for doc_id=%s", request.doc_id)
            raise
        except Exception as exc:
            logger.exception("[INGEST] Unhandled job failure for doc_id=%s", request.doc_id)
            completed_at = datetime.now()
            save_uploaded_document(
                doc_id=request.doc_id,
                user_id=request.user_id,
                session_id=request.session_id,
                filename=request.filename,
                file_type=request.file_type,
                file_size=request.file_size,
                chunk_count=0,
                job_id=job_id,
                ingest_mode=request.content_mode,
                parser_used=None,
                ocr_provider=request.ocr_provider,
                error_message=str(exc),
                status="failed",
                indexed_at=None,
            )
            save_document_ingest_job(
                job_id=job_id,
                doc_id=request.doc_id,
                user_id=request.user_id,
                session_id=request.session_id,
                filename=request.filename,
                content_mode=request.content_mode,
                status="failed",
                ocr_provider=request.ocr_provider,
                parser_used=None,
                error_message=str(exc),
                started_at=started_at,
                completed_at=completed_at,
            )

    def list_documents(self, *, user_id: str, session_id: Optional[str] = None) -> Dict:
        documents = list_uploaded_documents(user_id=user_id, session_id=session_id)
        return {"documents": [self._serialize_document(document) for document in documents]}

    def get_document_status(self, *, doc_id: str, user_id: str) -> Dict:
        document = get_uploaded_document(doc_id, user_id=user_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Khong tim thay tai lieu.")
        job = get_document_ingest_job_for_doc(doc_id)
        payload = self._serialize_document(document)
        payload["doc_id"] = doc_id
        payload["job_id"] = document.job_id
        payload["status"] = document.status
        if job is not None:
            payload["job"] = job.to_dict()
        return payload

    async def delete_document(self, *, doc_id: str, user_id: str) -> Dict:
        document = get_uploaded_document(doc_id, user_id=user_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        task = self._tasks.get(document.job_id or "")
        if task and not task.done():
            task.cancel()

        artifact_dir = self.artifact_store.document_dir(document.user_id, document.session_id or "default", doc_id)
        await asyncio.to_thread(
            self.processor.delete_document,
            doc_id=doc_id,
            user_id=user_id,
            artifact_dir=artifact_dir,
        )
        delete_document_ingest_jobs(doc_id)
        delete_uploaded_document(doc_id, user_id)
        return {"message": f"Document {doc_id} deleted"}

    async def preview_file(self, upload: UploadFile) -> Dict:
        file_bytes = await upload.read()
        filename = upload.filename or "preview.bin"
        if filename.lower().endswith(".pdf"):
            import fitz

            document = fitz.open(stream=file_bytes, filetype="pdf")
            if len(document) < 1:
                return {"success": False, "error": "PDF khong co trang nao."}
            page = document.load_page(0)
            pixmap = page.get_pixmap()
            image_bytes = pixmap.tobytes("png")
            document.close()
            return {
                "success": True,
                "image_b64": base64.b64encode(image_bytes).decode("utf-8"),
                "type": "pdf",
            }
        return {
            "success": True,
            "image_b64": base64.b64encode(file_bytes).decode("utf-8"),
            "type": "image",
        }

    async def extract_debug(self, *, upload: UploadFile, mode: str, prompt_json: str) -> Dict:
        file_bytes = await upload.read()
        safe_filename = self._safe_filename(upload.filename or "ocr.bin")
        suffix = Path(safe_filename).suffix.lower()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            file_path = temp_root / safe_filename
            file_path.write_bytes(file_bytes)
            artifact_dir = temp_root / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            try:
                if str(mode or "Fast").strip().lower().startswith("fast"):
                    extracted = await asyncio.to_thread(self.processor.extractor_router.extract, file_path)
                else:
                    extracted = await asyncio.to_thread(
                        self.processor.ocr_router.process,
                        "default",
                        file_path,
                        artifact_dir,
                    )
            except (UnsupportedDocumentError, HTTPException) as exc:
                return {"success": False, "error": str(exc)}
            except Exception as exc:
                return {"success": False, "error": str(exc)}

            template = self._parse_prompt_json(prompt_json)
            pages = extracted.pages or []
            if not pages and extracted.normalized_text:
                pages = [type("_Page", (), {"page_number": 1, "text": extracted.normalized_text})()]

            results = []
            for page in pages:
                results.append({f"Trang {page.page_number}": self._shape_debug_result(page.text, template)})
            return {"success": True, "results": results}

    @staticmethod
    def _parse_prompt_json(prompt_json: str) -> Dict:
        try:
            value = json.loads(prompt_json or "{}")
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _shape_debug_result(text: str, template: Dict) -> Dict:
        if not template:
            return {"raw_text": text}
        result: Dict[str, object] = {}
        for key, value in template.items():
            normalized_key = str(key).lower()
            if isinstance(value, dict):
                result[key] = {}
            elif any(token in normalized_key for token in ["noi_dung", "content", "text", "raw"]):
                result[key] = text
            else:
                result[key] = value if value not in (None, "") else ""
        result["_raw_text"] = text
        return result

    @staticmethod
    def _serialize_document(document) -> Dict:
        payload = document.to_dict()
        payload["doc_id"] = payload.pop("id")
        return payload


_service: Optional[DocumentIngestionService] = None


def get_document_ingestion_service() -> DocumentIngestionService:
    global _service
    if _service is None:
        _service = DocumentIngestionService()
    return _service
