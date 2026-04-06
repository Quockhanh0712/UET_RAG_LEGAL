from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from ...document_ingestion_service import get_document_ingestion_service
from ..schemas import DocumentStatusResponse, UploadResponse

router = APIRouter()


@router.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(default="default_user"),
    session_id: Optional[str] = Form(default=None),
    content_mode: str = Form(default="text"),
    ocr_provider: str = Form(default="default"),
):
    service = get_document_ingestion_service()
    return await service.submit_upload(
        upload=file,
        user_id=user_id,
        session_id=session_id,
        content_mode=content_mode,
        ocr_provider=ocr_provider,
    )


@router.get("/api/documents")
async def get_documents(user_id: str = "default_user", session_id: Optional[str] = None):
    service = get_document_ingestion_service()
    return service.list_documents(user_id=user_id, session_id=session_id)


@router.get("/api/documents/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(doc_id: str, user_id: str = "default_user"):
    service = get_document_ingestion_service()
    return service.get_document_status(doc_id=doc_id, user_id=user_id)


@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, user_id: str = "default_user"):
    service = get_document_ingestion_service()
    return await service.delete_document(doc_id=doc_id, user_id=user_id)


@router.get("/api/history")
async def get_history(session_id: str):
    from ... import main as legacy

    return await legacy.get_history(session_id=session_id)


@router.get("/api/sessions")
async def get_sessions(user_id: str = "default_user"):
    from ... import main as legacy

    return await legacy.get_sessions(user_id=user_id)
