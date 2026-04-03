from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from ..schemas import UploadResponse

router = APIRouter()


@router.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(default="default_user"),
    session_id: Optional[str] = Form(default=None),
):
    from ... import main as legacy

    return await legacy.upload_file(file=file, user_id=user_id, session_id=session_id)


@router.get("/api/documents")
async def get_documents(user_id: str = "default_user", session_id: Optional[str] = None):
    from ... import main as legacy

    return await legacy.get_documents(user_id=user_id, session_id=session_id)


@router.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, user_id: str = "default_user"):
    from ... import main as legacy

    return await legacy.delete_document(doc_id=doc_id, user_id=user_id)


@router.get("/api/history")
async def get_history(session_id: str):
    from ... import main as legacy

    return await legacy.get_history(session_id=session_id)


@router.get("/api/sessions")
async def get_sessions(user_id: str = "default_user"):
    from ... import main as legacy

    return await legacy.get_sessions(user_id=user_id)
