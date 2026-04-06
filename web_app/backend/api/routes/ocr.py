from fastapi import APIRouter, File, Form, UploadFile

from ...document_ingestion_service import get_document_ingestion_service

router = APIRouter()


@router.post("/api/ocr/preview")
async def ocr_preview(file: UploadFile = File(...)):
    service = get_document_ingestion_service()
    return await service.preview_file(upload=file)


@router.post("/api/ocr/extract")
async def ocr_extract(
    file: UploadFile = File(...),
    mode: str = Form("Fast"),
    prompt_json: str = Form("{}"),
):
    service = get_document_ingestion_service()
    return await service.extract_debug(upload=file, mode=mode, prompt_json=prompt_json)
