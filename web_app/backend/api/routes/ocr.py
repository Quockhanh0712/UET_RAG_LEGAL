from fastapi import APIRouter, File, Form, UploadFile

router = APIRouter()


@router.post("/api/ocr/preview")
async def ocr_preview(file: UploadFile = File(...)):
    from ... import main as legacy

    return await legacy.ocr_preview(file=file)


@router.post("/api/ocr/extract")
async def ocr_extract(
    file: UploadFile = File(...),
    mode: str = Form("Fast"),
    prompt_json: str = Form("{}"),
):
    from ... import main as legacy

    return await legacy.ocr_extract(file=file, mode=mode, prompt_json=prompt_json)
