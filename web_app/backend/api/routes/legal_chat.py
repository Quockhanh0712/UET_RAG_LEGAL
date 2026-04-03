from fastapi import APIRouter

from ..schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    from ... import main as legacy

    return await legacy.chat(request)
