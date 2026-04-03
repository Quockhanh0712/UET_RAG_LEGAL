from fastapi import APIRouter

from ..schemas import SearchRequest, SearchResponse

router = APIRouter()


@router.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    from ... import main as legacy

    return await legacy.search(request)
