from typing import Optional

from fastapi import APIRouter

from ..schemas import ScoreRequest, ScoreResponse

router = APIRouter()


@router.post("/api/score", response_model=ScoreResponse)
async def score_answer(request: ScoreRequest):
    from ... import main as legacy

    return await legacy.score_answer(request)


@router.post("/api/preload-reranker")
async def preload_reranker():
    from ... import main as legacy

    return await legacy.preload_reranker()


@router.get("/api/metrics/stats")
async def get_metrics_statistics(user_id: Optional[str] = None, days: int = 7):
    from ... import main as legacy

    return await legacy.get_metrics_statistics(user_id=user_id, days=days)


@router.get("/api/metrics/history")
async def get_metrics_history(session_id: str):
    from ... import main as legacy

    return await legacy.get_metrics_history(session_id=session_id)


@router.get("/api/metrics/all")
async def get_all_metrics(
    user_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    grade_filter: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
):
    from ... import main as legacy

    return await legacy.get_all_metrics(
        user_id=user_id,
        limit=limit,
        offset=offset,
        grade_filter=grade_filter,
        sort_by=sort_by,
        sort_order=sort_order,
    )


@router.get("/api/metrics/export")
async def export_all_metrics(format: str = "json", user_id: Optional[str] = None):
    from ... import main as legacy

    return await legacy.export_all_metrics(format=format, user_id=user_id)
