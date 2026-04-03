from fastapi import APIRouter, HTTPException

from ..schemas import RerankerSettingsResponse

router = APIRouter()


@router.get("/api/settings/model-profiles")
async def get_model_profiles():
    from ... import main as legacy

    return await legacy.get_model_profiles()


@router.get("/api/settings/reranker", response_model=RerankerSettingsResponse)
async def get_reranker_settings():
    from ... import main as legacy

    return await legacy.get_reranker_settings()


@router.put("/api/settings/reranker")
async def update_reranker_settings(enabled: bool = None, alpha: float = None):
    from ... import main as legacy

    return await legacy.update_reranker_settings(enabled=enabled, alpha=alpha)
