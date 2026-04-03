from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root():
    from ... import main as legacy

    return await legacy.root()


@router.get("/api/status")
async def get_status():
    from ... import main as legacy

    return await legacy.get_status()
