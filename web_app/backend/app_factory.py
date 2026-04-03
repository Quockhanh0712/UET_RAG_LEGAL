from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from .api.routes.documents import router as documents_router
    from .api.routes.health import router as health_router
    from .api.routes.legal_chat import router as legal_chat_router
    from .api.routes.legal_search import router as legal_search_router
    from .api.routes.metrics import router as metrics_router
    from .api.routes.ocr import router as ocr_router
    from .api.routes.settings import router as settings_router
except ImportError:
    from api.routes.documents import router as documents_router
    from api.routes.health import router as health_router
    from api.routes.legal_chat import router as legal_chat_router
    from api.routes.legal_search import router as legal_search_router
    from api.routes.metrics import router as metrics_router
    from api.routes.ocr import router as ocr_router
    from api.routes.settings import router as settings_router


def create_app() -> FastAPI:
    try:
        from . import main as legacy
    except ImportError:
        import main as legacy

    app = FastAPI(
        title="Legal RAG API",
        description="Vietnamese Legal RAG Chatbot API with Dual-Store Architecture",
        version="1.0.0",
        lifespan=legacy.lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(settings_router)
    app.include_router(legal_chat_router)
    app.include_router(legal_search_router)
    app.include_router(documents_router)
    app.include_router(metrics_router)
    app.include_router(ocr_router)

    return app
