from __future__ import annotations

from threading import Lock

from src.config.runtime_settings import RuntimeSettings
from src.runtime.qdrant_contract import resolve_collection_name
from src.services.query_rewrite_service import QueryRewriteService

_PIPELINE_CACHE = {}
_PIPELINE_LOCK = Lock()


def _pipeline_cache_key(settings: RuntimeSettings) -> str:
    return "|".join(
        [
            settings.profile_name,
            settings.qdrant_url,
            settings.collection_override or settings.collection_base,
            settings.embedding_model,
            str(settings.normalize_embeddings).lower(),
            settings.reranker_model,
            str(settings.query_rewriter_enabled).lower(),
            str(settings.query_rewriter_force).lower(),
        ]
    )


def get_legal_pipeline(settings: RuntimeSettings):
    from src.legal_rag_pipeline import LegalRAGPipeline

    key = _pipeline_cache_key(settings)
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        return cached

    with _PIPELINE_LOCK:
        cached = _PIPELINE_CACHE.get(key)
        if cached is not None:
            return cached

        settings.apply_env()
        rewrite_service = QueryRewriteService.from_settings(settings)
        pipeline = LegalRAGPipeline(
            embedding_model_name=settings.embedding_model,
            normalize_embeddings=settings.normalize_embeddings,
            query_rewrite_service=rewrite_service,
        )
        pipeline.initialize()
        collection_name = resolve_collection_name(settings, int(pipeline.embedding_model.embedding_dim))
        pipeline.configure_runtime(
            collection_name=collection_name,
            bm25_path=settings.bm25_path,
        )
        pipeline.rerank_alpha = settings.reranker_alpha
        _PIPELINE_CACHE[key] = pipeline
        return pipeline

