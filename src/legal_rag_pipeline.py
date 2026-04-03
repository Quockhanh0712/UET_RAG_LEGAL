"""Facade export for Legal RAG pipeline public contracts."""

from __future__ import annotations

import logging
from typing import Optional

from src.legal_rag.contracts.results import LegalAnswer, LegalSearchResult
from src.legal_rag.pipeline.orchestrator import LegalRAGPipeline

_pipeline: Optional[LegalRAGPipeline] = None
_pipeline_lock = None


def _get_pipeline_lock():
    global _pipeline_lock
    if _pipeline_lock is None:
        import threading

        _pipeline_lock = threading.Lock()
    return _pipeline_lock


def get_legal_rag_pipeline() -> LegalRAGPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    lock = _get_pipeline_lock()
    with lock:
        if _pipeline is None:
            _pipeline = LegalRAGPipeline()
    return _pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = LegalRAGPipeline()
    pipeline.initialize()
    print("Stats:", pipeline.get_stats())

