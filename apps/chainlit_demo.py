"""Chainlit demo app for hybrid legal retrieval on Qdrant Cloud."""

from __future__ import annotations

import asyncio
import sys
from functools import lru_cache
from pathlib import Path

import chainlit as cl
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.adapters.chainlit_adapter import execute_chat_query, format_chainlit_message
from src.config.runtime_settings import load_runtime_settings
from src.factories.legal_pipeline_factory import get_legal_pipeline
from src.runtime.qdrant_contract import validate_qdrant_contract

SETTINGS = load_runtime_settings("chainlit_demo")
SETTINGS.apply_env()


@lru_cache(maxsize=1)
def _get_runtime_context():
    pipeline = get_legal_pipeline(SETTINGS)
    contract = validate_qdrant_contract(SETTINGS, int(pipeline.embedding_model.embedding_dim))
    return pipeline, contract


@cl.on_chat_start
async def on_chat_start() -> None:
    try:
        _, contract = await asyncio.to_thread(_get_runtime_context)
        await cl.Message(
            content=(
                "Hybrid Legal Demo is ready.\n\n"
                f"- model: `{SETTINGS.embedding_model}`\n"
                f"- qdrant_url: `{SETTINGS.qdrant_url}`\n"
                f"- collection: `{contract.collection_name}`\n"
                f"- collection_status: `{contract.collection_status}`\n"
                f"- vector_dim: `{contract.vector_dim}`\n"
                f"- top_k: `{SETTINGS.chainlit_top_k}`\n"
                f"- use_reranker: `{SETTINGS.chainlit_use_reranker}`\n"
                f"- reranker_model: `{SETTINGS.reranker_model}`\n"
                f"- reranker_alpha: `{SETTINGS.reranker_alpha}`\n"
                f"- query_rewriter_enabled: `{SETTINGS.query_rewriter_enabled}`\n"
                f"- query_rewriter_force: `{SETTINGS.query_rewriter_force}`\n"
                f"- use_graph: `{SETTINGS.chainlit_use_graph}`\n"
                f"- require_active_only: `{SETTINGS.chainlit_require_active_only}`"
            )
        ).send()
    except Exception as exc:
        await cl.Message(content=f"Startup error: {exc}").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = str(message.content or "").strip()
    if not query:
        await cl.Message(content="Empty query.").send()
        return

    try:
        pipeline, _ = await asyncio.to_thread(_get_runtime_context)
    except Exception as exc:
        await cl.Message(content=f"Configuration error: {exc}").send()
        return

    result = await asyncio.to_thread(
        execute_chat_query,
        pipeline,
        query,
        SETTINGS.chainlit_top_k,
        SETTINGS.chainlit_use_reranker,
        SETTINGS.chainlit_require_active_only,
        SETTINGS.chainlit_use_graph,
    )
    await cl.Message(content=format_chainlit_message(result, top_k=SETTINGS.chainlit_top_k)).send()
