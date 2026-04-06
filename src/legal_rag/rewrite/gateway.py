"""LLM gateway helpers for rewrite extraction."""

from __future__ import annotations

from typing import Dict

from src.services.remote_llm_client import RemoteLLMClient


def call_rewrite_gateway(
    *,
    prompt: str,
    provider_type: str,
    base_url: str,
    model_name: str,
    timeout_s: float,
    extraction_schema: Dict[str, object],
) -> str:
    client = RemoteLLMClient(
        provider_type=provider_type,
        base_url=base_url,
        model=model_name,
        timeout_s=timeout_s,
    )
    return client.generate_prompt(
        prompt=prompt,
        task="rewrite",
        format_schema=extraction_schema,
    )

