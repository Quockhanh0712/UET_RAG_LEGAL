from __future__ import annotations

import os


DEFAULT_LEGAL_EMBEDDING_MODEL = "truro7/vn-law-embedding"
DEFAULT_EMBEDDING_NORMALIZE = True


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_embedding_model_name(explicit: str | None = None) -> str:
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()

    value = os.getenv("EMBEDDING_MODEL")
    if value is None or not str(value).strip():
        return DEFAULT_LEGAL_EMBEDDING_MODEL
    return str(value).strip()


def resolve_normalize_embeddings(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return _env_bool("EMBEDDING_NORMALIZE", DEFAULT_EMBEDDING_NORMALIZE)
