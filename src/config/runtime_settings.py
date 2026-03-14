from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return int(default)
    return int(str(value).strip())


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return float(default)
    return float(str(value).strip())


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return str(default)
    return str(value).strip()


CHAINLIT_LOCKED_DEFAULTS: Dict[str, str] = {
    "QDRANT_USE_CLOUD": "true",
    "QDRANT_COLLECTION": "legal_documents",
    "QDRANT_COLLECTION_BY_MODEL": "true",
    "EMBEDDING_MODEL": "truro7/vn-law-embedding",
    "EMBEDDING_NORMALIZE": "true",
    "EMBEDDING_DEVICE": "cuda",
    "USE_FP16": "true",
    "LLM_PROVIDER": "ollama",
    "RERANKER_MODEL": "AITeamVN/Vietnamese_Reranker",
    "RERANKER_ALPHA": "1.0",
    "RERANKER_MAX_LENGTH": "2304",
    "RERANKER_BATCH_SIZE": "8",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


CHAINLIT_ENV_OVERRIDES: Dict[str, str] = {
    "OLLAMA_MODEL": "qwen3:8b",
    "OLLAMA_URL": "http://222.255.250.24:11434",
    "QUERY_REWRITER_ENABLED": "false",
    "QUERY_REWRITER_URL": "http://222.255.250.24:8080/v1/completions",
    "QUERY_REWRITER_TIMEOUT_S": "20",
    "QUERY_REWRITER_MAX_SUBQUERIES": "6",
    "QUERY_REWRITER_FORCE": "false",
    "CHAINLIT_TOP_K": "5",
    "CHAINLIT_USE_RERANKER": "true",
    "CHAINLIT_USE_GRAPH": "false",
    "CHAINLIT_REQUIRE_ACTIVE_ONLY": "true",
    "CHAINLIT_BM25_PATH": str(PROJECT_ROOT / "data" / "qdrant_db" / "bm25_encoder.json"),
    "CHAINLIT_BM25_MANIFEST": "",
    "CHAINLIT_COLLECTION_OVERRIDE": "",
    "QDRANT_CONTRACT_STRICT": "false",
}


@dataclass(frozen=True)
class RuntimeSettings:
    profile_name: str
    qdrant_use_cloud: bool
    qdrant_url: str
    qdrant_api_key: str
    collection_base: str
    collection_by_model: bool
    collection_override: str
    embedding_model: str
    normalize_embeddings: bool
    embedding_device: str
    use_fp16: bool
    llm_provider: str
    ollama_model: str
    ollama_url: str
    reranker_model: str
    reranker_alpha: float
    reranker_max_length: int
    reranker_batch_size: int
    query_rewriter_enabled: bool
    query_rewriter_url: str
    query_rewriter_timeout_s: float
    query_rewriter_max_subqueries: int
    query_rewriter_force: bool
    chainlit_top_k: int
    chainlit_use_reranker: bool
    chainlit_use_graph: bool
    chainlit_require_active_only: bool
    bm25_path: Path
    bm25_manifest: Path | None
    qdrant_contract_strict: bool

    def apply_env(self) -> None:
        env_map = {
            "QDRANT_USE_CLOUD": "true" if self.qdrant_use_cloud else "false",
            "QDRANT_CLOUD_URL": self.qdrant_url,
            "QDRANT_API_KEY": self.qdrant_api_key,
            "QDRANT_COLLECTION": self.collection_base,
            "QDRANT_COLLECTION_BY_MODEL": "true" if self.collection_by_model else "false",
            "EMBEDDING_MODEL": self.embedding_model,
            "EMBEDDING_NORMALIZE": "true" if self.normalize_embeddings else "false",
            "EMBEDDING_DEVICE": self.embedding_device,
            "USE_FP16": "true" if self.use_fp16 else "false",
            "LLM_PROVIDER": self.llm_provider,
            "OLLAMA_MODEL": self.ollama_model,
            "OLLAMA_URL": self.ollama_url,
            "RERANKER_MODEL": self.reranker_model,
            "RERANKER_ALPHA": str(self.reranker_alpha),
            "RERANKER_MAX_LENGTH": str(self.reranker_max_length),
            "RERANKER_BATCH_SIZE": str(self.reranker_batch_size),
            "QUERY_REWRITER_ENABLED": "true" if self.query_rewriter_enabled else "false",
            "QUERY_REWRITER_URL": self.query_rewriter_url,
            "QUERY_REWRITER_TIMEOUT_S": str(self.query_rewriter_timeout_s),
            "QUERY_REWRITER_MAX_SUBQUERIES": str(self.query_rewriter_max_subqueries),
            "QUERY_REWRITER_FORCE": "true" if self.query_rewriter_force else "false",
            "CHAINLIT_TOP_K": str(self.chainlit_top_k),
            "CHAINLIT_USE_RERANKER": "true" if self.chainlit_use_reranker else "false",
            "CHAINLIT_USE_GRAPH": "true" if self.chainlit_use_graph else "false",
            "CHAINLIT_REQUIRE_ACTIVE_ONLY": "true" if self.chainlit_require_active_only else "false",
            "CHAINLIT_BM25_PATH": str(self.bm25_path),
            "CHAINLIT_BM25_MANIFEST": str(self.bm25_manifest or ""),
            "CHAINLIT_COLLECTION_OVERRIDE": self.collection_override,
            "QDRANT_CONTRACT_STRICT": "true" if self.qdrant_contract_strict else "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        for key, value in env_map.items():
            os.environ[key] = value


def _select_profile_value(key: str, profile_defaults: Dict[str, str]) -> str:
    if key in CHAINLIT_LOCKED_DEFAULTS:
        return str(profile_defaults[key])
    env_value = os.getenv(key)
    if env_value is not None:
        return str(env_value).strip()
    return str(profile_defaults.get(key, "")).strip()


def load_runtime_settings(profile_name: str = "chainlit_demo") -> RuntimeSettings:
    if profile_name not in {"chainlit_demo", "chainlit_smoke"}:
        raise ValueError(f"Unsupported runtime profile: {profile_name}")

    profile_defaults = dict(CHAINLIT_LOCKED_DEFAULTS)
    profile_defaults.update(CHAINLIT_ENV_OVERRIDES)

    bm25_manifest_raw = _select_profile_value("CHAINLIT_BM25_MANIFEST", profile_defaults)
    bm25_manifest = Path(bm25_manifest_raw) if bm25_manifest_raw else None

    settings = RuntimeSettings(
        profile_name=profile_name,
        qdrant_use_cloud=_select_profile_value("QDRANT_USE_CLOUD", profile_defaults).lower() == "true",
        qdrant_url=_env_str("QDRANT_CLOUD_URL", ""),
        qdrant_api_key=_env_str("QDRANT_API_KEY", ""),
        collection_base=_select_profile_value("QDRANT_COLLECTION", profile_defaults),
        collection_by_model=_select_profile_value("QDRANT_COLLECTION_BY_MODEL", profile_defaults).lower() == "true",
        collection_override=_select_profile_value("CHAINLIT_COLLECTION_OVERRIDE", profile_defaults),
        embedding_model=_select_profile_value("EMBEDDING_MODEL", profile_defaults),
        normalize_embeddings=_select_profile_value("EMBEDDING_NORMALIZE", profile_defaults).lower() == "true",
        embedding_device=_select_profile_value("EMBEDDING_DEVICE", profile_defaults),
        use_fp16=_select_profile_value("USE_FP16", profile_defaults).lower() == "true",
        llm_provider=_select_profile_value("LLM_PROVIDER", profile_defaults),
        ollama_model=_select_profile_value("OLLAMA_MODEL", profile_defaults),
        ollama_url=_select_profile_value("OLLAMA_URL", profile_defaults),
        reranker_model=_select_profile_value("RERANKER_MODEL", profile_defaults),
        reranker_alpha=_env_float("RERANKER_ALPHA", float(profile_defaults["RERANKER_ALPHA"])),
        reranker_max_length=_env_int("RERANKER_MAX_LENGTH", int(profile_defaults["RERANKER_MAX_LENGTH"])),
        reranker_batch_size=_env_int("RERANKER_BATCH_SIZE", int(profile_defaults["RERANKER_BATCH_SIZE"])),
        query_rewriter_enabled=_env_bool("QUERY_REWRITER_ENABLED", False),
        query_rewriter_url=_select_profile_value("QUERY_REWRITER_URL", profile_defaults),
        query_rewriter_timeout_s=_env_float(
            "QUERY_REWRITER_TIMEOUT_S", float(profile_defaults["QUERY_REWRITER_TIMEOUT_S"])
        ),
        query_rewriter_max_subqueries=_env_int(
            "QUERY_REWRITER_MAX_SUBQUERIES", int(profile_defaults["QUERY_REWRITER_MAX_SUBQUERIES"])
        ),
        query_rewriter_force=_env_bool("QUERY_REWRITER_FORCE", False),
        chainlit_top_k=_env_int("CHAINLIT_TOP_K", int(profile_defaults["CHAINLIT_TOP_K"])),
        chainlit_use_reranker=_env_bool("CHAINLIT_USE_RERANKER", True),
        chainlit_use_graph=_env_bool("CHAINLIT_USE_GRAPH", False),
        chainlit_require_active_only=_env_bool("CHAINLIT_REQUIRE_ACTIVE_ONLY", True),
        bm25_path=Path(_select_profile_value("CHAINLIT_BM25_PATH", profile_defaults)),
        bm25_manifest=bm25_manifest,
        qdrant_contract_strict=_env_bool("QDRANT_CONTRACT_STRICT", False),
    )
    return settings

