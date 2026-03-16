from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    provider_type: str
    base_url: str
    model: str
    timeout_s: float = 60.0
    supports_rewrite: bool = True
    supports_answer: bool = True
    enabled: bool = True

    def to_public_dict(self) -> Dict[str, object]:
        return asdict(self)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip() or default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return float(default)
    return float(str(value).strip())


def _build_profiles() -> Dict[str, ModelProfile]:
    profiles = {
        'qwen3_5_9b_ollama_remote': ModelProfile(
            id='qwen3_5_9b_ollama_remote',
            label='Qwen3.5 9B',
            provider_type='ollama_generate',
            base_url=_env_str('OLLAMA_QWEN35_URL', _env_str('OLLAMA_URL', 'http://222.255.250.24:11434')),
            model=_env_str('OLLAMA_QWEN35_MODEL', 'qwen3.5:9b'),
            timeout_s=_env_float('OLLAMA_QWEN35_TIMEOUT_S', 120.0),
            supports_rewrite=True,
            supports_answer=True,
            enabled=_env_str('OLLAMA_QWEN35_ENABLED', 'true').lower() in {'1', 'true', 'yes', 'on'},
        ),
        'legal_tgi_remote': ModelProfile(
            id='legal_tgi_remote',
            label='Model Luat',
            provider_type='tgi_completions',
            base_url=_env_str('LEGAL_TGI_URL', 'http://222.255.250.24:8080/v1/completions'),
            model=_env_str('LEGAL_TGI_MODEL', 'tgi'),
            timeout_s=_env_float('LEGAL_TGI_TIMEOUT_S', 45.0),
            supports_rewrite=False,
            supports_answer=True,
            enabled=_env_str('LEGAL_TGI_ENABLED', 'false').lower() in {'1', 'true', 'yes', 'on'},
        ),
        'qwen3_8b_ollama_remote': ModelProfile(
            id='qwen3_8b_ollama_remote',
            label='Qwen3 8B',
            provider_type='ollama_generate',
            base_url=_env_str('OLLAMA_URL', 'http://222.255.250.24:11434'),
            model=_env_str('OLLAMA_MODEL', 'qwen3:8b'),
            timeout_s=_env_float('OLLAMA_TIMEOUT_S', 120.0),
            supports_rewrite=True,
            supports_answer=True,
            enabled=_env_str('OLLAMA_QWEN3_ENABLED', 'false').lower() in {'1', 'true', 'yes', 'on'},
        ),
    }
    return profiles


def list_model_profiles(enabled_only: bool = True) -> List[ModelProfile]:
    profiles = list(_build_profiles().values())
    if enabled_only:
        profiles = [profile for profile in profiles if profile.enabled]
    return profiles


def get_default_model_profile_id() -> str:
    return 'qwen3_5_9b_ollama_remote'


def get_model_profile(profile_id: str | None = None) -> ModelProfile:
    profiles = _build_profiles()
    selected_id = str(profile_id or get_default_model_profile_id()).strip()
    profile = profiles.get(selected_id)
    if profile is None or not profile.enabled:
        return profiles[get_default_model_profile_id()]
    return profile
