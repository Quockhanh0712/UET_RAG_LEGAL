from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class RemoteLLMClient:
    def __init__(
        self,
        provider_type: str,
        base_url: str,
        model: str,
        timeout_s: float = 60.0,
        system_prompt: str = "",
    ):
        self.provider_type = str(provider_type or "").strip()
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.model = str(model or "").strip()
        self.timeout_s = float(timeout_s)
        self.system_prompt = str(system_prompt or "").strip()

    def generate(self, query: str, context: str = "", system_prompt: Optional[str] = None) -> str:
        prompt = self._build_answer_prompt(query=query, context=context)
        return self.generate_prompt(prompt=prompt, system_prompt=system_prompt)

    def generate_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if self.provider_type == "tgi_completions":
            return self._generate_tgi(prompt=prompt, system_prompt=system_prompt)
        if self.provider_type == "ollama_generate":
            return self._generate_ollama(prompt=prompt, system_prompt=system_prompt)
        raise ValueError(f"Unsupported provider_type: {self.provider_type}")

    def _generate_tgi(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        merged_prompt = prompt
        final_system = str(system_prompt or self.system_prompt or "").strip()
        if final_system:
            merged_prompt = f"{final_system}\n\n{prompt}"
        response = requests.post(
            self.base_url,
            json={
                "model": self.model or "tgi",
                "prompt": merged_prompt,
                "max_tokens": 2048,
                "temperature": 0.1,
                "repetition_penalty": 1.1,
                "stop": ["</queries>", "\n\n\n"],
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("text", "")).strip()

    def _generate_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        endpoint = self.base_url
        if not endpoint.endswith("/api/generate"):
            endpoint = f"{endpoint}/api/generate"
        response = requests.post(
            endpoint,
            json={
                "model": self.model,
                "prompt": prompt,
                "system": str(system_prompt or self.system_prompt or "").strip(),
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 2048,
                    "num_ctx": 32768,
                },
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()

    @staticmethod
    def _build_answer_prompt(query: str, context: str = "") -> str:
        context_text = str(context or "").strip()
        if not context_text:
            return f"CAU HOI: {query}\n\nTRA LOI:"
        return f"CONTEXT:\n{context_text}\n\nCAU HOI: {query}\n\nTRA LOI:"
