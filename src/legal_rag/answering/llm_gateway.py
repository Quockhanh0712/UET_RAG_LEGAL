"""LLM gateway adapter for Legal RAG answers."""

from __future__ import annotations

import logging
import os
import re

from src.legal_rag.answering.prompt_builder import build_answer_prompt

logger = logging.getLogger(__name__)


class LegalLLM:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "ollama")
        self.model = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.system_prompt = (
            "Ban la tro ly phap ly Viet Nam. "
            "Chi duoc tra loi dua tren EVIDENCE_CONTEXT. "
            "QUERY_CONTEXT chi de hieu y dinh, khong phai can cu phap ly. "
            "Khong duoc them ghi chu, luu y, hoac lap lai doan van. "
            "Neu khong du thong tin, phai noi ro trong muc CHUA_DU_CAN_CU."
        )

    def generate(self, query: str, context) -> str:
        prompt = build_answer_prompt(query, context)
        if self.provider == "ollama":
            return self._generate_ollama(prompt)
        return self._generate_fallback(prompt)

    def _generate_ollama(self, prompt: str) -> str:
        import requests

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": self.system_prompt,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.85,
                        "num_predict": 640,
                        "num_ctx": 16384,
                        "repeat_penalty": 1.35,
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            if "<think>" in text:
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text
        except Exception as exc:
            logger.error("[LLM] Error: %s", exc)
            return f"Loi khi goi LLM: {exc}"

    @staticmethod
    def _generate_fallback(prompt: str) -> str:
        return "LLM khong kha dung. Vui long kiem tra cau hinh."

