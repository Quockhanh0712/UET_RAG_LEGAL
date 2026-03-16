from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

import requests

from src.legal_data_processing import strip_accents

logger = logging.getLogger(__name__)


class RemoteLLMClient:
    _BANNED_FOOTER_PREFIXES = (
        "ghi chu",
        "ghi chu bo sung",
        "ghi chu cuoi cung",
        "luu y",
        "ket thuc cau tra loi",
        "toi rat mong",
        "tham khao y kien chuyen gia",
    )

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

    def generate(
        self,
        query: str,
        context: Any = "",
        system_prompt: Optional[str] = None,
        task: str = "answer",
    ) -> str:
        prompt = self._build_answer_prompt(query=query, context=context)
        return self.generate_prompt(prompt=prompt, system_prompt=system_prompt, task=task)

    def generate_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task: str = "answer",
        format_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        task_name = str(task or "answer").strip().lower()
        if self.provider_type == "tgi_completions":
            return self._generate_tgi(
                prompt=prompt,
                system_prompt=system_prompt,
                task=task_name,
            )
        if self.provider_type == "ollama_generate":
            return self._generate_ollama(
                prompt=prompt,
                system_prompt=system_prompt,
                task=task_name,
                format_schema=format_schema,
            )
        raise ValueError(f"Unsupported provider_type: {self.provider_type}")

    def _generate_tgi(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task: str = "answer",
    ) -> str:
        merged_prompt = prompt
        final_system = str(system_prompt or self.system_prompt or "").strip()
        if final_system:
            merged_prompt = f"{final_system}\n\n{prompt}"

        max_tokens = 256 if task == "rewrite" else 640
        temperature = 0.0 if task == "rewrite" else 0.1
        repetition_penalty = 1.10 if task == "rewrite" else 1.28
        stop = ["</queries>", "\n\n\n"] if task == "rewrite" else ["\nGhi chú:", "\nLưu ý:", "\nKết thúc câu trả lời"]

        response = requests.post(
            self.base_url,
            json={
                "model": self.model or "tgi",
                "prompt": merged_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "stop": stop,
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        text = str(choices[0].get("text", "")).strip()
        return self._postprocess_answer(text) if task == "answer" else text

    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task: str = "answer",
        format_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        endpoint = self.base_url
        if not endpoint.endswith("/api/generate"):
            endpoint = f"{endpoint}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": str(system_prompt or self.system_prompt or "").strip(),
            "stream": False,
            "think": False,
            "options": self._ollama_options_for_task(task),
        }
        if format_schema:
            payload["format"] = format_schema

        response = requests.post(endpoint, json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        text = str(data.get("response", "")).strip()
        return self._postprocess_answer(text) if task == "answer" else text

    @staticmethod
    def _ollama_options_for_task(task: str) -> Dict[str, Any]:
        if task == "rewrite":
            return {
                "temperature": 0.0,
                "top_p": 0.8,
                "num_predict": 320,
                "num_ctx": 12288,
                "repeat_penalty": 1.15,
            }
        return {
            "temperature": 0.1,
            "top_p": 0.85,
            "num_predict": 640,
            "num_ctx": 16384,
            "repeat_penalty": 1.35,
            "stop": ["\nGhi chú:", "\nLưu ý:", "\nKết thúc câu trả lời"],
        }

    @staticmethod
    def _build_answer_prompt(query: str, context: Any = "") -> str:
        if isinstance(context, dict):
            return RemoteLLMClient._build_structured_answer_prompt(query=query, context=context)
        context_text = str(context or "").strip()
        if not context_text:
            return (
                "TRA LOI DUNG DINH DANG SAU:\n"
                "KET_LUAN:\n"
                "CAN_CU:\n"
                "PHAN_TICH_NGAN:\n\n"
                f"CAU_HOI_GOC: {query}"
            )
        return (
            "CHI SU DUNG THONG TIN TRONG CONTEXT.\n"
            "KHONG THEM GHI CHU, LUU Y HOAC KET THUC CAU TRA LOI.\n"
            "TRA LOI DUNG DINH DANG SAU:\n"
            "KET_LUAN:\n"
            "CAN_CU:\n"
            "PHAN_TICH_NGAN:\n"
            "CHUA_DU_CAN_CU:\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"CAU_HOI_GOC: {query}"
        )

    @staticmethod
    def _build_structured_answer_prompt(query: str, context: Dict[str, Any]) -> str:
        query_context = str(context.get("query_context") or "").strip()
        evidence_context = str(context.get("evidence_context") or "").strip()

        return (
            "BAN LA TRO LY PHAP LY VIET NAM.\n"
            "CHI DUOC SU DUNG THONG TIN TRONG EVIDENCE_CONTEXT.\n"
            "QUERY_CONTEXT CHI DE HIEU Y DINH, KHONG PHAI CAN CU PHAP LY.\n"
            "NAMED_LAW_HINTS TRONG QUERY_CONTEXT CHI LA GOI Y CUA NGUOI DUNG, KHONG PHAI CAN CU PHAP LY.\n"
            "NEU QUERY_CONTEXT CO QUESTION_UNITS, HAY TRA LOI LAN LUOT TUNG Y HOI VA KHONG BO SOT Y NAO CO EVIDENCE.\n"
            "KHONG DUOC THEM GHI CHU, LUU Y, KHUYEN CAO, HOAC KET THUC CAU TRA LOI.\n"
            "KHONG LAP LAI DOAN VAN.\n"
            "NEU CHUA DU CAN CU, GHI RO TRONG MUC CHUA_DU_CAN_CU.\n"
            "TRA LOI DUNG DINH DANG SAU:\n"
            "KET_LUAN:\n"
            "CAN_CU:\n"
            "PHAN_TICH_NGAN:\n"
            "CHUA_DU_CAN_CU:\n\n"
            f"QUERY_CONTEXT:\n{query_context or '<none>'}\n\n"
            f"EVIDENCE_CONTEXT:\n{evidence_context or '<none>'}\n\n"
            f"CAU_HOI_GOC: {query}"
        )

    def _postprocess_answer(self, text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""

        value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL | re.IGNORECASE).strip()
        value = value.replace("\r\n", "\n")

        raw_paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", value) if paragraph.strip()]
        cleaned_paragraphs = []
        seen = set()

        for paragraph in raw_paragraphs:
            paragraph = self._dedupe_sentences(paragraph)
            normalized = strip_accents(paragraph).lower().strip(" *:_-")
            if any(normalized.startswith(prefix) for prefix in self._BANNED_FOOTER_PREFIXES):
                continue
            compact = re.sub(r"\s+", " ", normalized)
            if compact in seen:
                continue
            seen.add(compact)
            cleaned_paragraphs.append(paragraph)

        if not cleaned_paragraphs:
            cleaned_paragraphs = raw_paragraphs[:3]

        result = "\n\n".join(cleaned_paragraphs).strip()
        result = re.sub(r"\n{3,}", "\n\n", result)

        for marker in (
            "Ghi chú:",
            "Ghi chu:",
            "Lưu ý:",
            "Luu y:",
            "Kết thúc câu trả lời",
            "Ket thuc cau tra loi",
            "KET THUC CAU TRA LOI",
        ):
            index = result.find(marker)
            if index > 0:
                result = result[:index].rstrip()

        return result.strip()

    @staticmethod
    def _dedupe_sentences(paragraph: str) -> str:
        text = str(paragraph or "").strip()
        if not text:
            return ""
        if "\n-" in text or "\n*" in text:
            return text

        sentences = [
            item.strip()
            for item in re.split(r"(?<=[.!?])\s+", text)
            if item and item.strip()
        ]
        filtered = []
        seen = set()
        for sentence in sentences:
            normalized = re.sub(r"\s+", " ", strip_accents(sentence).lower()).strip(" *:_-")
            if not normalized:
                continue
            if filtered and normalized == filtered[-1][0]:
                continue
            if normalized in seen and len(normalized) > 40:
                continue
            seen.add(normalized)
            filtered.append((normalized, sentence.strip()))

        if not filtered:
            return text
        return " ".join(sentence for _, sentence in filtered).strip()

    @staticmethod
    def safe_json_loads(text: str) -> Dict[str, Any]:
        value = str(text or "").strip()
        if not value:
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", value, flags=re.DOTALL)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
