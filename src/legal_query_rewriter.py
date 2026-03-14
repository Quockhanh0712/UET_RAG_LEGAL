"""
Legal query rewriting with defensive parsing and fallback behavior.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import List

from src.legal_data_processing import strip_accents
from src.services.remote_llm_client import RemoteLLMClient

logger = logging.getLogger(__name__)


@dataclass
class QueryRewriteResult:
    original_query: str
    queries: List[str]
    status: str
    raw_response: str = ""
    missing_numbers: List[str] = field(default_factory=list)
    fallback_reason: str = ""
    latency_ms: float = 0.0

    @property
    def rewrite_used(self) -> bool:
        return self.status == "ok" and any(q != self.original_query for q in self.queries)


class LegalQueryRewriter:
    def __init__(
        self,
        tgi_url: str | None = None,
        timeout_s: float | None = None,
        max_subqueries: int | None = None,
        provider_type: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ):
        self.tgi_url = tgi_url or os.getenv("QUERY_REWRITER_URL", "http://222.255.250.24:8080/v1/completions")
        self.timeout_s = float(timeout_s or os.getenv("QUERY_REWRITER_TIMEOUT_S", "20"))
        self.max_subqueries = int(max_subqueries or os.getenv("QUERY_REWRITER_MAX_SUBQUERIES", "6"))
        self.temperature = 0.01
        self.max_tokens = 256
        self.repetition_penalty = 1.15
        self.provider_type = str(provider_type or os.getenv("QUERY_REWRITER_PROVIDER", "tgi_completions")).strip()
        self.base_url = str(base_url or self.tgi_url).strip()
        self.model_name = str(model_name or os.getenv("QUERY_REWRITER_MODEL", "tgi")).strip()

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        return sorted(set(re.findall(r"\d+", str(text or ""))))

    @staticmethod
    def _should_rewrite(query: str) -> bool:
        text = strip_accents(str(query or "")).lower()
        if len(text) >= 80:
            return True
        indicators = [
            "trach nhiem",
            "hinh su",
            "dan su",
            "hanh chinh",
            "thuoc ve ai",
            "ca hai",
            "va ",
            " hay ",
            "/",
            ":",
        ]
        return any(token in text for token in indicators)

    def _call_llm(self, prompt: str) -> str:
        client = RemoteLLMClient(
            provider_type=self.provider_type,
            base_url=self.base_url,
            model=self.model_name,
            timeout_s=self.timeout_s,
        )
        text = client.generate_prompt(prompt=prompt)
        if "<queries>" in text and "</queries>" not in text:
            text = f"{text}\n</queries>"
        return text

    def _build_prompt(self, user_query: str) -> str:
        return f"""Ban la chuyen gia phan tich ho so phap ly.
Nhiem vu: Phan ra mot tinh huong phap ly phuc tap thanh cac cau hoi tim kiem luat cu the, tach biet ro tung chu the va tung loai trach nhiem, nhung van giu nguyen hanh vi goc.

Thuc hien tu duy trong the <analysis> theo 3 y:
- Hanh vi/Su kien chinh
- Cac chu the lien quan
- Van de phap ly

Sau do, sinh cac cau truy van nam trong the <query> va </query>. Moi cau chi tap trung vao 1 chu the va 1 loai trach nhiem gan voi hanh vi chinh.

<example>
Cau hoi: Tai xe A lai xe giao hang cua cong ty B, do say ruou nen dam chet nguoi. Trach nhiem boi thuong va di tu thuoc ve ai?
<analysis>
- Hanh vi chinh: Lai xe say ruou gay tai nan chet nguoi.
- Chu the: Tai xe, Cong ty.
- Van de phap ly: Trach nhiem hinh su, trach nhiem boi thuong dan su.
</analysis>
<queries>
<query>Trach nhiem hinh su cua tai xe lai xe say ruou gay tai nan chet nguoi?</query>
<query>Trach nhiem boi thuong dan su cua cong ty khi nhan vien lai xe gay tai nan trong luc lam viec?</query>
<query>Trach nhiem boi thuong dan su cua tai xe gay tai nan chet nguoi?</query>
</queries>
</example>

Cau hoi: {user_query}
<analysis>
"""

    def rewrite_query(self, user_query: str, force: bool = False) -> QueryRewriteResult:
        original_query = str(user_query or "").strip()
        if not original_query:
            return QueryRewriteResult(
                original_query="",
                queries=[],
                status="empty_input",
                fallback_reason="empty_input",
            )

        if not force and not self._should_rewrite(original_query):
            return QueryRewriteResult(
                original_query=original_query,
                queries=[original_query],
                status="skipped",
                fallback_reason="not_complex_enough",
            )

        prompt = self._build_prompt(original_query)
        started = time.perf_counter()
        raw_output = ""
        try:
            raw_output = self._call_llm(prompt)
        except Exception as exc:
            logger.warning("[REWRITER] API call failed: %s", exc)
            return QueryRewriteResult(
                original_query=original_query,
                queries=[original_query],
                status="api_error",
                fallback_reason=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        extracted = re.findall(r"<query>(.*?)</query>", raw_output, flags=re.IGNORECASE | re.DOTALL)
        cleaned = []
        for item in extracted:
            text = re.sub(r"\s+", " ", str(item or "")).strip()
            if 10 < len(text) < 200:
                cleaned.append(text)

        unique_queries = list(dict.fromkeys(cleaned))
        if not unique_queries or len(unique_queries) > self.max_subqueries:
            return QueryRewriteResult(
                original_query=original_query,
                queries=[original_query],
                status="fallback_original",
                raw_response=raw_output,
                fallback_reason="invalid_query_count",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        original_numbers = self._extract_numbers(original_query)
        generated_numbers = self._extract_numbers(" ".join(unique_queries))
        missing_numbers = sorted(set(original_numbers) - set(generated_numbers))

        if missing_numbers:
            return QueryRewriteResult(
                original_query=original_query,
                queries=[original_query],
                status="fallback_original",
                raw_response=raw_output,
                missing_numbers=missing_numbers,
                fallback_reason="missing_numbers",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        final_queries = list(unique_queries)
        if original_query not in final_queries:
            final_queries.append(original_query)

        return QueryRewriteResult(
            original_query=original_query,
            queries=final_queries,
            status="ok",
            raw_response=raw_output,
            missing_numbers=[],
            fallback_reason="",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
