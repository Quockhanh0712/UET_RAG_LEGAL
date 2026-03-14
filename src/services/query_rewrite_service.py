from __future__ import annotations

import logging

from src.config.runtime_settings import RuntimeSettings
from src.legal_query_rewriter import LegalQueryRewriter, QueryRewriteResult

logger = logging.getLogger(__name__)


class QueryRewriteService:
    def __init__(
        self,
        enabled: bool,
        force: bool,
        rewriter: LegalQueryRewriter | None = None,
    ):
        self.enabled = bool(enabled)
        self.force = bool(force)
        self.rewriter = rewriter or LegalQueryRewriter()

    @classmethod
    def from_settings(cls, settings: RuntimeSettings) -> "QueryRewriteService":
        rewriter = LegalQueryRewriter(
            tgi_url=settings.query_rewriter_url,
            timeout_s=settings.query_rewriter_timeout_s,
            max_subqueries=settings.query_rewriter_max_subqueries,
        )
        return cls(
            enabled=settings.query_rewriter_enabled,
            force=settings.query_rewriter_force,
            rewriter=rewriter,
        )

    def rewrite(self, question: str) -> QueryRewriteResult:
        if not self.enabled:
            return QueryRewriteResult(
                original_query=str(question or ""),
                queries=[str(question or "")],
                status="disabled",
            )
        try:
            return self.rewriter.rewrite_query(str(question or ""), force=self.force)
        except Exception as exc:
            logger.warning("[REWRITE-SERVICE] rewrite failed, fallback to original query: %s", exc)
            return QueryRewriteResult(
                original_query=str(question or ""),
                queries=[str(question or "")],
                status="api_error",
                fallback_reason=str(exc),
            )

