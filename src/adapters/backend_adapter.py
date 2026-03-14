from __future__ import annotations

from typing import Any, Dict, List

from src.legal_rag_pipeline import LegalAnswer, LegalSearchResult


def format_legal_source(citation: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = str(citation.get('doc_id') or citation.get('van_ban_id') or '')
    legal_path = str(citation.get('legal_path') or citation.get('path') or '')
    citation_full = str(citation.get('citation_full') or '').strip()
    label = citation_full or str(citation.get('ten_van_ban') or doc_id or 'Van ban phap ly')
    detail = legal_path
    return {
        'type': 'legal',
        'label': label,
        'detail': detail,
        'score': float(citation.get('score', 0.0) or 0.0),
        'content_preview': str(citation.get('content', '') or ''),
        'content': str(citation.get('content', '') or ''),
        'doc_id': doc_id,
        'citation_full': citation_full,
        'legal_path': legal_path,
        'chunk_type': str(citation.get('chunk_type', '') or ''),
        'is_active': bool(citation.get('is_active', True)),
        'ten_van_ban': str(citation.get('ten_van_ban', '') or ''),
    }


def format_legal_sources(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [format_legal_source(citation) for citation in list(citations or [])]


def _result_metrics(result: LegalSearchResult | LegalAnswer) -> Dict[str, Any]:
    metrics = {
        'rewrite_used': bool(getattr(result, 'rewrite_used', False)),
        'rewrite_status': str(getattr(result, 'rewrite_status', 'disabled')),
        'rewrite_fallback_reason': str(getattr(result, 'rewrite_fallback_reason', '')),
        'rewritten_queries': list(getattr(result, 'rewritten_queries', []) or []),
        'missing_numbers': list(getattr(result, 'missing_numbers', []) or []),
        'per_subquery_hits': list(getattr(result, 'per_subquery_hits', []) or []),
        'retrieval_union_count': int(getattr(result, 'retrieval_union_count', 0) or 0),
        'reranker_used': bool(getattr(result, 'used_reranker', False)),
        'rerank_time': float(getattr(result, 'rerank_time', 0.0) or 0.0),
        'retrieval_time': float(getattr(result, 'retrieval_time', 0.0) or 0.0),
    }
    if isinstance(result, LegalAnswer):
        metrics['generation_time'] = float(getattr(result, 'generation_time', 0.0) or 0.0)
    return metrics


def build_search_payload(
    result: LegalSearchResult,
    *,
    mode: str,
    collection: str,
    model_profile_id: str,
    query_mode: str,
) -> Dict[str, Any]:
    metrics = _result_metrics(result)
    return {
        'results': format_legal_sources(result.citations),
        'search_time': float(metrics['retrieval_time']),
        'mode': mode,
        'rerank_time': float(metrics['rerank_time']),
        'reranker_used': bool(metrics['reranker_used']),
        'model_profile_id': model_profile_id,
        'query_mode': query_mode,
        'rewrite_used': bool(metrics['rewrite_used']),
        'rewrite_status': str(metrics['rewrite_status']),
        'rewrite_fallback_reason': str(metrics['rewrite_fallback_reason']),
        'rewritten_queries': list(metrics['rewritten_queries']),
        'per_subquery_hits': list(metrics['per_subquery_hits']),
        'retrieval_union_count': int(metrics['retrieval_union_count']),
        'fallback_used': False,
        'fallback_error': '',
        'collection': collection,
    }


def build_chat_payload(
    result: LegalAnswer,
    *,
    answer: str,
    collection: str,
    model_profile_id: str,
    query_mode: str,
    fallback_used: bool = False,
    fallback_error: str = '',
) -> Dict[str, Any]:
    metrics = _result_metrics(result)
    generation_time = float(getattr(result, 'generation_time', 0.0) or 0.0)
    total_time = float(metrics['retrieval_time']) + float(metrics['rerank_time']) + generation_time
    return {
        'answer': answer,
        'sources': format_legal_sources(result.citations),
        'search_time': float(metrics['retrieval_time']),
        'generation_time': generation_time,
        'total_time': total_time,
        'rerank_time': float(metrics['rerank_time']),
        'reranker_used': bool(metrics['reranker_used']),
        'model_profile_id': model_profile_id,
        'query_mode': query_mode,
        'rewrite_used': bool(metrics['rewrite_used']),
        'rewrite_status': str(metrics['rewrite_status']),
        'rewrite_fallback_reason': str(metrics['rewrite_fallback_reason']),
        'rewritten_queries': list(metrics['rewritten_queries']),
        'per_subquery_hits': list(metrics['per_subquery_hits']),
        'retrieval_union_count': int(metrics['retrieval_union_count']),
        'fallback_used': bool(fallback_used),
        'fallback_error': str(fallback_error or ''),
        'collection': collection,
    }
