from __future__ import annotations

from typing import Any, Dict, List

from src.legal_rag.contracts.results import LegalAnswer, LegalSearchResult


def format_legal_source(citation: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = str(citation.get('doc_id') or citation.get('van_ban_id') or '')
    legal_path = str(citation.get('legal_path') or citation.get('path') or '')
    citation_full = str(citation.get('citation_full') or '').strip()
    source_type = str(citation.get('source_type') or 'legal').strip().lower()
    filename = str(citation.get('filename') or citation.get('ten_van_ban') or '').strip()
    page_number = citation.get('page_number')
    page_detail = f"Trang {page_number}" if page_number not in (None, "") else ""

    if source_type == 'user_upload':
        source_type = 'user'

    if source_type == 'user':
        label = citation_full or filename or str(citation.get('ten_van_ban') or doc_id or 'Tai lieu nguoi dung')
        detail_parts = [item for item in (filename, legal_path, page_detail) if str(item or '').strip()]
        detail = " | ".join(detail_parts)
    else:
        label = citation_full or str(citation.get('ten_van_ban') or doc_id or 'Van ban phap ly')
        detail = legal_path
    return {
        'type': source_type if source_type in {'legal', 'user'} else 'legal',
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
        'filename': filename,
        'page_number': page_number,
        'source_type': source_type,
    }


def format_legal_sources(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [format_legal_source(citation) for citation in list(citations or [])]


def _result_metrics(result: LegalSearchResult | LegalAnswer) -> Dict[str, Any]:
    metrics = {
        'rewrite_used': bool(getattr(result, 'rewrite_used', False)),
        'rewrite_status': str(getattr(result, 'rewrite_status', 'disabled')),
        'rewrite_fallback_reason': str(getattr(result, 'rewrite_fallback_reason', '')),
        'rewrite_strategy': str(getattr(result, 'rewrite_strategy', 'structured_synthesis')),
        'rewrite_quality_passed': bool(getattr(result, 'rewrite_quality_passed', False)),
        'rewritten_queries': list(getattr(result, 'rewritten_queries', []) or []),
        'retrieval_query_set': list(getattr(result, 'retrieval_query_set', []) or []),
        'retrieval_query_plan': list(getattr(result, 'retrieval_query_plan', []) or []),
        'missing_numbers': list(getattr(result, 'missing_numbers', []) or []),
        'per_subquery_hits': list(getattr(result, 'per_subquery_hits', []) or []),
        'retrieval_union_count': int(getattr(result, 'retrieval_union_count', 0) or 0),
        'rewrite_mode': str(getattr(result, 'rewrite_mode', 'original') or 'original'),
        'rewrite_original_query': str(getattr(result, 'rewrite_original_query', '') or ''),
        'rewrite_queries_compact': list(getattr(result, 'rewrite_queries_compact', []) or []),
        'query_facets': dict(getattr(result, 'query_facets', {}) or {}),
        'rewrite_debug': dict(getattr(result, 'rewrite_debug', {}) or {}),
        'retrieval_debug': dict(getattr(result, 'retrieval_debug', {}) or {}),
        'reranker_used': bool(getattr(result, 'used_reranker', False)),
        'rerank_time': float(getattr(result, 'rerank_time', 0.0) or 0.0),
        'retrieval_time': float(getattr(result, 'retrieval_time', 0.0) or 0.0),
    }
    if isinstance(result, LegalAnswer):
        metrics['generation_time'] = float(getattr(result, 'generation_time', 0.0) or 0.0)
    return metrics


def _build_rewrite_context(metrics: Dict[str, Any], query_mode: str) -> Dict[str, Any]:
    query_facets = dict(metrics.get('query_facets', {}) or {})
    rewrite_debug = dict(metrics.get('rewrite_debug', {}) or {})
    return {
        'original_query': str(metrics.get('rewrite_original_query', '') or ''),
        'rewrite_mode': str(query_mode or metrics.get('rewrite_mode') or 'original'),
        'rewrite_status': str(metrics.get('rewrite_status', 'disabled') or 'disabled'),
        'rewrite_queries_compact': list(metrics.get('rewrite_queries_compact', []) or []),
        'rewrite_fallback_reason': str(metrics.get('rewrite_fallback_reason', '') or ''),
        'query_facets': query_facets,
        'rewrite_quality_passed': bool(metrics.get('rewrite_quality_passed', False)),
        'main_act': str(query_facets.get('main_act') or rewrite_debug.get('main_act') or ''),
        'conditions': list(rewrite_debug.get('conditions', []) or []),
        'clause_units': list(rewrite_debug.get('clause_units', []) or []),
        'named_law_hints': list(rewrite_debug.get('named_law_hints', []) or []),
        'question_units': list(rewrite_debug.get('question_units', []) or []),
        'main_act_candidate_reasons': list(rewrite_debug.get('main_act_candidate_reasons', []) or []),
        'decomposition_policy': str(rewrite_debug.get('decomposition_policy') or 'objective_question_unit'),
        'quality_fail_reasons': list(rewrite_debug.get('quality_fail_reasons', []) or []),
    }


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
        'rewrite_strategy': str(metrics['rewrite_strategy']),
        'rewrite_quality_passed': bool(metrics['rewrite_quality_passed']),
        'rewritten_queries': list(metrics['rewritten_queries']),
        'retrieval_query_set': list(metrics['retrieval_query_set']),
        'retrieval_query_plan': list(metrics['retrieval_query_plan']),
        'rewrite_context': _build_rewrite_context(metrics, query_mode),
        'rewrite_debug': dict(metrics['rewrite_debug']),
        'retrieval_debug': dict(metrics['retrieval_debug']),
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
        'rewrite_strategy': str(metrics['rewrite_strategy']),
        'rewrite_quality_passed': bool(metrics['rewrite_quality_passed']),
        'rewritten_queries': list(metrics['rewritten_queries']),
        'retrieval_query_set': list(metrics['retrieval_query_set']),
        'retrieval_query_plan': list(metrics['retrieval_query_plan']),
        'rewrite_context': _build_rewrite_context(metrics, query_mode),
        'rewrite_debug': dict(metrics['rewrite_debug']),
        'retrieval_debug': dict(metrics['retrieval_debug']),
        'per_subquery_hits': list(metrics['per_subquery_hits']),
        'retrieval_union_count': int(metrics['retrieval_union_count']),
        'fallback_used': bool(fallback_used),
        'fallback_error': str(fallback_error or ''),
        'collection': collection,
    }
