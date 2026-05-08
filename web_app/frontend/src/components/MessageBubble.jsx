import React, { useState } from 'react';
import {
  Bot,
  User,
  Copy,
  Download,
  RotateCcw,
  Bookmark,
  ChevronDown,
  ChevronUp,
  Scale,
  FileText,
  ExternalLink,
  Star,
  Award,
  Sparkles,
  Eye,
} from 'lucide-react';
import clsx from 'clsx';
import ReactMarkdown from 'react-markdown';
import QualityDetailsModal from './QualityDetailsModal';

function MessageBubble({ message, onExport, onCopy, onRegenerate, onScore }) {
  const [showSources, setShowSources] = useState(true);
  const [scoreResult, setScoreResult] = useState(null);
  const [isScoring, setIsScoring] = useState(false);
  const [showQualityModal, setShowQualityModal] = useState(false);
  const isUser = message.role === 'user';
  const isError = message.isError;
  const rewriteContext = message.rewriteContext || null;
  const compactQueries = rewriteContext?.rewrite_queries_compact || [];
  const queryFacets = rewriteContext?.query_facets || {};
  const rewriteDebug = message.rewriteDebug || null;
  const retrievalDebug = message.retrievalDebug || null;
  const conditions = rewriteContext?.conditions || rewriteDebug?.conditions || [];
  const clauseUnits = rewriteContext?.clause_units || rewriteDebug?.clause_units || [];
  const namedLawHints = rewriteContext?.named_law_hints || rewriteDebug?.named_law_hints || [];
  const questionUnits = rewriteContext?.question_units || rewriteDebug?.question_units || [];
  const retrievalQueryPlan = message.retrievalQueryPlan || rewriteDebug?.retrieval_query_plan || [];

  const handleScore = async () => {
    if (isScoring) return;

    // If metrics already in message (from backend), use them directly
    if (message.qualityScore) {
      setScoreResult({
        grade: message.qualityGrade,
        overall_score: message.qualityScore,
        bert_score: message.bertScore,
        hallucination_score: message.hallucinationScore,
        factuality_score: message.factualityScore,
        context_relevance: message.contextRelevance,
        feedback: message.qualityFeedback
      });
      return;
    }

    // Otherwise, fetch from API
    if (!onScore) return;
    setIsScoring(true);
    try {
      const result = await onScore(message);
      setScoreResult(result);
    } catch (error) {
      console.error('Score error:', error);
    }
    setIsScoring(false);
  };

  const getGradeColor = (grade) => {
    switch (grade) {
      case 'A': return 'text-emerald-500 bg-emerald-100 dark:bg-emerald-900/50';
      case 'B': return 'text-blue-500 bg-blue-100 dark:bg-blue-900/50';
      case 'C': return 'text-amber-500 bg-amber-100 dark:bg-amber-900/50';
      case 'D': return 'text-orange-500 bg-orange-100 dark:bg-orange-900/50';
      case 'F': return 'text-red-500 bg-red-100 dark:bg-red-900/50';
      default: return 'text-primary-500 bg-primary-100 dark:bg-primary-800';
    }
  };

  return (
    <div className={clsx("flex items-start gap-3", isUser && "flex-row-reverse")}>
      {/* Avatar */}
      <div
        className={clsx(
          "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
          isUser
            ? "bg-primary-200 dark:bg-primary-700"
            : "bg-indigo-100 dark:bg-indigo-900/50"
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 text-primary-600 dark:text-primary-400" />
        ) : (
          <Bot className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
        )}
      </div>

      {/* Message Content */}
      <div className={clsx("flex-1 max-w-[85%]", isUser && "flex flex-col items-end")}>
        {!isUser && (message.modelProfileId || message.queryMode || message.fallbackUsed) && (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {message.modelProfileLabel ? (
              <span className="rounded-full bg-indigo-100 px-2.5 py-1 text-[11px] font-medium text-indigo-700 dark:bg-indigo-900/50 dark:text-indigo-300">
                {message.modelProfileLabel}
              </span>
            ) : null}
            {message.queryMode ? (
              <span className="rounded-full bg-primary-100 px-2.5 py-1 text-[11px] font-medium text-primary-700 dark:bg-primary-800 dark:text-primary-300">
                {message.queryMode === 'rewrite_combine' ? 'Rewrite + Combine' : 'Original Query'}
              </span>
            ) : null}
            {message.fallbackUsed ? (
              <span className="rounded-full bg-amber-100 px-2.5 py-1 text-[11px] font-medium text-amber-700 dark:bg-amber-900/40 dark:text-amber-300">
                Retrieval-only fallback
              </span>
            ) : null}
          </div>
        )}

        {/* Sources (for AI messages) */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mb-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-1.5 text-xs font-medium text-primary-500 dark:text-primary-400 mb-1.5 hover:text-primary-700 dark:hover:text-primary-300"
            >
              <span>Nguồn tham khảo ({message.sources.length})</span>
              {showSources ? (
                <ChevronUp className="w-3.5 h-3.5" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5" />
              )}
            </button>

            {showSources && (
              <div className="flex flex-wrap gap-1.5">
                {message.sources.map((source, i) => (
                  <SourceBadge key={i} source={source} />
                ))}
              </div>
            )}
          </div>
        )}

        {!isUser && retrievalDebug && (
          <div className="mb-2 rounded-xl border border-primary-200 bg-white px-3 py-2 text-xs text-primary-600 dark:border-primary-700 dark:bg-primary-900/40 dark:text-primary-300">
            <div className="flex flex-wrap gap-3">
              <span>scope: <strong>{retrievalDebug.document_scope || 'current_session'}</strong></span>
              <span>ready: <strong>{retrievalDebug.ready_doc_count_in_scope ?? 0}</strong></span>
              <span>total ready: <strong>{retrievalDebug.ready_doc_count_total ?? 0}</strong></span>
              {retrievalDebug.search_strategy_used ? <span>strategy: <strong>{retrievalDebug.search_strategy_used}</strong></span> : null}
            </div>
            {retrievalDebug.no_result_reason ? (
              <div className="mt-1">
                no-result: <strong>{retrievalDebug.no_result_reason}</strong>
              </div>
            ) : null}
            {retrievalDebug.scope_broaden_available ? (
              <div className="mt-1 text-amber-700 dark:text-amber-300">
                Có tài liệu `ready` ở ngoài phạm vi hiện tại. Hãy đổi sang `all_user_docs` nếu cần.
              </div>
            ) : null}
          </div>
        )}

        {!isUser && (message.rewriteStatus || message.rewrittenQueries?.length || message.rewriteContext) && (
          <div className="mb-2 rounded-xl border border-primary-200 bg-primary-50 px-3 py-2 text-xs text-primary-600 dark:border-primary-700 dark:bg-primary-900/40 dark:text-primary-300">
            <div className="flex flex-wrap gap-3">
              <span>rewrite: <strong>{message.rewriteStatus || 'disabled'}</strong></span>
              <span>used: <strong>{String(!!message.rewriteUsed)}</strong></span>
              <span>quality: <strong>{String(!!message.rewriteQualityPassed)}</strong></span>
              {typeof message.retrievalUnionCount === 'number' ? <span>union: <strong>{message.retrievalUnionCount}</strong></span> : null}
            </div>
            {message.rewriteStrategy ? (
              <div className="mt-1">strategy: <strong>{message.rewriteStrategy}</strong></div>
            ) : null}
            {message.rewriteFallbackReason ? (
              <div className="mt-1 text-amber-700 dark:text-amber-300">fallback: {message.rewriteFallbackReason}</div>
            ) : null}
            {!!message.rewrittenQueries?.length && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {message.rewrittenQueries.map((query, idx) => (
                  <span key={`${idx}-${query}`} className="rounded-full bg-white px-2.5 py-1 dark:bg-primary-800">
                    {query}
                  </span>
                ))}
              </div>
            )}
            {rewriteContext?.original_query && (
              <div className="mt-2 rounded-lg border border-primary-200 bg-white px-2.5 py-2 dark:border-primary-700 dark:bg-primary-800">
                <div className="text-[11px] font-semibold uppercase tracking-wide text-primary-500 dark:text-primary-400">Original vs Rewrite</div>
                <div className="mt-1 text-xs">
                  <div className="font-semibold">Original</div>
                  <div className="mt-1 rounded-md bg-primary-50 px-2 py-1.5 dark:bg-primary-900/70">{rewriteContext.original_query}</div>
                </div>
                <div className="mt-2 text-xs">
                  <div className="font-semibold">Rewrite (compact)</div>
                  {!compactQueries.length ? (
                    <div className="mt-1 rounded-md bg-primary-50 px-2 py-1.5 dark:bg-primary-900/70">&lt;none&gt;</div>
                  ) : (
                    <div className="mt-1 flex flex-wrap gap-1.5">
                      {compactQueries.map((query, idx) => (
                        <span key={`${idx}-${query}`} className="rounded-full bg-primary-100 px-2 py-1 dark:bg-primary-900/70">
                          {query}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                {(queryFacets?.actors?.length || queryFacets?.issues?.length || queryFacets?.focus) && (
                  <div className="mt-2 text-xs">
                    <div className="font-semibold">Query Facets</div>
                    <div className="mt-1 space-y-1 rounded-md bg-primary-50 px-2 py-1.5 dark:bg-primary-900/70">
                      <div>main act: {queryFacets.main_act || rewriteContext?.main_act || '&lt;none&gt;'}</div>
                      <div>actors: {(queryFacets.actors || []).join(', ') || '&lt;none&gt;'}</div>
                      <div>issues: {(queryFacets.issues || []).join(', ') || '&lt;none&gt;'}</div>
                      <div>focus: {queryFacets.focus || '&lt;none&gt;'}</div>
                    </div>
                  </div>
                )}
                {!!conditions.length && (
                  <div className="mt-2 text-xs">
                    <div className="font-semibold">Conditions</div>
                    <div className="mt-1 space-y-1 rounded-md bg-primary-50 px-2 py-1.5 dark:bg-primary-900/70">
                      {conditions.map((item, idx) => (
                        <div key={`${idx}-${item.value || item.source_span || idx}`}>
                          {(item.kind || 'khac')}: {item.value || item.source_span || '&lt;none&gt;'}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {!!namedLawHints.length && (
                  <div className="mt-2 text-xs">
                    <div className="font-semibold">Named Law Hints</div>
                    <div className="mt-1 rounded-md bg-primary-50 px-2 py-1.5 dark:bg-primary-900/70">
                      {namedLawHints.join(', ')}
                    </div>
                  </div>
                )}
                {!!questionUnits.length && (
                  <div className="mt-2 text-xs">
                    <div className="font-semibold">Question Units</div>
                    <div className="mt-1 space-y-1 rounded-md bg-primary-50 px-2 py-1.5 dark:bg-primary-900/70">
                      {questionUnits.map((unit, idx) => (
                        <div key={`${idx}-${unit.question_label || idx}`}>
                          <span className="font-medium">{unit.intent || 'xac_dinh_chu_the'}</span>
                          {' | '}
                          <span>{unit.question_label || '&lt;none&gt;'}</span>
                          {' | '}
                          <span>{unit.user_asked_text || '&lt;none&gt;'}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {!!clauseUnits.length && (
                  <div className="mt-2 text-xs">
                    <div className="font-semibold">Clause Units</div>
                    <div className="mt-1 space-y-1 rounded-md bg-primary-50 px-2 py-1.5 dark:bg-primary-900/70">
                      {clauseUnits.map((unit, idx) => (
                        <div key={`${idx}-${unit.clause_label || idx}`}>
                          <span className="font-medium">{unit.intent || 'xac_dinh_chu_the'}</span>
                          {' | '}
                          <span>{(unit.actor_refs || []).join(', ') || '&lt;none&gt;'}</span>
                          {' | '}
                          <span>{(unit.condition_refs || []).join(', ') || unit.act_span || '&lt;none&gt;'}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {!!retrievalQueryPlan.length && (
                  <div className="mt-2 text-xs">
                    <div className="font-semibold">Retrieval Query Plan</div>
                    <div className="mt-1 space-y-1 rounded-md bg-primary-50 px-2 py-1.5 dark:bg-primary-900/70">
                      {retrievalQueryPlan.map((item, idx) => (
                        <div key={`${idx}-${item.query || idx}`}>
                          <span className="font-medium">{item.label || item.source_question_unit || `query_${idx + 1}`}</span>
                          {' | '}
                          <span>{item.query || '&lt;none&gt;'}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {!!message.retrievalQuerySet?.length && (
                  <div className="mt-2 text-xs">
                    <div className="font-semibold">Retrieval Query Set</div>
                    <div className="mt-1 flex flex-wrap gap-1.5">
                      {message.retrievalQuerySet.map((query, idx) => (
                        <span key={`${idx}-${query}`} className="rounded-full bg-primary-100 px-2 py-1 dark:bg-primary-900/70">
                          {query}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {!!rewriteDebug?.rejection_reasons?.length && (
                  <div className="mt-2 text-xs text-amber-700 dark:text-amber-300">
                    rejection: {rewriteDebug.rejection_reasons.join(', ')}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Message Bubble */}
        <div
          className={clsx(
            "rounded-2xl px-4 py-3",
            isUser
              ? "bg-indigo-600 text-white rounded-tr-none"
              : isError
                ? "bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-tl-none"
                : "bg-primary-100 dark:bg-primary-800 text-primary-900 dark:text-primary-100 rounded-tl-none"
          )}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="markdown-content">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Action Toolbar (for AI messages) */}
        {!isUser && !isError && (
          <div className="flex items-center gap-1 mt-2 flex-wrap">
            <ActionButton
              icon={Copy}
              label="Sao chép"
              onClick={onCopy}
            />
            <ActionButton
              icon={Download}
              label="Xuất file"
              onClick={onExport}
            />
            <ActionButton
              icon={RotateCcw}
              label="Tạo lại"
              onClick={onRegenerate}
            />

            {/* Quality Score Button */}
            <button
              onClick={handleScore}
              disabled={isScoring}
              className={clsx(
                "flex items-center gap-1 px-2 py-1 text-xs rounded transition-colors font-medium",
                scoreResult
                  ? getGradeColor(scoreResult.grade)
                  : "text-primary-500 dark:text-primary-400 hover:text-amber-600 dark:hover:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-900/30"
              )}
              title="Chấm điểm câu trả lời"
            >
              {isScoring ? (
                <div className="animate-spin w-3.5 h-3.5 border-2 border-amber-500 border-t-transparent rounded-full" />
              ) : scoreResult ? (
                <Award className="w-3.5 h-3.5" />
              ) : (
                <Star className="w-3.5 h-3.5" />
              )}
              <span className="hidden sm:inline">
                {isScoring ? "Đang chấm..." : scoreResult ? `${scoreResult.grade} (${(scoreResult.overall_score * 100).toFixed(0)}%)` : "Chấm điểm"}
              </span>
            </button>

            {/* View Details Button */}
            {scoreResult && (
              <button
                onClick={() => setShowQualityModal(true)}
                className="flex items-center gap-1 px-2 py-1 text-xs rounded transition-colors font-medium text-indigo-600 dark:text-indigo-400 hover:bg-indigo-50 dark:hover:bg-indigo-900/30"
                title="Xem chi tiết đánh giá"
              >
                <Eye className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Chi tiết</span>
              </button>
            )}

            {/* Performance Metrics */}
            {message.totalTime && (
              <span className="ml-2 text-xs text-primary-400 dark:text-primary-500">
                ⏱️ {message.totalTime.toFixed(2)}s
                {message.rerankerUsed && (
                  <span className="ml-1 text-emerald-500" title="Vietnamese Reranker applied">
                    <Sparkles className="w-3 h-3 inline" />
                  </span>
                )}
              </span>
            )}
          </div>
        )}

        {/* Score Result Display - Compact Version */}
        {scoreResult && !isUser && !showQualityModal && (
          <div className="mt-2 p-3 bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-amber-900/30 dark:to-yellow-900/30 rounded-lg border border-amber-200 dark:border-amber-800">
            <div className="flex items-center gap-3">
              <div className={clsx("w-10 h-10 rounded-full flex items-center justify-center text-xl font-bold", getGradeColor(scoreResult.grade))}>
                {scoreResult.grade}
              </div>
              <div className="flex-1">
                <div className="text-sm font-medium text-primary-900 dark:text-white">
                  Điểm: {(scoreResult.overall_score * 100).toFixed(1)}%
                </div>
                <p className="text-xs text-primary-600 dark:text-primary-400 line-clamp-1">
                  {scoreResult.feedback}
                </p>
              </div>
              <button
                onClick={() => setShowQualityModal(true)}
                className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium rounded-lg transition-colors flex items-center gap-1"
              >
                <Eye className="w-3.5 h-3.5" />
                Xem chi tiết
              </button>
            </div>
          </div>
        )}

        {/* Timestamp */}
        {message.timestamp && (
          <p className="text-xs text-primary-400 dark:text-primary-500 mt-1">
            {new Date(message.timestamp).toLocaleTimeString('vi-VN', {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </p>
        )}
      </div>

      {/* Quality Details Modal */}
      {showQualityModal && scoreResult && (
        <QualityDetailsModal
          message={{
            ...message,
            query: message.query || 'N/A', // Will need to pass this from chat
          }}
          quality={scoreResult}
          onClose={() => setShowQualityModal(false)}
        />
      )}
    </div>
  );
}

function SourceBadge({ source }) {
  const isLegal = source.type === 'legal';
  const [showPreview, setShowPreview] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setShowPreview(!showPreview)}
        className={clsx(
          "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-colors",
          isLegal
            ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-900/60"
            : "bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 hover:bg-amber-200 dark:hover:bg-amber-900/60"
        )}
      >
        {isLegal ? (
          <Scale className="w-3 h-3" />
        ) : (
          <FileText className="w-3 h-3" />
        )}
        <span className="max-w-[150px] truncate">{source.label}</span>
        {source.detail && (
          <span className="opacity-70 max-w-[100px] truncate">• {source.detail}</span>
        )}
      </button>

      {/* Preview Popup - Full Content */}
      {showPreview && (source.content || source.content_preview) && (
        <div className="absolute left-0 top-full mt-1 z-20 w-96 max-h-96 p-3 bg-white dark:bg-primary-800 rounded-lg shadow-xl border border-primary-200 dark:border-primary-700 overflow-hidden flex flex-col">
          <div className="flex items-start justify-between gap-2 mb-2">
            <h4 className="text-sm font-medium text-primary-900 dark:text-white truncate">
              {source.label}
            </h4>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowPreview(false);
              }}
              className="text-primary-400 hover:text-primary-600 dark:hover:text-primary-300 flex-shrink-0"
            >
              ✕
            </button>
          </div>
          {source.detail && (
            <p className="text-xs text-primary-500 dark:text-primary-400 mb-2">
              {source.detail}
            </p>
          )}
          <div className="flex-1 overflow-y-auto bg-primary-50 dark:bg-primary-900/50 rounded p-2">
            <p className="text-xs text-primary-700 dark:text-primary-300 whitespace-pre-wrap">
              {source.content || source.content_preview}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function ActionButton({ icon: Icon, label, onClick }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-1 px-2 py-1 text-xs text-primary-500 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 hover:bg-primary-100 dark:hover:bg-primary-800 rounded transition-colors"
      title={label}
    >
      <Icon className="w-3.5 h-3.5" />
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}

export default MessageBubble;
