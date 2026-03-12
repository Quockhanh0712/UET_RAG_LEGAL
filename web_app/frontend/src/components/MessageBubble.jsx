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
