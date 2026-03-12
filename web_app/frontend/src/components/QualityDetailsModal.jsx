import React, { useState } from 'react';
import {
  X,
  Download,
  Award,
  FileText,
  MessageSquare,
  CheckCircle,
  AlertTriangle,
  TrendingUp,
  Database,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import clsx from 'clsx';

function QualityDetailsModal({ message, quality, onClose }) {
  const [copied, setCopied] = useState(false);
  const [exportFormat, setExportFormat] = useState('json');
  const [expandedSources, setExpandedSources] = useState({});

  if (!quality) return null;

  const toggleSource = (index) => {
    setExpandedSources(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
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

  const getScoreBarColor = (score) => {
    if (score >= 0.9) return 'bg-emerald-500';
    if (score >= 0.8) return 'bg-blue-500';
    if (score >= 0.7) return 'bg-amber-500';
    if (score >= 0.6) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const exportData = (format) => {
    const data = {
      timestamp: new Date().toISOString(),
      query: message.query || 'N/A',
      answer: message.content,
      quality_metrics: {
        overall_score: quality.overall_score,
        grade: quality.grade,
        feedback: quality.feedback,
        bert_score: quality.bert_score,
        hallucination_score: quality.hallucination_score,
        factuality_score: quality.factuality_score,
        context_relevance: quality.context_relevance,
      },
      sources: message.sources || [],
      performance: {
        total_time: message.totalTime,
        reranker_used: message.rerankerUsed,
      }
    };

    let content, filename, mimeType;

    if (format === 'json') {
      content = JSON.stringify(data, null, 2);
      filename = `quality_report_${Date.now()}.json`;
      mimeType = 'application/json';
    } else if (format === 'txt') {
      content = formatAsText(data);
      filename = `quality_report_${Date.now()}.txt`;
      mimeType = 'text/plain';
    } else if (format === 'csv') {
      content = formatAsCSV(data);
      filename = `quality_report_${Date.now()}.csv`;
      mimeType = 'text/csv';
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const formatAsText = (data) => {
    return `
LEGAL RAG - QUALITY ASSESSMENT REPORT
======================================
Generated: ${new Date(data.timestamp).toLocaleString('vi-VN')}

QUERY:
${data.query}

ANSWER:
${data.answer}

QUALITY METRICS:
----------------
Overall Score: ${(data.quality_metrics.overall_score * 100).toFixed(1)}%
Grade: ${data.quality_metrics.grade}
Feedback: ${data.quality_metrics.feedback}

Detailed Scores:
- BERTScore (Semantic Similarity): ${(data.quality_metrics.bert_score * 100).toFixed(1)}%
- Hallucination Score (No False Claims): ${(data.quality_metrics.hallucination_score * 100).toFixed(1)}%
- Factuality Score (Accuracy): ${(data.quality_metrics.factuality_score * 100).toFixed(1)}%
- Context Relevance (Source Quality): ${(data.quality_metrics.context_relevance * 100).toFixed(1)}%

SOURCES (${data.sources.length}):
${data.sources.map((s, i) => `${i + 1}. ${s.label} ${s.detail ? '- ' + s.detail : ''}`).join('\n')}

PERFORMANCE:
Total Time: ${data.performance.total_time?.toFixed(2) || 'N/A'}s
Reranker Used: ${data.performance.reranker_used ? 'Yes' : 'No'}
======================================
`;
  };

  const formatAsCSV = (data) => {
    const rows = [
      ['Field', 'Value'],
      ['Timestamp', new Date(data.timestamp).toLocaleString('vi-VN')],
      ['Query', data.query],
      ['Answer', data.answer],
      ['Overall Score', (data.quality_metrics.overall_score * 100).toFixed(1) + '%'],
      ['Grade', data.quality_metrics.grade],
      ['Feedback', data.quality_metrics.feedback],
      ['BERTScore', (data.quality_metrics.bert_score * 100).toFixed(1) + '%'],
      ['Hallucination Score', (data.quality_metrics.hallucination_score * 100).toFixed(1) + '%'],
      ['Factuality Score', (data.quality_metrics.factuality_score * 100).toFixed(1) + '%'],
      ['Context Relevance', (data.quality_metrics.context_relevance * 100).toFixed(1) + '%'],
      ['Total Time', data.performance.total_time?.toFixed(2) + 's'],
      ['Reranker Used', data.performance.reranker_used ? 'Yes' : 'No'],
      ['Sources Count', data.sources.length],
    ];
    return rows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
  };

  return (
    <div className="fixed inset-0 bg-black/50 dark:bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fadeIn">
      <div className="bg-white dark:bg-primary-900 rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col animate-slideUp">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-primary-200 dark:border-primary-700">
          <div className="flex items-center gap-3">
            <div className={clsx("w-12 h-12 rounded-full flex items-center justify-center text-2xl font-bold", getGradeColor(quality.grade))}>
              {quality.grade}
            </div>
            <div>
              <h2 className="text-xl font-bold text-primary-900 dark:text-white">
                Chi Tiết Đánh Giá Chất Lượng
              </h2>
              <p className="text-sm text-primary-500 dark:text-primary-400">
                Quality Assessment Report
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-primary-100 dark:hover:bg-primary-800 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-primary-500 dark:text-primary-400" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Overall Score Section */}
          <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/30 dark:to-purple-900/30 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <Award className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
              <h3 className="text-lg font-semibold text-primary-900 dark:text-white">
                Điểm Tổng Quan
              </h3>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-4xl font-bold text-indigo-600 dark:text-indigo-400">
                  {(quality.overall_score * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-primary-600 dark:text-primary-400 mt-1">
                  Overall Quality Score
                </div>
              </div>
              <div className="flex items-center">
                <div className={clsx("text-6xl font-bold px-6 py-3 rounded-xl", getGradeColor(quality.grade))}>
                  {quality.grade}
                </div>
              </div>
            </div>
            <p className="mt-4 text-sm text-primary-700 dark:text-primary-300 italic">
              💬 {quality.feedback}
            </p>
          </div>

          {/* Detailed Metrics */}
          <div className="bg-primary-50 dark:bg-primary-800/50 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <TrendingUp className="w-6 h-6 text-primary-600 dark:text-primary-400" />
              <h3 className="text-lg font-semibold text-primary-900 dark:text-white">
                Chỉ Số Chi Tiết
              </h3>
            </div>
            
            <div className="space-y-4">
              {/* BERTScore */}
              <MetricBar
                label="BERTScore"
                description="Độ tương đồng ngữ nghĩa giữa câu trả lời và tài liệu"
                score={quality.bert_score || 0}
                icon={CheckCircle}
              />

              {/* Hallucination Score */}
              <MetricBar
                label="Hallucination Score"
                description="Phát hiện ảo giác - claims không có trong tài liệu (1.0 = không ảo giác)"
                score={quality.hallucination_score || 0}
                icon={AlertTriangle}
              />

              {/* Factuality Score */}
              <MetricBar
                label="Factuality Score"
                description="Độ chính xác thông tin - kiểm tra số liệu, ngày tháng, điều khoản"
                score={quality.factuality_score || 0}
                icon={Database}
              />

              {/* Context Relevance */}
              <MetricBar
                label="Context Relevance"
                description="Chất lượng tài liệu nguồn - độ liên quan với câu hỏi"
                score={quality.context_relevance || 0}
                icon={FileText}
              />
            </div>
          </div>

          {/* Query Section */}
          <div className="bg-blue-50 dark:bg-blue-900/30 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-3">
              <MessageSquare className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <h3 className="text-base font-semibold text-primary-900 dark:text-white">
                Câu Hỏi
              </h3>
            </div>
            <div className="bg-white dark:bg-primary-800 rounded-lg p-4 relative group">
              <p className="text-sm text-primary-700 dark:text-primary-300 whitespace-pre-wrap">
                {message.query || 'N/A'}
              </p>
              <button
                onClick={() => copyToClipboard(message.query || '')}
                className="absolute top-2 right-2 p-1.5 bg-primary-100 dark:bg-primary-700 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                title="Copy query"
              >
                {copied ? <Check className="w-4 h-4 text-green-600" /> : <Copy className="w-4 h-4 text-primary-600 dark:text-primary-400" />}
              </button>
            </div>
          </div>

          {/* Answer Section */}
          <div className="bg-green-50 dark:bg-green-900/30 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-3">
              <FileText className="w-5 h-5 text-green-600 dark:text-green-400" />
              <h3 className="text-base font-semibold text-primary-900 dark:text-white">
                Câu Trả Lời
              </h3>
            </div>
            <div className="bg-white dark:bg-primary-800 rounded-lg p-4 max-h-64 overflow-y-auto relative group">
              <p className="text-sm text-primary-700 dark:text-primary-300 whitespace-pre-wrap">
                {message.content}
              </p>
              <button
                onClick={() => copyToClipboard(message.content)}
                className="absolute top-2 right-2 p-1.5 bg-primary-100 dark:bg-primary-700 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                title="Copy answer"
              >
                {copied ? <Check className="w-4 h-4 text-green-600" /> : <Copy className="w-4 h-4 text-primary-600 dark:text-primary-400" />}
              </button>
            </div>
          </div>

          {/* Sources Section */}
          {message.sources && message.sources.length > 0 && (
            <div className="bg-amber-50 dark:bg-amber-900/30 rounded-xl p-6">
              <div className="flex items-center gap-3 mb-3">
                <Database className="w-5 h-5 text-amber-600 dark:text-amber-400" />
                <h3 className="text-base font-semibold text-primary-900 dark:text-white">
                  Nguồn Tham Khảo ({message.sources.length})
                </h3>
              </div>
              <div className="space-y-3">
                {message.sources.map((source, i) => {
                  const isExpanded = expandedSources[i];
                  const hasContent = source.content || source.content_preview;
                  
                  return (
                    <div key={i} className="bg-white dark:bg-primary-800 rounded-lg overflow-hidden">
                      {/* Header */}
                      <div className="p-3 flex items-start gap-3">
                        <div className="w-6 h-6 rounded-full bg-amber-100 dark:bg-amber-900/50 flex items-center justify-center text-xs font-medium text-amber-600 dark:text-amber-400 flex-shrink-0">
                          {i + 1}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between gap-2">
                            <div className="font-medium text-sm text-primary-900 dark:text-white">
                              {source.label}
                            </div>
                            {source.score && (
                              <div className="text-xs font-medium text-amber-600 dark:text-amber-400">
                                {(source.score * 100).toFixed(0)}%
                              </div>
                            )}
                          </div>
                          {(source.detail || source.details) && (
                            <div className="text-xs text-primary-500 dark:text-primary-400 mt-0.5">
                              {source.detail || source.details}
                            </div>
                          )}
                        </div>
                        {hasContent && (
                          <button
                            onClick={() => toggleSource(i)}
                            className="p-1 hover:bg-primary-100 dark:hover:bg-primary-700 rounded transition-colors flex-shrink-0"
                            title={isExpanded ? "Thu gọn" : "Xem đầy đủ"}
                          >
                            {isExpanded ? (
                              <ChevronUp className="w-4 h-4 text-primary-600 dark:text-primary-400" />
                            ) : (
                              <ChevronDown className="w-4 h-4 text-primary-600 dark:text-primary-400" />
                            )}
                          </button>
                        )}
                      </div>
                      
                      {/* Content */}
                      {hasContent && (
                        <div className={clsx(
                          "px-3 pb-3",
                          !isExpanded && "max-h-0 overflow-hidden",
                          isExpanded && "max-h-none"
                        )}>
                          <div className="bg-primary-50 dark:bg-primary-900/50 rounded p-3 text-xs text-primary-700 dark:text-primary-300 whitespace-pre-wrap max-h-96 overflow-y-auto">
                            {source.content || source.content_preview}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Footer - Export Options */}
        <div className="border-t border-primary-200 dark:border-primary-700 p-6 bg-primary-50 dark:bg-primary-800/50">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <Download className="w-5 h-5 text-primary-600 dark:text-primary-400" />
              <span className="text-sm font-medium text-primary-900 dark:text-white">
                Xuất Báo Cáo:
              </span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => exportData('json')}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                JSON
              </button>
              <button
                onClick={() => exportData('txt')}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                TXT
              </button>
              <button
                onClick={() => exportData('csv')}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                CSV
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricBar({ label, description, score, icon: Icon }) {
  const getScoreBarColor = (score) => {
    if (score >= 0.9) return 'bg-emerald-500';
    if (score >= 0.8) return 'bg-blue-500';
    if (score >= 0.7) return 'bg-amber-500';
    if (score >= 0.6) return 'bg-orange-500';
    return 'bg-red-500';
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-primary-600 dark:text-primary-400" />
          <span className="text-sm font-medium text-primary-900 dark:text-white">
            {label}
          </span>
        </div>
        <span className="text-sm font-bold text-primary-900 dark:text-white">
          {(score * 100).toFixed(1)}%
        </span>
      </div>
      <div className="w-full h-2.5 bg-primary-200 dark:bg-primary-700 rounded-full overflow-hidden">
        <div
          className={clsx("h-full transition-all duration-500", getScoreBarColor(score))}
          style={{ width: `${score * 100}%` }}
        />
      </div>
      <p className="text-xs text-primary-500 dark:text-primary-400 mt-1">
        {description}
      </p>
    </div>
  );
}

export default QualityDetailsModal;
