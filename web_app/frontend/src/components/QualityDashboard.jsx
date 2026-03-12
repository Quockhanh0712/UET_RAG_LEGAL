import { useState, useEffect } from 'react';
import { BarChart3, Download, Search, Filter, RefreshCw, Eye, X, TrendingUp, Award, Clock, AlertCircle, Sparkles } from 'lucide-react';
import QualityDetailsModal from './QualityDetailsModal';

/**
 * Quality Dashboard Component
 * 
 * Hiển thị tất cả metrics từ database với:
 * - Bảng danh sách với filter/search
 * - Biểu đồ phân bố grades
 * - Statistics overview
 * - Export all functionality
 */
const QualityDashboard = () => {
  const [metrics, setMetrics] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Filters & pagination
  const [searchQuery, setSearchQuery] = useState('');
  const [gradeFilter, setGradeFilter] = useState('');
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState('desc');
  const [page, setPage] = useState(0);
  const [totalRecords, setTotalRecords] = useState(0);
  const LIMIT = 20;
  
  // Modal state
  const [selectedMetric, setSelectedMetric] = useState(null);
  const [showModal, setShowModal] = useState(false);
  
  // Export state
  const [exporting, setExporting] = useState(false);

  // Load data on mount and when filters change
  useEffect(() => {
    loadMetrics();
    loadStats();
  }, [gradeFilter, sortBy, sortOrder, page]);

  const loadMetrics = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const params = new URLSearchParams({
        limit: LIMIT,
        offset: page * LIMIT,
        sort_by: sortBy,
        sort_order: sortOrder
      });
      
      if (gradeFilter) {
        params.append('grade_filter', gradeFilter);
      }
      
      const response = await fetch(`/api/metrics/all?${params}`);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to load metrics: ${response.status} - ${errorText}`);
      }
      
      const data = await response.json();
      
      if (data.status !== 'ok') {
        throw new Error(data.message || 'Invalid response from server');
      }
      
      setMetrics(data.metrics || []);
      setTotalRecords(data.total || 0);
      
    } catch (err) {
      console.error('Load metrics error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const response = await fetch('/api/metrics/stats?days=30');
      
      if (!response.ok) {
        throw new Error('Failed to load stats');
      }
      
      const data = await response.json();
      
      if (data.status === 'ok') {
        setStats(data.statistics);
      }
      
    } catch (err) {
      console.error('Load stats error:', err);
    }
  };

  const handleExport = async (format) => {
    try {
      setExporting(true);
      
      const response = await fetch(`/api/metrics/export?format=${format}`);
      
      if (!response.ok) {
        throw new Error('Export failed');
      }
      
      // Get filename from Content-Disposition header
      const contentDisposition = response.headers.get('Content-Disposition');
      const filenameMatch = contentDisposition?.match(/filename="?(.+)"?/);
      const filename = filenameMatch ? filenameMatch[1] : `metrics_export.${format}`;
      
      // Download file
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
    } catch (err) {
      console.error('Export error:', err);
      alert(`Export failed: ${err.message}`);
    } finally {
      setExporting(false);
    }
  };

  const handleViewDetails = (metric) => {
    setSelectedMetric(metric);
    setShowModal(true);
  };

  const getGradeColor = (grade) => {
    const colors = {
      'A': 'text-green-600 bg-green-100',
      'B': 'text-blue-600 bg-blue-100',
      'C': 'text-yellow-600 bg-yellow-100',
      'D': 'text-orange-600 bg-orange-100',
      'F': 'text-red-600 bg-red-100'
    };
    return colors[grade] || 'text-gray-600 bg-gray-100';
  };

  const getGradeBarColor = (grade) => {
    const colors = {
      'A': 'bg-green-500',
      'B': 'bg-blue-500',
      'C': 'bg-yellow-500',
      'D': 'bg-orange-500',
      'F': 'bg-red-500'
    };
    return colors[grade] || 'bg-gray-500';
  };

  // Filter metrics by search query
  const filteredMetrics = metrics.filter(m => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      (m.query && m.query.toLowerCase().includes(query)) ||
      (m.answer && m.answer.toLowerCase().includes(query)) ||
      (m.feedback && m.feedback.toLowerCase().includes(query))
    );
  });

  const totalPages = Math.ceil(totalRecords / LIMIT);

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <BarChart3 className="w-8 h-8 text-blue-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Quality Dashboard</h1>
              <p className="text-sm text-gray-600">Tổng quan đánh giá chất lượng câu trả lời</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={loadMetrics}
              disabled={loading}
              className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
            
            <div className="flex items-center gap-2 border-l border-gray-300 pl-2">
              <button
                onClick={() => handleExport('json')}
                disabled={exporting}
                className="px-4 py-2 text-blue-700 bg-blue-100 rounded-lg hover:bg-blue-200 transition-colors flex items-center gap-2 disabled:opacity-50"
              >
                <Download className="w-4 h-4" />
                JSON
              </button>
              <button
                onClick={() => handleExport('csv')}
                disabled={exporting}
                className="px-4 py-2 text-green-700 bg-green-100 rounded-lg hover:bg-green-200 transition-colors flex items-center gap-2 disabled:opacity-50"
              >
                <Download className="w-4 h-4" />
                CSV
              </button>
              <button
                onClick={() => handleExport('txt')}
                disabled={exporting}
                className="px-4 py-2 text-purple-700 bg-purple-100 rounded-lg hover:bg-purple-200 transition-colors flex items-center gap-2 disabled:opacity-50"
              >
                <Download className="w-4 h-4" />
                TXT
              </button>
            </div>
          </div>
        </div>

        {/* Statistics Cards */}
        {stats && (
          <div className="grid grid-cols-6 gap-4 mb-4">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-blue-700 font-medium">Total Answers</p>
                  <p className="text-2xl font-bold text-blue-900">{stats.total_answers}</p>
                </div>
                <BarChart3 className="w-8 h-8 text-blue-600 opacity-50" />
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-green-700 font-medium">Avg Score</p>
                  <p className="text-2xl font-bold text-green-900">{(stats.avg_overall_score * 100).toFixed(1)}%</p>
                </div>
                <TrendingUp className="w-8 h-8 text-green-600 opacity-50" />
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-indigo-700 font-medium">Avg BERTScore</p>
                  <p className="text-2xl font-bold text-indigo-900">{stats.avg_bert_score ? (stats.avg_bert_score * 100).toFixed(1) + '%' : 'N/A'}</p>
                </div>
                <Award className="w-8 h-8 text-indigo-600 opacity-50" />
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-emerald-700 font-medium">Avg Factuality</p>
                  <p className="text-2xl font-bold text-emerald-900">{stats.avg_factuality ? (stats.avg_factuality * 100).toFixed(1) + '%' : 'N/A'}</p>
                </div>
                <Clock className="w-8 h-8 text-emerald-600 opacity-50" />
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-purple-700 font-medium">Reranker Used</p>
                  <p className="text-2xl font-bold text-purple-900">{(stats.reranker_usage_rate * 100).toFixed(0)}%</p>
                </div>
                <Sparkles className="w-8 h-8 text-purple-600 opacity-50" />
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-orange-700 font-medium">Hallucination Risk</p>
                  <p className="text-2xl font-bold text-orange-900">{(stats.avg_hallucination_risk * 100).toFixed(1)}%</p>
                </div>
                <AlertCircle className="w-8 h-8 text-orange-600 opacity-50" />
              </div>
            </div>
          </div>
        )}

        {/* Grade Distribution Chart */}
        {stats && stats.grade_distribution && (
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Grade Distribution</h3>
            <div className="flex items-end gap-2 h-32">
              {['A', 'B', 'C', 'D', 'F'].map(grade => {
                const count = stats.grade_distribution[grade] || 0;
                const maxCount = Math.max(...Object.values(stats.grade_distribution), 1);
                const height = (count / maxCount) * 100;
                
                return (
                  <div key={grade} className="flex-1 flex flex-col items-center gap-2">
                    <div className="text-xs font-medium text-gray-600">{count}</div>
                    <div className="w-full bg-gray-100 rounded-t-lg overflow-hidden" style={{ height: '100px' }}>
                      <div
                        className={`w-full ${getGradeBarColor(grade)} transition-all duration-500 rounded-t-lg`}
                        style={{ height: `${height}%`, marginTop: `${100 - height}%` }}
                      />
                    </div>
                    <div className={`text-sm font-bold ${getGradeColor(grade).split(' ')[0]}`}>
                      {grade}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Filters & Search */}
        <div className="flex items-center gap-4 mt-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Tìm kiếm trong query, answer, feedback..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            />
          </div>
          
          <div className="flex items-center gap-2">
            <Filter className="w-5 h-5 text-gray-600" />
            <select
              value={gradeFilter}
              onChange={(e) => { setGradeFilter(e.target.value); setPage(0); }}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            >
              <option value="">All Grades</option>
              <option value="A">Grade A</option>
              <option value="B">Grade B</option>
              <option value="C">Grade C</option>
              <option value="D">Grade D</option>
              <option value="F">Grade F</option>
            </select>
          </div>
          
          <select
            value={sortBy}
            onChange={(e) => { setSortBy(e.target.value); setPage(0); }}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          >
            <option value="created_at">Sort by Date</option>
            <option value="overall_score">Sort by Score</option>
            <option value="grade">Sort by Grade</option>
          </select>
          
          <button
            onClick={() => { setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc'); setPage(0); }}
            className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            {sortOrder === 'asc' ? '↑' : '↓'}
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto p-4">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <RefreshCw className="w-8 h-8 text-blue-600 animate-spin" />
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-64 text-red-600">
            <AlertCircle className="w-6 h-6 mr-2" />
            {error}
          </div>
        ) : filteredMetrics.length === 0 ? (
          <div className="flex items-center justify-center h-64 text-gray-500">
            Không có dữ liệu
          </div>
        ) : (
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Time</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Grade</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Score</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Query</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Feedback</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Reranker</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {filteredMetrics.map((metric) => (
                  <tr key={metric.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-4 py-3 text-sm text-gray-600">
                      {new Date(metric.timestamp).toLocaleString('vi-VN', {
                        year: 'numeric',
                        month: '2-digit',
                        day: '2-digit',
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full text-sm font-bold ${getGradeColor(metric.grade)}`}>
                        {metric.grade}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">
                      {(metric.overall_score * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-700 max-w-xs truncate">
                      {metric.query}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600 max-w-xs truncate">
                      {metric.feedback}
                    </td>
                    <td className="px-4 py-3">
                      {metric.reranker_used ? (
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-700">
                          Yes
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-700">
                          No
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => handleViewDetails(metric)}
                        className="inline-flex items-center gap-1 px-3 py-1 text-sm text-blue-700 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors"
                      >
                        <Eye className="w-4 h-4" />
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="bg-white border-t border-gray-200 px-4 py-3 flex items-center justify-between">
          <div className="text-sm text-gray-700">
            Showing {page * LIMIT + 1} to {Math.min((page + 1) * LIMIT, totalRecords)} of {totalRecords} results
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage(Math.max(0, page - 1))}
              disabled={page === 0}
              className="px-3 py-1 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            
            <div className="flex items-center gap-1">
              {[...Array(Math.min(5, totalPages))].map((_, i) => {
                const pageNum = page < 3 ? i : page - 2 + i;
                if (pageNum >= totalPages) return null;
                
                return (
                  <button
                    key={pageNum}
                    onClick={() => setPage(pageNum)}
                    className={`w-8 h-8 text-sm rounded-lg ${
                      pageNum === page
                        ? 'bg-blue-600 text-white'
                        : 'border border-gray-300 hover:bg-gray-50'
                    }`}
                  >
                    {pageNum + 1}
                  </button>
                );
              })}
            </div>
            
            <button
              onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
              disabled={page >= totalPages - 1}
              className="px-3 py-1 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Details Modal */}
      {showModal && selectedMetric && (
        <QualityDetailsModal
          message={{
            query: selectedMetric.query,
            content: selectedMetric.answer,
            sources: selectedMetric.sources || []
          }}
          quality={{
            overall_score: selectedMetric.overall_score,
            grade: selectedMetric.grade,
            feedback: selectedMetric.feedback,
            bert_score: selectedMetric.bert_score,
            hallucination_score: selectedMetric.hallucination_score,
            factuality_score: selectedMetric.factuality_score,
            context_relevance: selectedMetric.context_relevance
          }}
          onClose={() => setShowModal(false)}
        />
      )}
    </div>
  );
};

export default QualityDashboard;
