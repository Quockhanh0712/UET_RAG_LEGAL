import React, { useState, useEffect, useRef, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import LegalLabArea from './components/LegalLabArea';
import QualityDashboard from './components/QualityDashboard';
import LegalOCR from './components/LegalOCR';
import Toast from './components/Toast';

// API Base URL
const API_BASE = '/api';

function App() {
  // Theme state
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });

  // Session and user state
  const [userId] = useState(() => localStorage.getItem('userId') || uuidv4());
  const [sessionId, setSessionId] = useState(() => uuidv4());

  // Chat state
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchMode, setSearchMode] = useState('hybrid'); // legal, user, hybrid
  const [documentScope, setDocumentScope] = useState(() => localStorage.getItem('documentScope') || 'current_session');
  const [rerankerEnabled, setRerankerEnabled] = useState(true); // Vietnamese Reranker toggle
  const [modelProfiles, setModelProfiles] = useState([]);
  const [selectedModelProfileId, setSelectedModelProfileId] = useState(() => localStorage.getItem('selectedModelProfileId') || 'qwen3_5_9b_ollama_remote');
  const [queryMode, setQueryMode] = useState(() => localStorage.getItem('queryMode') || 'original');
  const [labTab, setLabTab] = useState(() => localStorage.getItem('labTab') || 'search');
  const [requireActiveOnly, setRequireActiveOnly] = useState(() => {
    const saved = localStorage.getItem('requireActiveOnly');
    return saved ? JSON.parse(saved) : true;
  });
  const [topK, setTopK] = useState(() => {
    const saved = localStorage.getItem('topK');
    return saved ? Number(saved) : 5;
  });
  const [searchRuns, setSearchRuns] = useState([]);

  // Documents state
  const [documents, setDocuments] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [uploadMode, setUploadMode] = useState(() => localStorage.getItem('uploadMode') || 'text');

  // Chat history state
  const [sessions, setSessions] = useState([]);

  // Toast state
  const [toast, setToast] = useState(null);

  // System status
  const [systemStatus, setSystemStatus] = useState(null);

  // View state (chat or dashboard)
  const [currentView, setCurrentView] = useState('chat');

  // Save userId to localStorage
  useEffect(() => {
    localStorage.setItem('userId', userId);
  }, [userId]);

  useEffect(() => {
    localStorage.setItem('selectedModelProfileId', selectedModelProfileId);
  }, [selectedModelProfileId]);

  useEffect(() => {
    localStorage.setItem('queryMode', queryMode);
  }, [queryMode]);

  useEffect(() => {
    localStorage.setItem('documentScope', documentScope);
  }, [documentScope]);

  useEffect(() => {
    localStorage.setItem('labTab', labTab);
  }, [labTab]);

  useEffect(() => {
    localStorage.setItem('requireActiveOnly', JSON.stringify(requireActiveOnly));
  }, [requireActiveOnly]);

  useEffect(() => {
    localStorage.setItem('topK', String(topK));
  }, [topK]);

  useEffect(() => {
    localStorage.setItem('uploadMode', uploadMode);
  }, [uploadMode]);

  // Apply dark mode
  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Fetch system status on mount
  useEffect(() => {
    fetchSystemStatus();
    fetchDocuments();
    fetchSessions();
    fetchRerankerSettings();
    fetchModelProfiles();
  }, []);

  const fetchRerankerSettings = async () => {
    try {
      const res = await fetch(`${API_BASE}/settings/reranker`);
      const data = await res.json();
      setRerankerEnabled(data.enabled);
    } catch (error) {
      console.error('Failed to fetch reranker settings:', error);
    }
  };

  const updateRerankerSettings = async (enabled) => {
    try {
      await fetch(`${API_BASE}/settings/reranker?enabled=${enabled}`, {
        method: 'PUT',
      });
      setRerankerEnabled(enabled);
      setToast({
        type: 'success',
        message: enabled ? '✨ Vietnamese Reranker đã bật' : '🔄 Đã tắt Reranker',
      });
    } catch (error) {
      console.error('Failed to update reranker settings:', error);
    }
  };

  const fetchSystemStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/status`);
      const data = await res.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Failed to fetch status:', error);
    }
  };

  const fetchModelProfiles = async () => {
    try {
      const res = await fetch(`${API_BASE}/settings/model-profiles`);
      const data = await res.json();
      const profiles = data.profiles || [];
      setModelProfiles(profiles);
      if (profiles.length > 0 && !profiles.some((item) => item.id === selectedModelProfileId)) {
        setSelectedModelProfileId(data.default_profile_id || profiles[0].id);
      }
    } catch (error) {
      console.error('Failed to fetch model profiles:', error);
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/documents?user_id=${userId}`);
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    }
  };

  const pollDocumentStatus = useCallback(async (docId) => {
    const terminalStatuses = new Set(['ready', 'needs_ocr', 'failed']);
    for (let attempt = 0; attempt < 30; attempt += 1) {
      try {
        const res = await fetch(`${API_BASE}/documents/${docId}/status?user_id=${userId}`);
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Status poll failed');
        }
        setUploadProgress(prev => (
          prev && prev.docId === docId
            ? { ...prev, status: data.status, chunkCount: data.chunk_count || 0 }
            : prev
        ));
        await fetchDocuments();
        if (terminalStatuses.has(data.status)) {
          return data;
        }
      } catch (error) {
        console.error('Status poll error:', error);
        break;
      }
      // eslint-disable-next-line no-await-in-loop
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
    return null;
  }, [userId]);

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions?user_id=${userId}`);
      const data = await res.json();
      setSessions(data.sessions || []);
    } catch (error) {
      console.error('Failed to fetch sessions:', error);
    }
  };

  const loadSession = async (selectedSessionId) => {
    try {
      const res = await fetch(`${API_BASE}/history?session_id=${selectedSessionId}`);
      const data = await res.json();

      // Convert API messages to our format
      const loadedMessages = (data.messages || []).map(msg => ({
        id: msg.id,
        role: msg.role,
        content: msg.content,
        sources: msg.sources || [],
        timestamp: msg.timestamp,
        searchTime: msg.search_time,
        genTime: msg.generation_time,
        totalTime: msg.total_time,
        rerankTime: msg.rerank_time,
        rerankerUsed: msg.reranker_used,
        modelProfileId: msg.model_profile_id,
        modelProfileLabel: (modelProfiles || []).find((item) => item.id === msg.model_profile_id)?.label,
        queryMode: msg.query_mode,
        rewriteUsed: msg.rewrite_used,
        rewriteStatus: msg.rewrite_status,
        rewriteFallbackReason: msg.rewrite_fallback_reason,
        rewriteStrategy: msg.rewrite_strategy,
        rewriteQualityPassed: msg.rewrite_quality_passed,
        rewrittenQueries: msg.rewritten_queries || [],
        retrievalQuerySet: msg.retrieval_query_set || [],
        retrievalQueryPlan: msg.retrieval_query_plan || [],
        rewriteContext: msg.rewrite_context || null,
        rewriteDebug: msg.rewrite_debug || null,
        retrievalDebug: msg.retrieval_debug || null,
        perSubqueryHits: msg.per_subquery_hits || [],
        retrievalUnionCount: msg.retrieval_union_count,
        fallbackUsed: msg.fallback_used,
        fallbackError: msg.fallback_error,
        collection: msg.collection,
      }));

      setMessages(loadedMessages);
      setSessionId(selectedSessionId);
    } catch (error) {
      console.error('Failed to load session:', error);
    }
  };

  const startNewChat = () => {
    setSessionId(uuidv4());
    setMessages([]);
    setSearchRuns([]);
    fetchSessions();
  };

  const isSharedLegalMode = currentView === 'chat' && searchMode === 'legal' && !!systemStatus?.shared_pipeline_enabled;

  const sendMessage = async (content) => {
    if (!content.trim() || isLoading) return;

    if (isSharedLegalMode) {
      const userMessage = {
        id: uuidv4(),
        role: 'user',
        content: content.trim(),
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, userMessage]);
      setIsLoading(true);

      try {
        const res = await fetch(`${API_BASE}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: content.trim(),
            user_id: userId,
            session_id: sessionId,
            search_mode: 'legal',
            top_k: topK,
            use_reranker: rerankerEnabled,
            model_profile_id: selectedModelProfileId,
            query_mode: queryMode,
            document_scope: documentScope,
            response_mode: 'chat',
            require_active_only: requireActiveOnly,
            use_graph: false,
          }),
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Shared legal chat failed');
        }

        const selectedProfile = modelProfiles.find((item) => item.id === data.model_profile_id || item.id === selectedModelProfileId);
        const assistantMessage = {
          id: uuidv4(),
          role: 'assistant',
          content: data.answer,
          sources: data.sources || [],
          timestamp: new Date().toISOString(),
          searchTime: data.search_time,
          genTime: data.generation_time,
          totalTime: data.total_time,
          rerankTime: data.rerank_time,
          rerankerUsed: data.reranker_used,
          modelProfileId: data.model_profile_id || selectedModelProfileId,
          modelProfileLabel: selectedProfile?.label,
          queryMode: data.query_mode || queryMode,
          rewriteUsed: data.rewrite_used,
          rewriteStatus: data.rewrite_status,
          rewriteFallbackReason: data.rewrite_fallback_reason,
          rewriteStrategy: data.rewrite_strategy,
          rewriteQualityPassed: data.rewrite_quality_passed,
          rewrittenQueries: data.rewritten_queries || [],
          retrievalQuerySet: data.retrieval_query_set || [],
          retrievalQueryPlan: data.retrieval_query_plan || [],
          rewriteContext: data.rewrite_context || null,
          rewriteDebug: data.rewrite_debug || null,
          retrievalDebug: data.retrieval_debug || null,
          perSubqueryHits: data.per_subquery_hits || [],
          retrievalUnionCount: data.retrieval_union_count,
          fallbackUsed: data.fallback_used,
          fallbackError: data.fallback_error,
          collection: data.collection,
        };
        setMessages(prev => [...prev, assistantMessage]);
        fetchSessions();
      } catch (error) {
        console.error('Shared legal chat error:', error);
        const errorMessage = {
          id: uuidv4(),
          role: 'assistant',
          content: '❌ Có lỗi xảy ra khi xử lý legal pipeline. Vui lòng thử lại.',
          sources: [],
          timestamp: new Date().toISOString(),
          isError: true,
        };
        setMessages(prev => [...prev, errorMessage]);
      } finally {
        setIsLoading(false);
      }
      return;
    }

    // Add user message
    const userMessage = {
      id: uuidv4(),
      role: 'user',
      content: content.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: content.trim(),
          user_id: userId,
          session_id: sessionId,
          search_mode: searchMode,
          top_k: topK,
          use_reranker: rerankerEnabled,
          model_profile_id: selectedModelProfileId,
          query_mode: queryMode,
          document_scope: documentScope,
          require_active_only: requireActiveOnly,
        }),
      });

      const data = await res.json();

      // Add assistant message with quality metrics if available
      const assistantMessage = {
        id: uuidv4(),
        role: 'assistant',
        content: data.answer,
        sources: data.sources || [],
        timestamp: new Date().toISOString(),
        searchTime: data.search_time,
        genTime: data.generation_time,
        totalTime: data.total_time,
        rerankTime: data.rerank_time,
        rerankerUsed: data.reranker_used,
        modelProfileId: data.model_profile_id || selectedModelProfileId,
        modelProfileLabel: modelProfiles.find((item) => item.id === (data.model_profile_id || selectedModelProfileId))?.label,
        queryMode: data.query_mode || queryMode,
        rewriteUsed: data.rewrite_used,
        rewriteStatus: data.rewrite_status,
        rewriteFallbackReason: data.rewrite_fallback_reason,
        rewriteStrategy: data.rewrite_strategy,
        rewriteQualityPassed: data.rewrite_quality_passed,
        rewrittenQueries: data.rewritten_queries || [],
        retrievalQuerySet: data.retrieval_query_set || [],
        retrievalQueryPlan: data.retrieval_query_plan || [],
        rewriteContext: data.rewrite_context || null,
        rewriteDebug: data.rewrite_debug || null,
        retrievalDebug: data.retrieval_debug || null,
        perSubqueryHits: data.per_subquery_hits || [],
        retrievalUnionCount: data.retrieval_union_count,
        fallbackUsed: data.fallback_used,
        fallbackError: data.fallback_error,
        collection: data.collection,
        query: content.trim(), // Store query for quality details
        // Quality metrics from backend
        qualityGrade: data.quality_grade,
        qualityScore: data.quality_score,
        bertScore: data.bert_score,
        hallucinationScore: data.hallucination_score,
        factualityScore: data.factuality_score,
        contextRelevance: data.context_relevance,
        qualityFeedback: data.quality_feedback,
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Refresh sessions
      fetchSessions();

    } catch (error) {
      console.error('Chat error:', error);

      const errorMessage = {
        id: uuidv4(),
        role: 'assistant',
        content: '❌ Có lỗi xảy ra khi xử lý câu hỏi. Vui lòng thử lại.',
        sources: [],
        timestamp: new Date().toISOString(),
        isError: true,
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const runLegalSearch = async (content) => {
    if (!content.trim() || isLoading) return;
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: content.trim(),
          user_id: userId,
          session_id: sessionId,
          search_mode: 'legal',
          top_k: topK,
          use_reranker: rerankerEnabled,
          model_profile_id: selectedModelProfileId,
          query_mode: queryMode,
          require_active_only: requireActiveOnly,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Search failed');
      }
      setSearchRuns(prev => [
        {
          id: uuidv4(),
          query: content.trim(),
          timestamp: new Date().toISOString(),
          results: data.results || [],
          trace: {
            rewrite_used: data.rewrite_used,
            rewrite_status: data.rewrite_status,
            rewrite_fallback_reason: data.rewrite_fallback_reason,
            rewrite_strategy: data.rewrite_strategy,
            rewrite_quality_passed: data.rewrite_quality_passed,
            rewritten_queries: data.rewritten_queries || [],
            retrieval_query_set: data.retrieval_query_set || [],
            rewrite_context: data.rewrite_context || null,
            rewrite_debug: data.rewrite_debug || null,
            per_subquery_hits: data.per_subquery_hits || [],
            retrieval_union_count: data.retrieval_union_count,
            search_time_ms: Number(data.search_time || 0) * 1000,
            rerank_time_ms: Number(data.rerank_time || 0) * 1000,
          },
          modelProfileId: data.model_profile_id || selectedModelProfileId,
          queryMode: data.query_mode || queryMode,
          collection: data.collection,
        },
        ...prev,
      ]);
    } catch (error) {
      console.error('Shared legal search error:', error);
      setToast({
        type: 'error',
        message: `❌ Search thất bại: ${error.message}`,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const uploadFile = async (file, modeOverride = uploadMode) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', userId);
    formData.append('session_id', sessionId);
    formData.append('content_mode', modeOverride);
    formData.append('ocr_provider', 'default');

    setUploadProgress({ filename: file.name, status: 'queued', mode: modeOverride });

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (data.success) {
        setUploadProgress({
          filename: data.filename,
          docId: data.doc_id,
          status: data.status || 'queued',
          mode: modeOverride,
          chunkCount: data.chunk_count || 0,
        });
        await fetchDocuments();
        const finalStatus = await pollDocumentStatus(data.doc_id);
        if (finalStatus?.status === 'ready') {
          setToast({
            type: 'success',
            message: `✅ "${data.filename}" đã sẵn sàng (${finalStatus.chunk_count || 0} đoạn)`,
          });
        } else if (finalStatus?.status === 'needs_ocr') {
          setToast({
            type: 'info',
            message: `ℹ️ "${data.filename}" cần OCR. Hãy tải lại với chế độ OCR.`,
          });
        } else if (finalStatus?.status === 'failed') {
          throw new Error(finalStatus.error_message || 'Ingest failed');
        } else {
          setToast({
            type: 'success',
            message: `✅ "${data.filename}" đã được đưa vào hàng đợi xử lý`,
          });
        }
      } else {
        throw new Error(data.message || data.detail || 'Upload failed');
      }
    } catch (error) {
      console.error('Upload error:', error);
      setToast({
        type: 'error',
        message: `❌ Không thể tải lên file: ${error.message}`,
      });
    } finally {
      setUploadProgress(null);
    }
  };

  const deleteDocument = async (docId) => {
    try {
      const res = await fetch(`${API_BASE}/documents/${docId}?user_id=${userId}`, {
        method: 'DELETE',
      });

      if (res.ok) {
        setToast({
          type: 'success',
          message: '✅ Đã xóa tài liệu',
        });
        fetchDocuments();
      }
    } catch (error) {
      console.error('Delete error:', error);
      setToast({
        type: 'error',
        message: '❌ Không thể xóa tài liệu',
      });
    }
  };

  const exportAnswer = (message) => {
    // Create text content
    const content = `
# Câu hỏi pháp luật

${messages.find(m => messages.indexOf(m) === messages.indexOf(message) - 1)?.content || ''}

# Câu trả lời

${message.content}

# Nguồn tham khảo

${message.sources?.map((s, i) => `${i + 1}. ${s.label} ${s.detail ? `- ${s.detail}` : ''}`).join('\n') || 'Không có'}

---
Xuất từ Hệ thống Tư vấn Pháp luật AI
${new Date().toLocaleString('vi-VN')}
    `.trim();

    // Download as text file
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `legal-answer-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);

    setToast({
      type: 'success',
      message: '📥 Đã lưu câu trả lời',
    });
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setToast({
        type: 'success',
        message: '📋 Đã sao chép vào clipboard',
      });
    } catch (error) {
      setToast({
        type: 'error',
        message: '❌ Không thể sao chép',
      });
    }
  };

  const regenerateAnswer = async (messageIndex) => {
    // Find the user message before this assistant message
    const userMessage = messages[messageIndex - 1];
    if (userMessage?.role === 'user') {
      // Remove the assistant message
      setMessages(prev => prev.slice(0, messageIndex));
      // Resend the user message
      await sendMessage(userMessage.content);
    }
  };

  const scoreAnswer = async (message) => {
    // Find the user question before this answer
    const messageIndex = messages.findIndex(m => m.id === message.id);
    const userMessage = messageIndex > 0 ? messages[messageIndex - 1] : null;

    // Use stored query if available, or try to find user message
    const query = message.query || (userMessage?.role === 'user' ? userMessage.content : '');

    if (!query) {
      setToast({
        type: 'error',
        message: '❌ Không tìm thấy câu hỏi để đánh giá',
      });
      return null;
    }

    try {
      // Build contexts from sources
      const contexts = (message.sources || []).map(s => ({
        content: s.content_preview || '',
        ...s,
      }));

      const res = await fetch(`${API_BASE}/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          answer: message.content,
          contexts: contexts,
          top_k: 5,
        }),
      });

      const data = await res.json();

      // Return full metrics for modal display
      const fullMetrics = {
        overall_score: data.overall_score,
        grade: data.grade,
        feedback: data.feedback,
        bert_score: data.details?.query_answer_score || 0,
        hallucination_score: 1 - (data.details?.extractive_score || 0), // Invert for display
        factuality_score: data.details?.answer_context_score || 0,
        context_relevance: data.details?.query_answer_score || 0,
        details: data.details,
      };

      // Show feedback toast
      setToast({
        type: data.grade === 'A' || data.grade === 'B' ? 'success' : 'info',
        message: `📊 Điểm: ${data.grade} (${(data.overall_score * 100).toFixed(0)}%)`,
      });

      return fullMetrics;
    } catch (error) {
      console.error('Score error:', error);
      setToast({
        type: 'error',
        message: '❌ Không thể chấm điểm câu trả lời',
      });
      return null;
    }
  };

  return (
    <div className={`h-screen flex ${darkMode ? 'dark' : ''}`}>
      {/* Sidebar */}
      <Sidebar
        darkMode={darkMode}
        setDarkMode={setDarkMode}
        sessions={sessions}
        currentSessionId={sessionId}
        onSelectSession={loadSession}
        onNewChat={startNewChat}
        documents={documents}
        onDeleteDocument={deleteDocument}
        onUploadFile={uploadFile}
        uploadProgress={uploadProgress}
        uploadMode={uploadMode}
        setUploadMode={setUploadMode}
        systemStatus={systemStatus}
        searchMode={searchMode}
        setSearchMode={setSearchMode}
        documentScope={documentScope}
        setDocumentScope={setDocumentScope}
        rerankerEnabled={rerankerEnabled}
        setRerankerEnabled={updateRerankerSettings}
        currentView={currentView}
        setCurrentView={setCurrentView}
      />

      {/* Main Content Area */}
      {currentView === 'chat' ? (
        isSharedLegalMode ? (
          <LegalLabArea
            messages={messages}
            isLoading={isLoading}
            onSendChat={sendMessage}
            onRunSearch={runLegalSearch}
            onExport={exportAnswer}
            onCopy={copyToClipboard}
            onRegenerate={regenerateAnswer}
            onScore={scoreAnswer}
            systemStatus={systemStatus}
            modelProfiles={modelProfiles}
            selectedModelProfileId={selectedModelProfileId}
            setSelectedModelProfileId={setSelectedModelProfileId}
            queryMode={queryMode}
            setQueryMode={setQueryMode}
            labTab={labTab}
            setLabTab={setLabTab}
            rerankerEnabled={rerankerEnabled}
            requireActiveOnly={requireActiveOnly}
            setRequireActiveOnly={setRequireActiveOnly}
            topK={topK}
            setTopK={setTopK}
            searchRuns={searchRuns}
          />
        ) : (
          <ChatArea
            messages={messages}
            isLoading={isLoading}
            onSendMessage={sendMessage}
            onExport={exportAnswer}
            onCopy={copyToClipboard}
            onRegenerate={regenerateAnswer}
            onUploadFile={uploadFile}
            systemStatus={systemStatus}
            onScore={scoreAnswer}
          />
        )
      ) : currentView === 'ocr' ? (
        <LegalOCR />
      ) : (
        <QualityDashboard />
      )}

      {/* Toast Notifications */}
      {toast && (
        <Toast
          type={toast.type}
          message={toast.message}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  );
}

export default App;
