import React, { useState, useEffect, useRef, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
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
  const [rerankerEnabled, setRerankerEnabled] = useState(true); // Vietnamese Reranker toggle

  // Documents state
  const [documents, setDocuments] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(null);

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

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/documents?user_id=${userId}`);
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    }
  };

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
    fetchSessions();
  };

  const sendMessage = async (content) => {
    if (!content.trim() || isLoading) return;

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
          top_k: 10,
          use_reranker: rerankerEnabled,
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

  const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', userId);
    formData.append('session_id', sessionId);

    setUploadProgress({ filename: file.name, progress: 0 });

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (data.success) {
        setToast({
          type: 'success',
          message: `✅ Đã tải lên "${data.filename}" (${data.chunk_count} đoạn)`,
        });
        fetchDocuments();
      } else {
        throw new Error(data.message || 'Upload failed');
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
        systemStatus={systemStatus}
        searchMode={searchMode}
        setSearchMode={setSearchMode}
        rerankerEnabled={rerankerEnabled}
        setRerankerEnabled={updateRerankerSettings}
        currentView={currentView}
        setCurrentView={setCurrentView}
      />

      {/* Main Content Area */}
      {currentView === 'chat' ? (
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
