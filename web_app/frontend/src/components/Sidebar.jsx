import React, { useState, useRef, useCallback } from 'react';
import {
  MessageSquarePlus,
  History,
  FileText,
  Trash2,
  Upload,
  Moon,
  Sun,
  Database,
  Bot,
  FileUp,
  ChevronDown,
  Settings,
  Scale,
  Users,
  Combine,
  Sparkles,
  ScanEye,
  ToggleLeft,
  ToggleRight,
  BarChart3,
} from 'lucide-react';
import clsx from 'clsx';

function Sidebar({
  darkMode,
  setDarkMode,
  sessions,
  currentSessionId,
  onSelectSession,
  onNewChat,
  documents,
  onDeleteDocument,
  onUploadFile,
  uploadProgress,
  uploadMode,
  setUploadMode,
  systemStatus,
  searchMode,
  setSearchMode,
  rerankerEnabled,
  setRerankerEnabled,
  currentView,
  setCurrentView,
}) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [showHistory, setShowHistory] = useState(true);
  const [showDocuments, setShowDocuments] = useState(true);
  const fileInputRef = useRef(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragOver(false);

    const files = Array.from(e.dataTransfer.files);
    files.forEach(file => {
      if (file.type === 'application/pdf' ||
        file.name.endsWith('.docx') ||
        file.name.endsWith('.txt')) {
        onUploadFile(file, uploadMode);
      }
    });
  }, [onUploadFile, uploadMode]);

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    files.forEach(file => onUploadFile(file, uploadMode));
    e.target.value = '';
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const searchModes = [
    { value: 'hybrid', label: 'Kết hợp', icon: Combine, color: 'text-purple-500' },
    { value: 'legal', label: 'Văn bản luật', icon: Scale, color: 'text-blue-500' },
    { value: 'user', label: 'Tài liệu của tôi', icon: Users, color: 'text-amber-500' },
  ];

  return (
    <aside className="w-72 flex-shrink-0 bg-primary-50 dark:bg-primary-900 border-r border-primary-200 dark:border-primary-700 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-primary-200 dark:border-primary-700">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-xl flex items-center justify-center">
            <Scale className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-primary-900 dark:text-white">UET Legal Assistant</h1>
            <p className="text-xs text-primary-500 dark:text-primary-400">Tư vấn pháp luật</p>
          </div>
        </div>

        {/* New Chat Button */}
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-medium transition-colors"
        >
          <MessageSquarePlus className="w-4 h-4" />
          Cuộc trò chuyện mới
        </button>
      </div>

      {/* View Selector */}
      <div className="p-4 border-b border-primary-200 dark:border-primary-700">
        <div className="text-xs font-medium text-primary-500 dark:text-primary-400 mb-2 uppercase tracking-wide">
          Chế độ xem
        </div>
        <div className="space-y-1">
          <button
            onClick={() => setCurrentView('chat')}
            className={clsx(
              "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors",
              currentView === 'chat'
                ? "bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300"
                : "text-primary-600 dark:text-primary-400 hover:bg-primary-100 dark:hover:bg-primary-800"
            )}
          >
            <MessageSquarePlus className="w-4 h-4" />
            Chat với AI
          </button>
          <button
            onClick={() => setCurrentView('ocr')}
            className={clsx(
              "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors",
              currentView === 'ocr'
                ? "bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300"
                : "text-primary-600 dark:text-primary-400 hover:bg-primary-100 dark:hover:bg-primary-800"
            )}
          >
            <ScanEye className="w-4 h-4" />
            OCR Preview
          </button>
          <button
            onClick={() => setCurrentView('dashboard')}
            className={clsx(
              "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors",
              currentView === 'dashboard'
                ? "bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300"
                : "text-primary-600 dark:text-primary-400 hover:bg-primary-100 dark:hover:bg-primary-800"
            )}
          >
            <BarChart3 className="w-4 h-4" />
            Quality Dashboard
          </button>
        </div>
      </div>

      {/* Search Mode Selector - Only show in Chat view */}
      {currentView === 'chat' && (
        <div className="p-4 border-b border-primary-200 dark:border-primary-700">
          <div className="text-xs font-medium text-primary-500 dark:text-primary-400 mb-2 uppercase tracking-wide">
            Chế độ tìm kiếm
          </div>
          <div className="space-y-1">
            {searchModes.map(mode => (
              <button
                key={mode.value}
                onClick={() => setSearchMode(mode.value)}
                className={clsx(
                  "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors",
                  searchMode === mode.value
                    ? "bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300"
                    : "text-primary-600 dark:text-primary-400 hover:bg-primary-100 dark:hover:bg-primary-800"
                )}
              >
                <mode.icon className={clsx("w-4 h-4", mode.color)} />
                {mode.label}
              </button>
            ))}
          </div>

          {/* Vietnamese Reranker Toggle */}
          <div className="mt-3 pt-3 border-t border-primary-200 dark:border-primary-600">
            <button
              onClick={() => setRerankerEnabled(!rerankerEnabled)}
              className={clsx(
                "w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-colors",
                rerankerEnabled
                  ? "bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300"
                  : "text-primary-600 dark:text-primary-400 hover:bg-primary-100 dark:hover:bg-primary-800"
              )}
            >
              <span className="flex items-center gap-2">
                <Sparkles className={clsx("w-4 h-4", rerankerEnabled ? "text-emerald-500" : "text-primary-400")} />
                Vietnamese Reranker
              </span>
              {rerankerEnabled ? (
                <ToggleRight className="w-5 h-5 text-emerald-500" />
              ) : (
                <ToggleLeft className="w-5 h-5 text-primary-400" />
              )}
            </button>
            <p className="text-xs text-primary-400 dark:text-primary-500 px-3 mt-1">
              {rerankerEnabled ? "✨ Đang sử dụng AI reranking" : "Sắp xếp theo độ tương đồng"}
            </p>
          </div>
        </div>
      )}

      {/* Scrollable Content - Only show in Chat view */}
      {currentView === 'chat' && (
        <div className="flex-1 overflow-y-auto">
          {/* Chat History Section */}
          <div className="p-4">
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="flex items-center justify-between w-full text-xs font-medium text-primary-500 dark:text-primary-400 mb-2 uppercase tracking-wide"
            >
              <span className="flex items-center gap-1.5">
                <History className="w-3.5 h-3.5" />
                Lịch sử trò chuyện
              </span>
              <ChevronDown className={clsx("w-4 h-4 transition-transform", !showHistory && "-rotate-90")} />
            </button>

            {showHistory && (
              <div className="space-y-1">
                {sessions.length === 0 ? (
                  <p className="text-xs text-primary-400 dark:text-primary-500 italic py-2">
                    Chưa có cuộc trò chuyện nào
                  </p>
                ) : (
                  sessions.slice(0, 10).map(session => (
                    <button
                      key={session.id}
                      onClick={() => onSelectSession(session.id)}
                      className={clsx(
                        "w-full text-left px-3 py-2 rounded-lg text-sm truncate transition-colors",
                        session.id === currentSessionId
                          ? "bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300"
                          : "text-primary-600 dark:text-primary-400 hover:bg-primary-100 dark:hover:bg-primary-800"
                      )}
                      title={session.title}
                    >
                      {session.title}
                    </button>
                  ))
                )}
              </div>
            )}
          </div>

          {/* My Documents Section */}
          <div className="p-4 border-t border-primary-200 dark:border-primary-700">
            <button
              onClick={() => setShowDocuments(!showDocuments)}
              className="flex items-center justify-between w-full text-xs font-medium text-primary-500 dark:text-primary-400 mb-2 uppercase tracking-wide"
            >
              <span className="flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5" />
                Tài liệu của tôi
              </span>
              <ChevronDown className={clsx("w-4 h-4 transition-transform", !showDocuments && "-rotate-90")} />
            </button>

            {showDocuments && (
              <>
                {/* Document List */}
                <div className="space-y-1 mb-3">
                  {documents.length === 0 ? (
                    <p className="text-xs text-primary-400 dark:text-primary-500 italic py-2">
                      Chưa có tài liệu nào
                    </p>
                  ) : (
                    documents.map(doc => (
                      <div
                        key={doc.doc_id}
                        className="flex items-center justify-between px-3 py-2 rounded-lg bg-amber-50 dark:bg-amber-900/20 text-sm group"
                      >
                        <div className="flex items-start gap-2 truncate">
                          <FileText className="w-4 h-4 text-amber-600 dark:text-amber-400 flex-shrink-0" />
                          <div className="min-w-0">
                            <span className="block truncate text-primary-700 dark:text-primary-300" title={doc.filename}>
                              {doc.filename}
                            </span>
                            <div className="flex items-center gap-2 text-xs text-primary-500 dark:text-primary-400">
                              <span>{doc.status}</span>
                              <span>{doc.chunk_count || 0} đoạn</span>
                            </div>
                          </div>
                        </div>
                        <button
                          onClick={() => onDeleteDocument(doc.doc_id)}
                          className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-100 dark:hover:bg-red-900/50 rounded text-red-500 transition-all"
                          title="Xóa tài liệu"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ))
                  )}
                </div>

                {/* Upload Progress */}
                {uploadProgress && (
                  <div className="mb-3 p-3 bg-blue-50 dark:bg-blue-900/30 rounded-lg">
                    <div className="flex items-center gap-2 text-sm text-blue-700 dark:text-blue-300">
                      <div className="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full" />
                      <span className="truncate">
                        {uploadProgress.status === 'queued' ? 'Đã vào hàng đợi' : 'Đang xử lý'}: {uploadProgress.filename}
                      </span>
                    </div>
                    <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                      Chế độ: {uploadProgress.mode === 'ocr' ? 'OCR' : 'Text'}
                    </p>
                  </div>
                )}

                <div className="mb-3">
                  <div className="text-xs font-medium text-primary-500 dark:text-primary-400 mb-2 uppercase tracking-wide">
                    Chế độ ingest
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setUploadMode('text');
                      }}
                      className={clsx(
                        "px-3 py-2 rounded-lg text-sm border transition-colors",
                        uploadMode === 'text'
                          ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300"
                          : "border-primary-300 dark:border-primary-600 text-primary-600 dark:text-primary-400"
                      )}
                    >
                      Text
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setUploadMode('ocr');
                      }}
                      className={clsx(
                        "px-3 py-2 rounded-lg text-sm border transition-colors",
                        uploadMode === 'ocr'
                          ? "border-amber-500 bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300"
                          : "border-primary-300 dark:border-primary-600 text-primary-600 dark:text-primary-400"
                      )}
                    >
                      OCR
                    </button>
                  </div>
                </div>

                {/* Upload Area */}
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={clsx(
                    "border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors",
                    isDragOver
                      ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/30"
                      : "border-primary-300 dark:border-primary-600 hover:border-indigo-400 hover:bg-primary-100 dark:hover:bg-primary-800"
                  )}
                >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf,.docx,.txt"
                      onChange={handleFileSelect}
                      className="hidden"
                      multiple
                  />
                  <FileUp className="w-6 h-6 mx-auto text-primary-400 dark:text-primary-500 mb-2" />
                  <p className="text-xs text-primary-500 dark:text-primary-400">
                    Kéo thả file hoặc click để tải lên
                  </p>
                  <p className="text-xs text-primary-400 dark:text-primary-500 mt-1">
                    PDF, DOCX, TXT
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="p-4 border-t border-primary-200 dark:border-primary-700">
        {/* System Status */}
        {systemStatus && (
          <div className="mb-3 text-xs text-primary-500 dark:text-primary-400 space-y-1">
            <div className="flex items-center gap-1.5">
              <Database className="w-3 h-3" />
              <span>{systemStatus.legal_documents?.toLocaleString() || 0} văn bản luật</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Bot className="w-3 h-3" />
              <span>{systemStatus.llm_model || 'AI'}</span>
            </div>
          </div>
        )}

        {/* Dark Mode Toggle */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm text-primary-600 dark:text-primary-400 hover:bg-primary-100 dark:hover:bg-primary-800 transition-colors"
        >
          {darkMode ? (
            <>
              <Sun className="w-4 h-4" />
              Chế độ sáng
            </>
          ) : (
            <>
              <Moon className="w-4 h-4" />
              Chế độ tối
            </>
          )}
        </button>
      </div>
    </aside>
  );
}

export default Sidebar;
