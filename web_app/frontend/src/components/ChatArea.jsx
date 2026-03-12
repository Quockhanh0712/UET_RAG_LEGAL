import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Paperclip,
  Bot,
  User,
  Scale,
  Loader2,
} from 'lucide-react';
import clsx from 'clsx';
import MessageBubble from './MessageBubble';

function ChatArea({
  messages,
  isLoading,
  onSendMessage,
  onExport,
  onCopy,
  onRegenerate,
  onUploadFile,
  systemStatus,
  onScore,
}) {
  const [input, setInput] = useState('');
  const textareaRef = useRef(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
    }
  }, [input]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    files.forEach(file => onUploadFile(file));
    e.target.value = '';
  };

  const welcomeMessage = messages.length === 0;

  return (
    <main className="flex-1 flex flex-col bg-white dark:bg-primary-950 h-full">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto">
        {welcomeMessage ? (
          <div className="h-full flex flex-col items-center justify-center px-4 py-8">
            <div className="w-16 h-16 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-2xl flex items-center justify-center mb-6 shadow-lg">
              <Scale className="w-8 h-8 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-primary-900 dark:text-white mb-2">
              UET Legal Assistant
            </h2>
            <p className="text-primary-500 dark:text-primary-400 text-center max-w-md mb-8">
              Đặt câu hỏi về pháp luật Việt Nam. Hệ thống sẽ tìm kiếm trong{' '}
              <span className="font-semibold text-blue-600 dark:text-blue-400">
                {systemStatus?.legal_documents?.toLocaleString() || '100,000+'}
              </span>{' '}
              văn bản luật và trả lời dựa trên nguồn chính xác.
            </p>

          </div>
        ) : (
          <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
            {messages.map((message, index) => (
              <MessageBubble
                key={message.id}
                message={message}
                onExport={() => onExport(message)}
                onCopy={() => onCopy(message.content)}
                onRegenerate={() => onRegenerate(index)}
                onScore={onScore}
              />
            ))}

            {/* Loading Indicator */}
            {isLoading && (
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-indigo-100 dark:bg-indigo-900/50 flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
                </div>
                <div className="bg-primary-100 dark:bg-primary-800 rounded-2xl rounded-tl-none px-4 py-3">
                  <div className="flex items-center gap-2 text-primary-500 dark:text-primary-400">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm">Đang phân tích và tìm kiếm...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-primary-200 dark:border-primary-700 bg-primary-50 dark:bg-primary-900 p-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex items-end gap-2 bg-white dark:bg-primary-800 rounded-2xl border border-primary-200 dark:border-primary-700 p-2 shadow-sm">
            {/* Attachment Button */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="p-2 text-primary-400 hover:text-indigo-600 dark:hover:text-indigo-400 hover:bg-primary-100 dark:hover:bg-primary-700 rounded-lg transition-colors"
              title="Tải lên tài liệu"
            >
              <Paperclip className="w-5 h-5" />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.doc,.txt"
              onChange={handleFileSelect}
              className="hidden"
              multiple
            />

            {/* Text Input */}
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Nhập câu hỏi pháp luật..."
              className="flex-1 resize-none bg-transparent border-none outline-none text-primary-900 dark:text-white placeholder-primary-400 dark:placeholder-primary-500 py-2 px-1 max-h-48"
              rows={1}
              disabled={isLoading}
            />

            {/* Send Button */}
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className={clsx(
                "p-2 rounded-lg transition-colors",
                input.trim() && !isLoading
                  ? "bg-indigo-600 hover:bg-indigo-700 text-white"
                  : "bg-primary-100 dark:bg-primary-700 text-primary-400 cursor-not-allowed"
              )}
            >
              <Send className="w-5 h-5" />
            </button>
          </div>

          <p className="text-center text-xs text-primary-400 dark:text-primary-500 mt-2">
            AI có thể mắc sai sót. Hãy kiểm tra các nguồn trích dẫn để đảm bảo chính xác.
          </p>
        </form>
      </div>
    </main>
  );
}

export default ChatArea;
