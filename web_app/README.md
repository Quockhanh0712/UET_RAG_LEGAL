# Legal RAG Web App

Giao diện web chuyên nghiệp cho hệ thống tư vấn pháp luật AI.

## 🏗️ Kiến trúc

### Dual-Store Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│              (React + Tailwind CSS)                      │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                        │
│              (Python, Port 8080)                         │
└───────────────────────┬─────────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
┌───────────────────┐   ┌───────────────────────┐
│  🏛️ Kho Luật       │   │  👤 Kho Cá nhân        │
│  (legal_rag_hybrid)│   │  (user_docs_private)  │
│                   │   │                       │
│  ✅ Read-Only      │   │  ✅ Read/Write/Delete  │
│  ✅ Global Search  │   │  ✅ User Isolation     │
│  📊 100k+ điều luật│   │  📄 PDF/DOCX/TXT       │
└───────────────────┘   └───────────────────────┘
```

### Tech Stack

- **Frontend**: React 18, Tailwind CSS, Lucide Icons, Vite
- **Backend**: FastAPI, Uvicorn
- **Vector DB**: Qdrant (Hybrid Search: Dense + BM25)
- **Embedding**: huyydangg/DEk21_hcmute_embedding (768D)
- **LLM**: Ollama qwen2.5:3b

## 🚀 Khởi chạy

### 1. Backend (Port 8080)

```powershell
cd web_app/backend
pip install -r requirements.txt
python main.py
```

### 2. Frontend (Port 3000)

```powershell
cd web_app/frontend
npm install
npm run dev
```

### 3. Truy cập

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8080/docs

## 📡 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/api/status` | Trạng thái hệ thống |
| POST | `/api/chat` | Chat với AI |
| POST | `/api/search` | Tìm kiếm hybrid |
| POST | `/api/upload` | Upload tài liệu |
| GET | `/api/documents` | Danh sách tài liệu |
| DELETE | `/api/documents/{id}` | Xóa tài liệu |
| GET | `/api/history` | Lịch sử chat |
| GET | `/api/sessions` | Danh sách phiên |

## ✨ Tính năng

### 1. Dual-Store Search

- **Kho Luật (legal)**: Tìm trong 100k+ văn bản luật
- **Kho Cá nhân (user)**: Tìm trong file đã upload
- **Kết hợp (hybrid)**: Tìm cả hai, merge kết quả

### 2. Smart File Processing

- Recursive Chunking (Paragraph → Sentence → Word)
- Overlap 12% cho tính liên tục
- Context Injection: `[Nguồn: file.pdf | Trang X]`

### 3. Source Citations

- **Badge xanh (📘)**: Nguồn từ văn bản luật chính thức
- **Badge vàng (📄)**: Nguồn từ file upload

### 4. Chat Features

- Markdown rendering
- Copy/Export/Regenerate
- Performance metrics
- Dark mode

## 🎨 UI Components

```
├── App.jsx              # Main app, state management
├── components/
│   ├── Sidebar.jsx      # History, Documents, Search mode
│   ├── ChatArea.jsx     # Message list, Input
│   ├── MessageBubble.jsx # AI/User message, Citations
│   └── Toast.jsx        # Notifications
```

## 🔧 Configuration

Tạo file `.env` trong thư mục backend:

```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
EMBEDDING_MODEL=huyydangg/DEk21_hcmute_embedding
OLLAMA_MODEL=qwen2.5:3b
TOP_K=10
```

## 📝 Notes

- Đảm bảo Qdrant đang chạy: `docker start qdrant`
- Đảm bảo Ollama đã pull model: `ollama pull qwen2.5:3b`
- Collection `legal_rag_hybrid` phải được tạo trước
