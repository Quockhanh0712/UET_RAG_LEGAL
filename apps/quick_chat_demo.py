"""Lightweight legal retrieval demo UI for fast local testing."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.adapters.chainlit_adapter import execute_chat_query
from src.config.runtime_settings import RuntimeSettings, load_runtime_settings
from src.factories.legal_pipeline_factory import get_legal_pipeline
from src.runtime.qdrant_contract import QdrantContractStatus, validate_qdrant_contract

app = FastAPI(title="Quick Legal Chat Demo", version="1.0.0")
BASE_SETTINGS = load_runtime_settings("chainlit_demo")
BASE_SETTINGS.apply_env()


class DemoChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    mode: Literal["search", "qa"] = "search"
    top_k: int = Field(default=5, ge=1, le=10)
    use_reranker: bool = True
    use_graph: bool = False
    require_active_only: bool = True
    query_rewriter_enabled: bool = False
    query_rewriter_force: bool = False


def _settings_for_request(request: DemoChatRequest) -> RuntimeSettings:
    return replace(
        BASE_SETTINGS,
        query_rewriter_enabled=bool(request.query_rewriter_enabled),
        query_rewriter_force=bool(request.query_rewriter_force),
    )


def _runtime_context(settings: RuntimeSettings):
    pipeline = get_legal_pipeline(settings)
    contract = validate_qdrant_contract(settings, int(pipeline.embedding_model.embedding_dim))
    return pipeline, contract


def _search_only_payload(pipeline, request: DemoChatRequest) -> Dict[str, Any]:
    result = pipeline.search(
        question=request.message,
        top_k=request.top_k,
        filter_conditions=None,
        use_reranker=request.use_reranker,
        include_inactive=not request.require_active_only,
        use_graph_context=request.use_graph,
        graph_hops=1,
        graph_expand_docs=20,
    )
    return {
        "mode": "search",
        "status": "ok",
        "answer": "",
        "citations": list(result.citations or []),
        "metrics": {
            "retrieval_time_ms": float(result.retrieval_time) * 1000.0,
            "generation_time_ms": 0.0,
            "rerank_time_ms": float(result.rerank_time) * 1000.0,
            "rewrite_time_ms": float(result.rewrite_time) * 1000.0,
            "used_reranker": bool(result.used_reranker),
            "rewrite_used": bool(result.rewrite_used),
            "rewrite_status": str(result.rewrite_status),
            "rewrite_fallback_reason": str(result.rewrite_fallback_reason),
            "rewritten_queries": list(result.rewritten_queries),
            "missing_numbers": list(result.missing_numbers),
            "per_subquery_hits": list(result.per_subquery_hits),
            "retrieval_union_count": int(result.retrieval_union_count),
        },
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    try:
        pipeline, contract = await asyncio.to_thread(_runtime_context, BASE_SETTINGS)
        return {
            "ok": True,
            "embedding_model": BASE_SETTINGS.embedding_model,
            "collection": contract.collection_name,
            "collection_status": contract.collection_status,
            "vector_dim": contract.vector_dim,
            "top_k_default": BASE_SETTINGS.chainlit_top_k,
            "reranker_model": BASE_SETTINGS.reranker_model,
            "points_count": int(pipeline.qdrant_store.get_stats().get("points_count", 0) or 0),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/api/chat")
async def chat(request: DemoChatRequest) -> Dict[str, Any]:
    settings = _settings_for_request(request)
    try:
        pipeline, contract = await asyncio.to_thread(_runtime_context, settings)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Runtime contract failed: {exc}") from exc

    if request.mode == "search":
        payload = await asyncio.to_thread(_search_only_payload, pipeline, request)
    else:
        execution = await asyncio.to_thread(
            execute_chat_query,
            pipeline,
            request.message,
            request.top_k,
            request.use_reranker,
            request.require_active_only,
            request.use_graph,
        )
        payload = {
            "mode": "qa",
            "status": execution.status,
            "answer": execution.answer_text,
            "citations": list(execution.citations or []),
            "metrics": {
                "retrieval_time_ms": float(execution.retrieval_time_ms),
                "generation_time_ms": float(execution.generation_time_ms),
                "rerank_time_ms": float(execution.rerank_time_ms),
                "rewrite_time_ms": float(execution.rewrite_time_ms),
                "used_reranker": bool(execution.used_reranker),
                "rewrite_used": bool(execution.rewrite_used),
                "rewrite_status": str(execution.rewrite_status),
                "rewrite_fallback_reason": str(execution.rewrite_fallback_reason),
                "rewritten_queries": list(execution.rewritten_queries),
                "missing_numbers": list(execution.missing_numbers),
                "per_subquery_hits": list(execution.per_subquery_hits),
                "retrieval_union_count": int(execution.retrieval_union_count),
                "fallback_used": bool(execution.fallback_used),
                "fallback_error": str(execution.fallback_error),
            },
        }

    payload["runtime"] = {
        "collection": contract.collection_name,
        "collection_status": contract.collection_status,
        "embedding_model": settings.embedding_model,
        "query_rewriter_enabled": settings.query_rewriter_enabled,
        "query_rewriter_force": settings.query_rewriter_force,
        "reranker_model": settings.reranker_model,
    }
    return payload


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Quick Legal Demo</title>
  <style>
    :root {
      --bg: #f3efe6;
      --panel: rgba(255,255,255,0.82);
      --ink: #182126;
      --muted: #5a6670;
      --line: rgba(24,33,38,0.12);
      --accent: #0c6d5a;
      --accent-2: #b15c2f;
      --shadow: 0 20px 60px rgba(24,33,38,0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(12,109,90,0.16), transparent 34%),
        radial-gradient(circle at bottom right, rgba(177,92,47,0.12), transparent 28%),
        linear-gradient(180deg, #f5f2ea 0%, #ebe4d8 100%);
      min-height: 100vh;
    }
    .wrap {
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 20px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }
    .sidebar {
      padding: 22px;
      align-self: start;
      position: sticky;
      top: 20px;
    }
    .title {
      margin: 0 0 6px;
      font-size: 28px;
      line-height: 1.1;
      letter-spacing: -0.03em;
      font-family: "IBM Plex Serif", Georgia, serif;
    }
    .sub {
      margin: 0 0 20px;
      color: var(--muted);
      font-size: 14px;
    }
    .runtime {
      display: grid;
      gap: 10px;
      margin-bottom: 18px;
      font-size: 13px;
    }
    .runtime div, .metric-row {
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(24,33,38,0.08);
    }
    .label {
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .field {
      margin-bottom: 16px;
    }
    .field:last-child { margin-bottom: 0; }
    select, textarea, input[type="number"] {
      width: 100%;
      border: 1px solid rgba(24,33,38,0.14);
      background: rgba(255,255,255,0.9);
      border-radius: 14px;
      padding: 12px 14px;
      font: inherit;
      color: inherit;
    }
    textarea {
      min-height: 150px;
      resize: vertical;
      line-height: 1.45;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .toggle {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 11px 12px;
      border-radius: 14px;
      border: 1px solid rgba(24,33,38,0.1);
      background: rgba(255,255,255,0.72);
      margin-bottom: 8px;
      font-size: 14px;
    }
    .toggle:last-child { margin-bottom: 0; }
    button {
      width: 100%;
      border: 0;
      border-radius: 16px;
      padding: 14px 16px;
      font: inherit;
      font-weight: 700;
      color: white;
      background: linear-gradient(135deg, var(--accent), #0d7e68);
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    button.secondary {
      background: linear-gradient(135deg, var(--accent-2), #c97843);
      margin-top: 10px;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.55; cursor: wait; transform: none; }
    .main {
      padding: 20px;
      min-height: calc(100vh - 48px);
    }
    .chat-log {
      display: grid;
      gap: 16px;
    }
    .bubble {
      border-radius: 22px;
      padding: 16px 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.7);
    }
    .bubble.user {
      background: linear-gradient(135deg, rgba(12,109,90,0.16), rgba(12,109,90,0.08));
    }
    .bubble.assistant {
      background: rgba(255,255,255,0.92);
    }
    .bubble h3 {
      margin: 0 0 10px;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .answer {
      white-space: pre-wrap;
      line-height: 1.6;
      font-size: 15px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .sources {
      display: grid;
      gap: 12px;
      margin-top: 18px;
    }
    .source {
      padding: 14px;
      border-radius: 16px;
      border: 1px solid rgba(24,33,38,0.1);
      background: rgba(246,242,236,0.92);
    }
    .source-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 13px;
      margin-bottom: 8px;
      color: var(--muted);
    }
    .source-citation {
      font-size: 14px;
      line-height: 1.5;
    }
    .loading {
      display: none;
      margin-top: 12px;
      color: var(--muted);
      font-size: 14px;
    }
    .hint {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    @media (max-width: 900px) {
      .wrap { grid-template-columns: 1fr; }
      .sidebar { position: static; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <aside class="sidebar card">
      <h1 class="title">Quick Legal Demo</h1>
      <p class="sub">Demo nhẹ để test nhanh retrieval, citation và query rewrite mà không cần Chainlit.</p>
      <div id="runtime" class="runtime">
        <div>Loading runtime...</div>
      </div>
      <div class="field">
        <label class="label" for="message">Query</label>
        <textarea id="message">Công an xã xử phạt lỗi không mang bằng lái xe có đúng không?</textarea>
      </div>
      <div class="grid-2 field">
        <div>
          <label class="label" for="mode">Mode</label>
          <select id="mode">
            <option value="search">Search only</option>
            <option value="qa">Full QA</option>
          </select>
        </div>
        <div>
          <label class="label" for="topk">Top K</label>
          <input id="topk" type="number" min="1" max="10" value="5" />
        </div>
      </div>
      <div class="field">
        <div class="toggle"><span>Use reranker</span><input id="useReranker" type="checkbox" checked /></div>
        <div class="toggle"><span>Enable requery</span><input id="useRewrite" type="checkbox" /></div>
        <div class="toggle"><span>Force rewrite</span><input id="forceRewrite" type="checkbox" /></div>
        <div class="toggle"><span>Active docs only</span><input id="activeOnly" type="checkbox" checked /></div>
      </div>
      <button id="sendBtn" type="button">Run Query</button>
      <button id="complexBtn" class="secondary" type="button">Load Complex Rewrite Query</button>
      <div id="loading" class="loading">Running query. Search mode should be much faster than Full QA.</div>
      <p class="hint">Khuyến nghị: dùng <strong>Search only</strong> để test citation/retrieval trước. Chỉ bật <strong>Full QA</strong> khi cần kiểm tra đường LLM.</p>
    </aside>
    <main class="main card">
      <div id="chatLog" class="chat-log"></div>
    </main>
  </div>
  <script>
    const runtimeBox = document.getElementById('runtime');
    const chatLog = document.getElementById('chatLog');
    const loading = document.getElementById('loading');
    const sendBtn = document.getElementById('sendBtn');
    const complexBtn = document.getElementById('complexBtn');

    async function loadHealth() {
      const res = await fetch('/health');
      const data = await res.json();
      if (!data.ok) {
        runtimeBox.innerHTML = `<div><span class="label">Runtime Error</span>${data.error}</div>`;
        return;
      }
      runtimeBox.innerHTML = `
        <div><span class="label">Embedding</span>${data.embedding_model}</div>
        <div><span class="label">Collection</span>${data.collection}</div>
        <div><span class="label">Collection Status</span>${data.collection_status}</div>
        <div><span class="label">Points</span>${data.points_count}</div>
        <div><span class="label">Reranker</span>${data.reranker_model}</div>
      `;
    }

    function sourceCard(source, index) {
      return `
        <article class="source">
          <div class="source-head">
            <span>#${index + 1} | ${source.doc_id || ''}</span>
            <span>score=${Number(source.score || 0).toFixed(4)}</span>
          </div>
          <div class="source-citation">${source.citation_full || '(missing citation_full)'}</div>
        </article>
      `;
    }

    function metric(label, value) {
      return `<div class="metric-row"><span class="label">${label}</span>${value}</div>`;
    }

    function renderResponse(query, data) {
      const citations = Array.isArray(data.citations) ? data.citations : [];
      const metrics = data.metrics || {};
      const answerText = data.mode === 'qa'
        ? (data.answer || '(empty answer)')
        : 'Search only mode. Inspect citations and metrics below.';

      const block = document.createElement('section');
      block.className = 'bubble assistant';
      block.innerHTML = `
        <h3>Assistant</h3>
        <div class="answer">${answerText}</div>
        <div class="metrics">
          ${metric('Mode', data.mode)}
          ${metric('Status', data.status)}
          ${metric('Retrieval ms', Number(metrics.retrieval_time_ms || 0).toFixed(2))}
          ${metric('Generation ms', Number(metrics.generation_time_ms || 0).toFixed(2))}
          ${metric('Rewrite', metrics.rewrite_status || 'disabled')}
          ${metric('Rewrite used', String(Boolean(metrics.rewrite_used)))}
          ${metric('Rewrite queries', (metrics.rewritten_queries || []).length)}
          ${metric('Union count', metrics.retrieval_union_count || 0)}
          ${metric('Reranker', String(Boolean(metrics.used_reranker)))}
          ${metric('Collection', data.runtime?.collection || '')}
        </div>
        <div class="hint"><strong>Rewrite trace:</strong> ${(metrics.rewritten_queries || []).join(' | ') || 'original only'}</div>
        <div class="sources">${citations.map(sourceCard).join('')}</div>
      `;

      const userBlock = document.createElement('section');
      userBlock.className = 'bubble user';
      userBlock.innerHTML = `<h3>User</h3><div class="answer">${query}</div>`;

      chatLog.prepend(block);
      chatLog.prepend(userBlock);
    }

    async function runQuery() {
      const payload = {
        message: document.getElementById('message').value.trim(),
        mode: document.getElementById('mode').value,
        top_k: Number(document.getElementById('topk').value || 5),
        use_reranker: document.getElementById('useReranker').checked,
        use_graph: false,
        require_active_only: document.getElementById('activeOnly').checked,
        query_rewriter_enabled: document.getElementById('useRewrite').checked,
        query_rewriter_force: document.getElementById('forceRewrite').checked,
      };
      if (!payload.message) return;
      loading.style.display = 'block';
      sendBtn.disabled = true;
      complexBtn.disabled = true;
      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Request failed');
        renderResponse(payload.message, data);
      } catch (err) {
        renderResponse(payload.message, {
          mode: payload.mode,
          status: 'error',
          answer: String(err),
          citations: [],
          metrics: {},
          runtime: {},
        });
      } finally {
        loading.style.display = 'none';
        sendBtn.disabled = false;
        complexBtn.disabled = false;
      }
    }

    sendBtn.addEventListener('click', runQuery);
    complexBtn.addEventListener('click', () => {
      document.getElementById('message').value = 'Giám đốc công ty ra quyết định xả thải trái phép nhằm tăng lợi nhuận. Nhân viên kỹ thuật biết rõ nhưng vẫn thực hiện. Hỏi trách nhiệm hình sự và dân sự thuộc về pháp nhân, cá nhân hay cả hai?';
      document.getElementById('useRewrite').checked = true;
      document.getElementById('mode').value = 'search';
    });

    loadHealth();
  </script>
</body>
</html>"""
