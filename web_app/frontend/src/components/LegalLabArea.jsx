import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Bot, Loader2, Search, Send, Sparkles } from 'lucide-react';
import clsx from 'clsx';
import MessageBubble from './MessageBubble';

function SearchResultCard({ item, index }) {
  return (
    <div className="rounded-2xl border border-primary-200 bg-white px-4 py-4 shadow-sm dark:border-primary-700 dark:bg-primary-900/70">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-wide text-primary-500 dark:text-primary-400">Ket qua {index + 1}</div>
          <div className="mt-1 text-sm font-semibold text-primary-900 dark:text-white">{item.citation_full || item.label || item.doc_id || 'Van ban phap ly'}</div>
          <div className="mt-1 text-xs text-primary-500 dark:text-primary-400">{item.doc_id || '-'} {item.legal_path ? `| ${item.legal_path}` : ''} {item.chunk_type ? `| ${item.chunk_type}` : ''}</div>
        </div>
        <div className="rounded-full bg-primary-100 px-3 py-1 text-xs font-semibold text-primary-700 dark:bg-primary-800 dark:text-primary-200">{Number(item.score || 0).toFixed(4)}</div>
      </div>
      {item.content_preview && (
        <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-primary-700 dark:text-primary-200">{item.content_preview}</p>
      )}
    </div>
  );
}

function RewriteTrace({ trace }) {
  if (!trace) return null;
  const rewriteContext = trace.rewrite_context || null;
  const compactQueries = rewriteContext?.rewrite_queries_compact || [];
  const queryFacets = rewriteContext?.query_facets || {};
  const rewriteDebug = trace.rewrite_debug || null;
  const conditions = rewriteContext?.conditions || rewriteDebug?.conditions || [];
  const clauseUnits = rewriteContext?.clause_units || rewriteDebug?.clause_units || [];
  const namedLawHints = rewriteContext?.named_law_hints || rewriteDebug?.named_law_hints || [];
  const questionUnits = rewriteContext?.question_units || rewriteDebug?.question_units || [];
  const retrievalQueryPlan = trace.retrieval_query_plan || rewriteDebug?.retrieval_query_plan || [];
  return (
    <div className="rounded-2xl border border-primary-200 bg-primary-50 px-4 py-4 dark:border-primary-700 dark:bg-primary-900/60">
      <div className="text-xs uppercase tracking-wide text-primary-500 dark:text-primary-400">Rewrite Trace</div>
      <div className="mt-2 flex flex-wrap gap-2 text-xs text-primary-700 dark:text-primary-200">
        <span className="rounded-full bg-white px-3 py-1 dark:bg-primary-800">status: {trace.rewrite_status || 'disabled'}</span>
        <span className="rounded-full bg-white px-3 py-1 dark:bg-primary-800">used: {String(!!trace.rewrite_used)}</span>
        <span className="rounded-full bg-white px-3 py-1 dark:bg-primary-800">quality: {String(!!trace.rewrite_quality_passed)}</span>
        <span className="rounded-full bg-white px-3 py-1 dark:bg-primary-800">union: {trace.retrieval_union_count || 0}</span>
        <span className="rounded-full bg-white px-3 py-1 dark:bg-primary-800">search_ms: {Number(trace.search_time_ms || 0).toFixed(0)}</span>
        <span className="rounded-full bg-white px-3 py-1 dark:bg-primary-800">rerank_ms: {Number(trace.rerank_time_ms || 0).toFixed(0)}</span>
      </div>
      {trace.rewrite_strategy ? (
        <div className="mt-2 text-sm text-primary-600 dark:text-primary-300">strategy: {trace.rewrite_strategy}</div>
      ) : null}
      {trace.rewrite_fallback_reason && (
        <div className="mt-3 text-sm text-amber-700 dark:text-amber-300">fallback: {trace.rewrite_fallback_reason}</div>
      )}
      {rewriteContext?.original_query && (
        <div className="mt-3 rounded-xl border border-primary-200 bg-white px-3 py-3 dark:border-primary-700 dark:bg-primary-800">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-primary-500 dark:text-primary-400">Original vs Rewrite</div>
          <div className="mt-2 text-xs text-primary-700 dark:text-primary-200">
            <div className="font-semibold">Original Query</div>
            <div className="mt-1 rounded-lg bg-primary-50 px-2.5 py-2 dark:bg-primary-900/70">{rewriteContext.original_query}</div>
          </div>
          <div className="mt-3 text-xs text-primary-700 dark:text-primary-200">
            <div className="font-semibold">Rewrite Queries (compact)</div>
            {!compactQueries.length ? (
              <div className="mt-1 rounded-lg bg-primary-50 px-2.5 py-2 dark:bg-primary-900/70">&lt;none&gt;</div>
            ) : (
              <div className="mt-2 flex flex-wrap gap-2">
                {compactQueries.map((query, idx) => (
                  <span key={`${idx}-${query}`} className="rounded-xl bg-primary-100 px-2.5 py-1 dark:bg-primary-900/70">
                    {query}
                  </span>
                ))}
              </div>
            )}
          </div>
          {(queryFacets?.actors?.length || queryFacets?.issues?.length || queryFacets?.focus) && (
            <div className="mt-3 text-xs text-primary-700 dark:text-primary-200">
              <div className="font-semibold">Query Facets</div>
              <div className="mt-1 rounded-lg bg-primary-50 px-2.5 py-2 dark:bg-primary-900/70">
                <div>main act: {queryFacets.main_act || rewriteContext?.main_act || '&lt;none&gt;'}</div>
                <div>actors: {(queryFacets.actors || []).join(', ') || '&lt;none&gt;'}</div>
                <div className="mt-1">issues: {(queryFacets.issues || []).join(', ') || '&lt;none&gt;'}</div>
                <div className="mt-1">focus: {queryFacets.focus || '&lt;none&gt;'}</div>
              </div>
            </div>
          )}
          {!!conditions.length && (
            <div className="mt-3 text-xs text-primary-700 dark:text-primary-200">
              <div className="font-semibold">Conditions</div>
              <div className="mt-1 rounded-lg bg-primary-50 px-2.5 py-2 dark:bg-primary-900/70">
                {conditions.map((item, idx) => (
                  <div key={`${idx}-${item.value || item.source_span || idx}`}>
                    {(item.kind || 'khac')}: {item.value || item.source_span || '&lt;none&gt;'}
                  </div>
                ))}
              </div>
            </div>
          )}
          {!!namedLawHints.length && (
            <div className="mt-3 text-xs text-primary-700 dark:text-primary-200">
              <div className="font-semibold">Named Law Hints</div>
              <div className="mt-1 rounded-lg bg-primary-50 px-2.5 py-2 dark:bg-primary-900/70">
                {namedLawHints.join(', ')}
              </div>
            </div>
          )}
          {!!questionUnits.length && (
            <div className="mt-3 text-xs text-primary-700 dark:text-primary-200">
              <div className="font-semibold">Question Units</div>
              <div className="mt-1 rounded-lg bg-primary-50 px-2.5 py-2 dark:bg-primary-900/70 space-y-1">
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
            <div className="mt-3 text-xs text-primary-700 dark:text-primary-200">
              <div className="font-semibold">Clause Units</div>
              <div className="mt-1 rounded-lg bg-primary-50 px-2.5 py-2 dark:bg-primary-900/70 space-y-1">
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
        </div>
      )}
      {!!retrievalQueryPlan.length && (
        <div className="mt-3">
          <div className="text-xs font-semibold uppercase tracking-wide text-primary-500 dark:text-primary-400">Retrieval Query Plan</div>
          <div className="mt-2 space-y-2">
            {retrievalQueryPlan.map((item, idx) => (
              <div key={`${idx}-${item.query || idx}`} className="rounded-xl bg-white px-3 py-2 text-xs text-primary-700 dark:bg-primary-800 dark:text-primary-200">
                <div className="font-semibold">{item.label || item.source_question_unit || `query_${idx + 1}`}</div>
                <div className="mt-1">{item.query || '&lt;none&gt;'}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      {!!trace.retrieval_query_set?.length && (
        <div className="mt-3">
          <div className="text-xs font-semibold uppercase tracking-wide text-primary-500 dark:text-primary-400">Retrieval Query Set</div>
          <div className="mt-2 flex flex-wrap gap-2">
            {trace.retrieval_query_set.map((query, idx) => (
              <span key={`${idx}-${query}`} className="rounded-xl bg-white px-3 py-1 text-xs text-primary-700 dark:bg-primary-800 dark:text-primary-200">{query}</span>
            ))}
          </div>
        </div>
      )}
      {!!rewriteDebug?.rejection_reasons?.length && (
        <div className="mt-3 text-sm text-amber-700 dark:text-amber-300">rejection: {rewriteDebug.rejection_reasons.join(', ')}</div>
      )}
      {!!trace.rewritten_queries?.length && (
        <div className="mt-3">
          <div className="text-xs font-semibold uppercase tracking-wide text-primary-500 dark:text-primary-400">Queries</div>
          <div className="mt-2 flex flex-wrap gap-2">
            {trace.rewritten_queries.map((query, idx) => (
              <span key={`${idx}-${query}`} className="rounded-xl bg-white px-3 py-1 text-xs text-primary-700 dark:bg-primary-800 dark:text-primary-200">{query}</span>
            ))}
          </div>
        </div>
      )}
      {!!trace.per_subquery_hits?.length && (
        <div className="mt-3 space-y-1 text-xs text-primary-600 dark:text-primary-300">
          {trace.per_subquery_hits.map((row, idx) => (
            <div key={`${idx}-${row.query}`} className="rounded-xl bg-white px-3 py-2 dark:bg-primary-800">
              <span className="font-semibold">{row.hits}</span> hits - {row.query}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ControlChip({ active, onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        'rounded-full border px-4 py-2 text-sm font-medium transition-colors',
        active
          ? 'border-indigo-500 bg-indigo-600 text-white'
          : 'border-primary-200 bg-white text-primary-700 hover:bg-primary-50 dark:border-primary-700 dark:bg-primary-900 dark:text-primary-200 dark:hover:bg-primary-800'
      )}
    >
      {children}
    </button>
  );
}

function LegalLabArea({
  messages,
  isLoading,
  onSendChat,
  onRunSearch,
  onExport,
  onCopy,
  onRegenerate,
  onScore,
  systemStatus,
  modelProfiles,
  selectedModelProfileId,
  setSelectedModelProfileId,
  queryMode,
  setQueryMode,
  labTab,
  setLabTab,
  rerankerEnabled,
  requireActiveOnly,
  setRequireActiveOnly,
  topK,
  setTopK,
  searchRuns,
}) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const latestSearch = searchRuns[0] || null;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading, latestSearch]);

  const selectedModel = useMemo(
    () => modelProfiles.find((item) => item.id === selectedModelProfileId) || modelProfiles[0] || null,
    [modelProfiles, selectedModelProfileId]
  );

  const submit = async (event) => {
    event?.preventDefault?.();
    if (!input.trim() || isLoading) return;
    const value = input.trim();
    if (labTab === 'search') {
      await onRunSearch(value);
    } else {
      await onSendChat(value);
    }
    setInput('');
  };

  return (
    <main className="flex-1 flex flex-col bg-white dark:bg-primary-950 h-full">
      <div className="border-b border-primary-200 bg-primary-50 px-6 py-4 dark:border-primary-700 dark:bg-primary-900/60">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold text-primary-900 dark:text-white">Legal Lab</h2>
            <p className="text-sm text-primary-500 dark:text-primary-400">Qdrant cloud: {systemStatus?.collection_resolved || systemStatus?.legal_collection || '-'}</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <ControlChip active={labTab === 'search'} onClick={() => setLabTab('search')}>Search Lab</ControlChip>
            <ControlChip active={labTab === 'chat'} onClick={() => setLabTab('chat')}>Chat Lab</ControlChip>
          </div>
        </div>

        <div className="mt-4 space-y-4">
          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-primary-500 dark:text-primary-400">Model</div>
            <div className="flex flex-wrap gap-2">
              {modelProfiles.map((profile) => (
                <ControlChip
                  key={profile.id}
                  active={selectedModelProfileId === profile.id}
                  onClick={() => setSelectedModelProfileId(profile.id)}
                >
                  {profile.label}
                </ControlChip>
              ))}
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-[1fr_auto_auto]">
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-primary-500 dark:text-primary-400">Query mode</div>
              <div className="flex flex-wrap gap-2">
                <ControlChip active={queryMode === 'original'} onClick={() => setQueryMode('original')}>Original Query</ControlChip>
                <ControlChip active={queryMode === 'rewrite_combine'} onClick={() => setQueryMode('rewrite_combine')}>Rewrite + Combine</ControlChip>
              </div>
            </div>
            <div>
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-primary-500 dark:text-primary-400">Top K</div>
              <input
                type="number"
                min="1"
                max="10"
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value) || 5)}
                className="w-24 rounded-xl border border-primary-200 bg-white px-3 py-2 text-sm dark:border-primary-700 dark:bg-primary-900"
              />
            </div>
            <div className="flex items-end">
              <label className="flex items-center gap-2 text-sm text-primary-700 dark:text-primary-200">
                <input type="checkbox" checked={requireActiveOnly} onChange={(event) => setRequireActiveOnly(event.target.checked)} />
                Active only
              </label>
            </div>
          </div>

          <div className="flex flex-wrap gap-2 text-xs text-primary-500 dark:text-primary-400">
            <span className="rounded-full bg-white px-3 py-1 dark:bg-primary-800">model: {selectedModel?.label || '-'}</span>
            <span className="rounded-full bg-white px-3 py-1 dark:bg-primary-800">query mode: {queryMode === 'rewrite_combine' ? 'Rewrite + Combine' : 'Original Query'}</span>
            <span className="rounded-full bg-white px-3 py-1 dark:bg-primary-800">reranker: {String(!!rerankerEnabled)}</span>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6">
        {labTab === 'search' ? (
          <div className="mx-auto max-w-5xl space-y-4">
            {!latestSearch && !isLoading ? (
              <div className="rounded-3xl border border-dashed border-primary-300 bg-primary-50 px-6 py-10 text-center dark:border-primary-700 dark:bg-primary-900/40">
                <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-600 text-white"><Search className="h-6 w-6" /></div>
                <div className="text-xl font-semibold text-primary-900 dark:text-white">Search Lab</div>
                <p className="mt-2 text-sm text-primary-500 dark:text-primary-400">Thu nghiem retrieval nhanh voi model da chon va 2 che do query.</p>
              </div>
            ) : null}
            {latestSearch ? (
              <>
                <RewriteTrace trace={latestSearch.trace} />
                <div className="grid gap-4">
                  {latestSearch.results.map((item, index) => (
                    <SearchResultCard key={`${item.doc_id || 'doc'}-${index}`} item={item} index={index} />
                  ))}
                </div>
              </>
            ) : null}
          </div>
        ) : (
          <div className="mx-auto max-w-4xl space-y-6">
            {messages.length === 0 ? (
              <div className="rounded-3xl border border-dashed border-primary-300 bg-primary-50 px-6 py-10 text-center dark:border-primary-700 dark:bg-primary-900/40">
                <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-600 text-white"><Bot className="h-6 w-6" /></div>
                <div className="text-xl font-semibold text-primary-900 dark:text-white">Chat Lab</div>
                <p className="mt-2 text-sm text-primary-500 dark:text-primary-400">Dung cung mot model de rewrite va tra loi tren shared legal pipeline.</p>
              </div>
            ) : null}
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
            {isLoading && (
              <div className="flex items-start gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-indigo-100 text-indigo-600 dark:bg-indigo-900/50 dark:text-indigo-400"><Loader2 className="h-4 w-4 animate-spin" /></div>
                <div className="rounded-2xl rounded-tl-none bg-primary-100 px-4 py-3 text-sm text-primary-600 dark:bg-primary-800 dark:text-primary-300">Dang xu ly truy van...</div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className="border-t border-primary-200 bg-primary-50 p-4 dark:border-primary-700 dark:bg-primary-900/60">
        <form onSubmit={submit} className="mx-auto max-w-5xl">
          <div className="rounded-3xl border border-primary-200 bg-white p-3 shadow-sm dark:border-primary-700 dark:bg-primary-900">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder={labTab === 'search' ? 'Nhap truy van de test retrieval...' : 'Nhap cau hoi de chat voi legal pipeline...'}
              className="min-h-[120px] w-full resize-none border-0 bg-transparent text-sm text-primary-900 outline-none placeholder:text-primary-400 dark:text-white"
              disabled={isLoading}
            />
            <div className="mt-3 flex items-center justify-between gap-3">
              <div className="flex flex-wrap gap-2 text-xs text-primary-500 dark:text-primary-400">
                <span className="rounded-full bg-primary-100 px-3 py-1 dark:bg-primary-800">{labTab === 'search' ? 'Search only' : 'Full QA'}</span>
                {queryMode === 'rewrite_combine' ? <span className="rounded-full bg-amber-100 px-3 py-1 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300"><Sparkles className="mr-1 inline h-3 w-3" />rewrite on</span> : null}
              </div>
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                className="inline-flex items-center gap-2 rounded-2xl bg-indigo-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {labTab === 'search' ? <Search className="h-4 w-4" /> : <Send className="h-4 w-4" />}
                {labTab === 'search' ? 'Run Search' : 'Run Chat'}
              </button>
            </div>
          </div>
        </form>
      </div>
    </main>
  );
}

export default LegalLabArea;
