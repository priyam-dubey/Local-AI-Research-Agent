import { Search } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import AgentFeed from '../components/AgentFeed'
import ReportView from '../components/ReportView'
import { ErrorBanner, Notice, PageHeader, ProviderBadge, Spinner } from '../components/ui'
import { getResearchResult, startResearch, streamResearch } from '../lib/api'
import type { AgentEvent, ResearchResult } from '../lib/types'
import { useApp } from '../state/AppContext'

export default function Research() {
  const { provider, model, providerProblem, workspaces, workspaceId, readyDocuments } = useApp()
  const [query, setQuery] = useState('')
  const [useDocs, setUseDocs] = useState(true)
  const [running, setRunning] = useState(false)
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [result, setResult] = useState<ResearchResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const close = useRef<(() => void) | null>(null)
  useEffect(() => () => close.current?.(), [])

  const wsName = workspaces.find(w => w.id === workspaceId)?.name
  async function run() {
    setError(null); setResult(null); setEvents([]); setRunning(true)
    try {
      const { session_id } = await startResearch({ query: query.trim(), provider, model, workspace_id: useDocs && readyDocuments.length ? workspaceId : undefined })
      close.current = streamResearch(session_id, e => setEvents(ev => [...ev, e]), async status => {
        try {
          const r = await getResearchResult(session_id)
          if (status === 'complete') setResult(r)
          else setError('The research run failed. See the activity log above.')
        } catch (e) { setError((e as Error).message) } finally { setRunning(false) }
      })
    } catch (e) { setError((e as Error).message); setRunning(false) }
  }

  return (
    <div className="mx-auto max-w-4xl">
      <PageHeader title="New Research" subtitle="A multi-agent team plans, researches, writes and critiques a report." right={<ProviderBadge provider={provider} />} />
      <form onSubmit={e => { e.preventDefault(); if (query.trim().length >= 5) void run() }} className="panel space-y-3 p-5">
        <label htmlFor="q" className="text-sm font-medium text-slate-300">Research question</label>
        <textarea id="q" className="input min-h-24 resize-y" maxLength={2000} placeholder="e.g. What are the trade-offs between RAG and fine-tuning for domain-specific assistants?" value={query} onChange={e => setQuery(e.target.value)} />
        {readyDocuments.length > 0 && (
          <label className="flex items-center gap-2 text-sm text-slate-300"><input type="checkbox" className="accent-indigo-500" checked={useDocs} onChange={e => setUseDocs(e.target.checked)} />Also use documents from “{wsName}” as local evidence (web search is used for anything they don’t cover)</label>)}
        {provider === 'gemini' && <Notice tone="info">Gemini is online: your question and the evidence passages are sent to Google.</Notice>}
        {providerProblem && <Notice>{providerProblem}</Notice>}
        <button className="btn-primary" disabled={running || query.trim().length < 5}>{running ? <><Spinner />Researching…</> : <><Search className="h-4 w-4" />Start research</>}</button>
      </form>
      {error && <div className="mt-4"><ErrorBanner onClose={() => setError(null)}>{error}</ErrorBanner></div>}
      {events.length > 0 && <section className="mt-6"><h2 className="mb-3 font-semibold text-white">Agent activity</h2><AgentFeed events={events} /></section>}
      {result && <section className="mt-6"><h2 className="mb-3 font-semibold text-white">Report</h2><ReportView result={result} /></section>}
    </div>
  )
}
