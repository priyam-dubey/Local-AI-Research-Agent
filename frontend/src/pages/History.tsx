import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import ReportView from '../components/ReportView'
import { ErrorBanner, PageHeader, Spinner } from '../components/ui'
import { getResearchResult, researchHistory } from '../lib/api'
import type { ResearchHistoryItem, ResearchResult } from '../lib/types'

export default function History() {
  const { id } = useParams()
  const [items, setItems] = useState<ResearchHistoryItem[] | null>(null)
  const [result, setResult] = useState<ResearchResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { researchHistory().then(setItems).catch(e => setError((e as Error).message)) }, [])
  useEffect(() => {
    setResult(null)
    if (id) getResearchResult(id).then(setResult).catch(e => setError((e as Error).message))
  }, [id])

  return (
    <div className="mx-auto max-w-4xl">
      <PageHeader title="Research History" subtitle="Saved reports from previous sessions." />
      {error && <ErrorBanner onClose={() => setError(null)}>{error}</ErrorBanner>}
      {id ? (
        <div className="space-y-3"><Link to="/history" className="text-sm text-accent-soft hover:underline">← All sessions</Link>
          {result ? <><h2 className="text-lg font-semibold text-white">{result.query}</h2><ReportView result={result} /></> : <Spinner className="h-6 w-6" />}</div>
      ) : items === null ? <Spinner className="h-6 w-6" /> : items.length === 0 ? <p className="text-slate-500">No sessions yet. Start a new research.</p> : (
        <ul className="panel divide-y divide-white/5">
          {items.map(h => (
            <li key={h.id}><Link to={`/history/${h.id}`} className="block px-4 py-3 hover:bg-white/5">
              <p className="truncate text-sm font-medium text-slate-100">{h.query}</p>
              <p className="text-xs text-slate-500">{new Date(h.created_at * 1000).toLocaleString()} · {h.provider === 'ollama' ? 'Local (Ollama)' : 'Gemini'} · {h.status}{h.status === 'complete' ? ` · quality ${(h.quality * 100).toFixed(0)}%` : ''}</p>
            </Link></li>))}
        </ul>)}
    </div>
  )
}
