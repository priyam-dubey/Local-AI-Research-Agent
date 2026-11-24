import { FileText, Search } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import DocumentList from '../components/DocumentList'
import UploadDropzone from '../components/UploadDropzone'
import { PageHeader, ProviderBadge, StatusBadge } from '../components/ui'
import { researchHistory } from '../lib/api'
import type { ResearchHistoryItem } from '../lib/types'
import { useApp } from '../state/AppContext'

export default function Dashboard() {
  const { provider, documents, workspaces, workspaceId, status, statusError } = useApp()
  const nav = useNavigate()
  const [history, setHistory] = useState<ResearchHistoryItem[]>([])
  useEffect(() => { researchHistory().then(h => setHistory(h.slice(0, 4))).catch(() => undefined) }, [])
  const ws = workspaces.find(w => w.id === workspaceId)

  return (
    <div className="mx-auto max-w-6xl">
      <PageHeader
        title="Welcome to your AI Research & Document Intelligence Platform ✨"
        subtitle="Research anything, or upload your documents and ask questions."
        right={<div className="flex items-center gap-2"><ProviderBadge provider={provider} />
          <span className={`rounded-full border px-3 py-1 text-xs ${statusError ? 'border-red-500/30 text-red-300' : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'}`}>{statusError ? 'Backend offline' : status ? 'Ready to help!' : 'Connecting…'}</span></div>}
      />
      <div className="grid gap-4 md:grid-cols-2">
        <section className="panel flex flex-col gap-3 p-5">
          <div className="flex items-center gap-3"><Search className="h-8 w-8 text-accent-soft" /><h2 className="text-xl font-semibold text-white">AI Research Agent</h2></div>
          <p className="text-sm text-slate-400">Research any topic with our multi-agent system (Planner → Researcher → Writer → Critic).</p>
          <button className="btn-primary w-fit" onClick={() => nav('/research')}>Start Research →</button>
        </section>
        <section className="panel flex flex-col gap-3 p-5">
          <div className="flex items-center gap-3"><FileText className="h-8 w-8 text-accent-soft" /><h2 className="text-xl font-semibold text-white">Document Intelligence</h2></div>
          <p className="text-sm text-slate-400">Upload your documents and get grounded answers with page citations, suggested questions, and comparisons.</p>
          <button className="btn-primary w-fit" onClick={() => nav('/assistant')}>Open Assistant →</button>
        </section>
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <section className="panel p-5">
          <h2 className="mb-3 font-semibold text-white">Upload Documents <span className="text-xs font-normal text-slate-500">· workspace “{ws?.name ?? 'My Documents'}”</span></h2>
          <UploadDropzone />
        </section>
        <section className="panel p-5">
          <div className="mb-2 flex items-center justify-between"><h2 className="font-semibold text-white">Recent documents</h2><Link className="text-xs text-accent-soft hover:underline" to="/documents">View all</Link></div>
          <DocumentList docs={documents.slice(0, 5)} empty="Nothing uploaded yet." />
        </section>
      </div>
      <section className="panel mt-4 p-5">
        <div className="mb-2 flex items-center justify-between"><h2 className="font-semibold text-white">Recent research sessions</h2><Link className="text-xs text-accent-soft hover:underline" to="/history">View all</Link></div>
        {history.length === 0 ? <p className="text-sm text-slate-500">No research yet.</p> : (
          <ul className="divide-y divide-white/5">{history.map(h => (
            <li key={h.id}><Link to={`/history/${h.id}`} className="flex items-center justify-between gap-3 rounded-lg px-2 py-2.5 hover:bg-white/5">
              <span className="truncate text-sm text-slate-200">{h.query}</span>
              <span className="shrink-0 text-xs text-slate-500">{h.provider} · {h.status === 'complete' ? <StatusBadge status="ready" /> : <StatusBadge status="failed" />}</span>
            </Link></li>))}</ul>)}
      </section>
    </div>
  )
}
