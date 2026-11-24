import { Scale } from 'lucide-react'
import { useState } from 'react'
import { chatStream } from '../lib/api'
import type { ChatMessage, DocumentRecord } from '../lib/types'
import { useApp } from '../state/AppContext'
import DocumentList from './DocumentList'
import { AnswerView } from './MessageBubble'
import { ErrorBanner, Notice, Spinner } from './ui'

export default function ComparePanel({ docs }: { docs: DocumentRecord[] }) {
  const { workspaceId, provider, model, providerProblem } = useApp()
  const [picked, setPicked] = useState<string[]>([])
  const [focus, setFocus] = useState('')
  const [stage, setStage] = useState<string | null>(null)
  const [result, setResult] = useState<ChatMessage | null>(null)
  const [error, setError] = useState<string | null>(null)

  const toggle = (id: string) => setPicked(p => (p.includes(id) ? p.filter(x => x !== id) : [...p, id]))
  const ready = docs.filter(d => d.status === 'ready')

  function run() {
    setResult(null); setError(null); setStage('retrieving')
    void chatStream(workspaceId, {
      question: focus.trim() || 'Compare these documents.', provider, model, scope: { mode: 'compare', document_ids: picked },
    }, {
      onStage: setStage,
      onFinal: a => { setStage(null); setResult({ id: a.message_id, role: 'assistant', content: a.answer, sources: a.sources, refused: a.refused, warnings: a.warnings }) },
      onError: e => { setStage(null); setError(e.message) },
    })
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto">
      <p className="text-sm text-slate-400">Pick two or more processed documents. The comparison covers similarities, differences, conflicting findings, methodologies and conclusions, with citations for every document.</p>
      {ready.length < 2 ? <Notice tone="info">Upload at least two documents to compare them.</Notice> : (
        <div className="rounded-xl border border-white/10 bg-white/[0.03] p-2"><DocumentList docs={ready} mode="multi" selected={picked} onSelect={toggle} /></div>
      )}
      <input className="input" placeholder="Optional: what should be compared? (e.g. methodology, datasets, limitations)" value={focus} maxLength={500} onChange={e => setFocus(e.target.value)} />
      <div className="flex items-center gap-3">
        <button className="btn-primary" disabled={picked.length < 2 || picked.length > 5 || !!stage} onClick={run}><Scale className="h-4 w-4" />Compare {picked.length || ''} documents</button>
        {stage && <span className="flex items-center gap-2 text-sm text-slate-300"><Spinner />Comparing…</span>}
      </div>
      {providerProblem && <Notice>{providerProblem}</Notice>}
      {error && <ErrorBanner onClose={() => setError(null)}>{error}</ErrorBanner>}
      {result && <div className="panel animate-fade-in p-4"><AnswerView m={result} /></div>}
    </div>
  )
}
