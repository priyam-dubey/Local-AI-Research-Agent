import { BrainCircuit, CheckCircle2, SearchX, User } from 'lucide-react'
import { useState } from 'react'
import type { ChatMessage } from '../lib/types'
import Markdown from './Markdown'
import SourceList from './SourceList'
import { ErrorBanner, Spinner } from './ui'

const STAGES: Record<string, string> = {
  retrieving: 'Searching your documents…',
  generating: 'Writing an answer from the excerpts…',
  verifying: 'Checking every statement against the sources…',
}

/** Verified answer with grounding badge + sources. Shared by chat and compare views. */
export function AnswerView({ m }: { m: Pick<ChatMessage, 'content' | 'sources' | 'refused' | 'warnings'> }) {
  const [hl, setHl] = useState<string | null>(null)
  const sources = m.sources ?? []
  const cite = (id: string) => {            // markers use the server's ids; chips map to the nth source
    setHl(id)
    document.getElementById(`src-${id}`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    setTimeout(() => setHl(null), 1500)
  }
  return (
    <>
      <Markdown text={m.content} citations onCite={cite} />
      <SourceList sources={sources} highlight={hl} />
      {m.warnings?.map(w => <p key={w} className="mt-2 text-xs text-amber-300">⚠ {w}</p>)}
      <div className="mt-3">
        {m.refused
          ? <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/10 px-2.5 py-1 text-xs text-amber-300"><SearchX className="h-3.5 w-3.5" />Not found in the uploaded documents — nothing was guessed</span>
          : <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-2.5 py-1 text-xs text-emerald-300"><CheckCircle2 className="h-3.5 w-3.5" />Answer based only on the uploaded documents</span>}
      </div>
    </>
  )
}

export default function MessageBubble({ m }: { m: ChatMessage }) {
  if (m.role === 'user') {
    return (
      <div className="flex animate-fade-in justify-end gap-3">
        <div className="max-w-[80%] rounded-2xl bg-accent/25 px-4 py-2.5 text-sm text-white">{m.content}</div>
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-accent/40"><User className="h-4 w-4" /></span>
      </div>
    )
  }
  return (
    <div className="flex animate-fade-in gap-3">
      <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-ink-600"><BrainCircuit className="h-4 w-4 text-accent-soft" /></span>
      <div className="max-w-[85%] rounded-2xl border border-white/10 bg-ink-800 px-4 py-3">
        {m.pending && <p className="flex items-center gap-2 text-sm text-slate-300" aria-live="polite"><Spinner />{STAGES[m.stage ?? 'retrieving'] ?? 'Working…'}</p>}
        {m.error && <ErrorBanner>{m.error}</ErrorBanner>}
        {!m.pending && !m.error && <AnswerView m={m} />}
      </div>
    </div>
  )
}
