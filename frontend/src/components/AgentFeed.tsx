import { useEffect, useRef } from 'react'
import type { AgentEvent } from '../lib/types'
import Markdown from './Markdown'

const ROLE: Record<string, { label: string; cls: string }> = {
  planner: { label: 'Planner', cls: 'bg-sky-500/20 text-sky-300' },
  researcher: { label: 'Researcher', cls: 'bg-emerald-500/20 text-emerald-300' },
  writer: { label: 'Writer', cls: 'bg-violet-500/20 text-violet-300' },
  critic: { label: 'Critic', cls: 'bg-amber-500/20 text-amber-300' },
  system: { label: 'System', cls: 'bg-slate-500/20 text-slate-300' },
}

export default function AgentFeed({ events }: { events: AgentEvent[] }) {
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [events.length])
  return (
    <ol className="space-y-3" aria-label="Agent activity" aria-live="polite">
      {events.map((e, i) => {
        const r = ROLE[e.role] ?? ROLE.system
        return (
          <li key={i} className="animate-fade-in rounded-xl border border-white/10 bg-ink-900/60 p-3">
            <span className={`mb-1.5 inline-block rounded-full px-2 py-0.5 text-[11px] font-semibold ${r.cls}`}>{r.label}</span>
            {e.role === 'writer' && e.content.length > 600
              ? <details><summary className="cursor-pointer text-sm text-slate-400">Draft report ({e.content.length.toLocaleString()} characters)</summary><Markdown text={e.content} /></details>
              : <Markdown text={e.content} />}
          </li>
        )
      })}
      <div ref={end} />
    </ol>
  )
}
