import { FileText } from 'lucide-react'
import { useState } from 'react'
import { sourceLabel } from '../lib/citations'
import type { Source } from '../lib/types'

export default function SourceList({ sources, highlight }: { sources: Source[]; highlight?: string | null }) {
  const [open, setOpen] = useState<string | null>(null)
  if (!sources.length) return null
  return (
    <div className="mt-3 space-y-1.5" aria-label="Sources">
      <p className="flex items-center gap-1.5 text-xs font-semibold text-slate-300"><FileText className="h-3.5 w-3.5" />Source{sources.length > 1 ? 's' : ''}:</p>
      {sources.map((s, i) => (
        <div key={s.id} id={`src-${s.id}`} className={`rounded-lg border px-2.5 py-1.5 text-xs transition ${highlight === s.id ? 'border-accent bg-accent/15' : 'border-white/10 bg-black/20'}`}>
          <button type="button" className="flex w-full items-center gap-2 text-left" onClick={() => setOpen(open === s.id ? null : s.id)} aria-expanded={open === s.id}>
            <span className="grid h-4 min-w-4 place-items-center rounded bg-accent/30 px-1 text-[10px] font-semibold text-accent-soft">{i + 1}</span>
            <span className="text-accent-soft underline-offset-2 hover:underline">{sourceLabel(s)}</span>
          </button>
          {open === s.id && <p className="mt-1.5 border-l-2 border-white/10 pl-2 italic text-slate-400">“{s.excerpt}{s.excerpt.length >= 300 ? '…' : ''}”</p>}
        </div>
      ))}
    </div>
  )
}
