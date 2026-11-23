import { FileSpreadsheet, FileText, Trash2 } from 'lucide-react'
import { formatBytes } from '../lib/citations'
import type { DocumentRecord } from '../lib/types'
import { StatusBadge } from './ui'

const ICON: Record<string, string> = { pdf: 'bg-red-500', docx: 'bg-blue-500', txt: 'bg-slate-500', md: 'bg-violet-500', csv: 'bg-emerald-500', url: 'bg-sky-500' }

interface Props {
  docs: DocumentRecord[]
  /** 'single' = click selects one, 'multi' = checkboxes, 'none' = display only */
  mode?: 'single' | 'multi' | 'none'
  selected?: string[]
  onSelect?: (id: string) => void
  onDelete?: (doc: DocumentRecord) => void
  empty?: string
}

export default function DocumentList({ docs, mode = 'none', selected = [], onSelect, onDelete, empty = 'No documents yet.' }: Props) {
  if (!docs.length) return <p className="px-1 py-3 text-sm text-slate-500">{empty}</p>
  return (
    <ul className="divide-y divide-white/5">
      {docs.map(d => {
        const active = selected.includes(d.id)
        const selectable = mode !== 'none' && d.status === 'ready'
        return (
          <li key={d.id} className={`group flex items-center gap-3 rounded-lg px-2 py-2.5 ${active ? 'bg-accent/15' : ''}`}>
            {mode === 'multi' && <input type="checkbox" aria-label={`Select ${d.filename}`} className="h-4 w-4 accent-indigo-500" checked={active} disabled={!selectable} onChange={() => onSelect?.(d.id)} />}
            <button type="button" disabled={!selectable || mode === 'multi'} onClick={() => onSelect?.(d.id)}
              className={`flex min-w-0 flex-1 items-center gap-3 text-left ${selectable && mode === 'single' ? 'cursor-pointer' : 'cursor-default'}`}>
              <span className={`grid h-9 w-8 shrink-0 place-items-center rounded-md text-[9px] font-bold uppercase text-white ${ICON[d.file_type] ?? 'bg-slate-500'}`}>
                {d.file_type === 'csv' ? <FileSpreadsheet className="h-4 w-4" /> : d.file_type === 'txt' ? <FileText className="h-4 w-4" /> : d.file_type}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium text-slate-100" title={d.filename}>{d.filename}</span>
                <span className="flex flex-wrap items-center gap-x-2 text-xs text-slate-400">
                  <span>{formatBytes(d.size_bytes)}</span><span>•</span><StatusBadge status={d.status} error={d.error} />
                </span>
                {d.status === 'failed' && d.error && <span className="mt-0.5 block text-xs text-red-300">{d.error}</span>}
              </span>
            </button>
            {onDelete && (
              <button type="button" onClick={() => onDelete(d)} aria-label={`Remove ${d.filename}`}
                disabled={d.status === 'queued' || d.status === 'processing'}
                className="rounded-md p-1.5 text-slate-500 hover:bg-red-500/10 hover:text-red-300 disabled:opacity-30"><Trash2 className="h-4 w-4" /></button>
            )}
          </li>
        )
      })}
    </ul>
  )
}
