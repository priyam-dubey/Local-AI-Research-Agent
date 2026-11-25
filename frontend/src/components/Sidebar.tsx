import { BrainCircuit, FolderOpen, History, MessageSquareText, Search, Settings } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { useApp } from '../state/AppContext'
import ModelSelector from './ModelSelector'

const NAV = [
  { to: '/research', label: 'New Research', icon: Search },
  { to: '/assistant', label: 'Document Assistant', icon: MessageSquareText },
  { to: '/documents', label: 'Uploaded Documents', icon: FolderOpen },
  { to: '/history', label: 'Research History', icon: History },
  { to: '/settings', label: 'Settings', icon: Settings },
]

function Dot({ ok }: { ok: boolean | null }) {
  return <span className={`h-2 w-2 rounded-full ${ok === null ? 'bg-slate-500' : ok ? 'bg-emerald-400' : 'bg-red-400'}`} />
}

export default function Sidebar() {
  const { status, statusError } = useApp()
  return (
    <aside className="flex h-full w-72 shrink-0 flex-col gap-5 overflow-y-auto border-r border-white/10 bg-ink-900/80 p-4">
      <NavLink to="/" className="flex items-center gap-3 px-1 pt-1">
        <BrainCircuit className="h-9 w-9 text-accent-soft" />
        <span className="text-lg font-semibold leading-tight text-white">Local AI<br />Research Agent</span>
      </NavLink>
      <nav className="space-y-1">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to}
            className={({ isActive }) => `flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition ${isActive ? 'bg-accent text-white' : 'text-slate-300 hover:bg-white/5'}`}>
            <Icon className="h-4 w-4" />{label}
          </NavLink>
        ))}
      </nav>
      <section aria-label="Model selection" className="border-t border-white/10 pt-4">
        <h2 className="mb-2 text-sm font-medium text-slate-300">Model Selection</h2>
        <ModelSelector />
      </section>
      <section aria-label="System status" className="mt-auto space-y-1.5 rounded-xl border border-white/10 bg-white/5 p-3 text-xs">
        <div className="mb-1 flex items-center gap-2 text-sm font-medium text-slate-200"><Dot ok={statusError ? false : status ? true : null} />System Status</div>
        {statusError ? <p className="text-red-300">Backend unreachable</p> : <>
          <div className="flex items-center justify-between text-slate-400"><span>Ollama</span><span className={status?.ollama.available ? 'text-emerald-400' : 'text-red-400'}>{status ? (status.ollama.available ? 'Online' : 'Offline') : '…'}</span></div>
          <div className="flex items-center justify-between text-slate-400"><span>FAISS</span><span className="text-emerald-400">{status?.vector_store.ready ? 'Ready' : '…'}</span></div>
          <div className="flex items-center justify-between text-slate-400"><span>Embeddings</span>
            {status?.embeddings.error
              ? <span className="truncate pl-2 text-red-400" title={status.embeddings.error}>Error</span>
              : <span className="truncate pl-2 text-slate-300" title={status?.embeddings.model}>{!status ? '…' : status.embeddings.backend === 'hashing' ? 'Test mode' : status.embeddings.loaded ? 'Ready' : 'Loading…'}</span>}
          </div>
          {status?.embeddings.error && <p className="text-[11px] leading-snug text-red-300">{status.embeddings.error}</p>}
        </>}
      </section>
      <p className="px-1 text-xs text-slate-600">v2.0.0</p>
    </aside>
  )
}
