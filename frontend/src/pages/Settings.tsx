import { Plus, Trash2 } from 'lucide-react'
import { useState } from 'react'
import ModelSelector from '../components/ModelSelector'
import { ErrorBanner, Notice, PageHeader } from '../components/ui'
import { createWorkspace, deleteWorkspace } from '../lib/api'
import { useApp } from '../state/AppContext'

export default function Settings() {
  const { workspaces, workspaceId, setWorkspaceId, refreshWorkspaces, status } = useApp()
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)

  async function add() {
    if (!name.trim()) return
    try { const w = await createWorkspace(name.trim()); setName(''); await refreshWorkspaces(); setWorkspaceId(w.id) } catch (e) { setError((e as Error).message) }
  }
  async function remove(id: string, label: string) {
    if (!confirm(`Delete workspace “${label}” and all its documents?`)) return
    try { await deleteWorkspace(id); if (workspaceId === id) setWorkspaceId('default'); await refreshWorkspaces() } catch (e) { setError((e as Error).message) }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <PageHeader title="Settings" />
      {error && <ErrorBanner onClose={() => setError(null)}>{error}</ErrorBanner>}
      <section className="panel space-y-3 p-5"><h2 className="font-semibold text-white">AI model</h2>
        <p className="text-sm text-slate-400">You choose the provider explicitly; the app never switches silently. Embeddings and the vector index always run locally, whichever provider you pick.</p>
        <div className="max-w-sm"><ModelSelector /></div></section>
      <section className="panel space-y-3 p-5"><h2 className="font-semibold text-white">Workspaces</h2>
        <ul className="divide-y divide-white/5">{workspaces.map(w => (
          <li key={w.id} className="flex items-center justify-between py-2 text-sm"><span className="text-slate-200">{w.name} <span className="text-xs text-slate-500">· {w.document_count ?? 0} docs{w.id === workspaceId ? ' · active' : ''}</span></span>
            {w.id !== 'default' && <button aria-label={`Delete ${w.name}`} onClick={() => void remove(w.id, w.name)} className="rounded-md p-1.5 text-slate-500 hover:bg-red-500/10 hover:text-red-300"><Trash2 className="h-4 w-4" /></button>}</li>))}</ul>
        <form className="flex gap-2" onSubmit={e => { e.preventDefault(); void add() }}>
          <input className="input" placeholder="New workspace name" maxLength={80} value={name} onChange={e => setName(e.target.value)} />
          <button className="btn-primary"><Plus className="h-4 w-4" />Add</button></form></section>
      <section className="panel space-y-2 p-5"><h2 className="font-semibold text-white">Privacy & security</h2>
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-400">
          <li><b className="text-slate-200">Local mode</b>: prompts and document excerpts stay on this machine. Research can still use DuckDuckGo web search for topics your documents don’t cover.</li>
          <li><b className="text-slate-200">Gemini mode</b>: the question and the retrieved excerpts are sent to Google’s API.</li>
          <li>Your Gemini key is held in backend memory only (or in the backend’s <code>.env</code>), never in this browser.</li>
          <li>Embedding model: <code>{status?.embeddings.model ?? '…'}</code></li>
        </ul>
        <Notice>This app has no user accounts. Run it for yourself on localhost; do not expose it to a shared network before adding authentication.</Notice></section>
    </div>
  )
}
