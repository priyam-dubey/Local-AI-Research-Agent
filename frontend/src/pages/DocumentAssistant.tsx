import { MessageSquarePlus, Scale, MessageSquareText } from 'lucide-react'
import { useRef, useState } from 'react'
import ChatPanel, { type ChatHandle } from '../components/ChatPanel'
import ComparePanel from '../components/ComparePanel'
import DocumentList from '../components/DocumentList'
import SuggestedQuestions from '../components/SuggestedQuestions'
import UploadDropzone from '../components/UploadDropzone'
import { Notice, PageHeader, ProviderBadge } from '../components/ui'
import { deleteDocument } from '../lib/api'
import type { DocumentRecord } from '../lib/types'
import { useApp } from '../state/AppContext'

export default function DocumentAssistant() {
  const { workspaces, workspaceId, setWorkspaceId, documents, readyDocuments, refreshDocuments, provider } = useApp()
  const [tab, setTab] = useState<'chat' | 'compare'>('chat')
  const [selected, setSelected] = useState<string | null>(null)       // null = all documents
  const [chatKey, setChatKey] = useState(0)
  const chat = useRef<ChatHandle>(null)

  const sel = readyDocuments.find(d => d.id === selected)
  const mode = sel ? 'document' : 'all'
  const nothingReady = readyDocuments.length === 0

  async function remove(d: DocumentRecord) {
    if (!confirm(`Remove “${d.filename}” from this workspace?`)) return
    try { await deleteDocument(workspaceId, d.id) } finally {
      if (selected === d.id) setSelected(null)
      await refreshDocuments()
    }
  }

  return (
    <div className="mx-auto flex h-full max-w-7xl flex-col">
      <PageHeader title="Document Assistant" subtitle="Grounded answers from your own documents — with sources." right={<ProviderBadge provider={provider} />} />
      <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[22rem_1fr]">
        <aside className="panel flex min-h-0 flex-col gap-3 overflow-y-auto p-4">
          <div>
            <label htmlFor="ws" className="mb-1 block text-xs font-medium text-slate-400">Workspace</label>
            <select id="ws" className="input" value={workspaceId} onChange={e => { setWorkspaceId(e.target.value); setSelected(null); setChatKey(k => k + 1) }}>
              {workspaces.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}
            </select>
          </div>
          <UploadDropzone compact />
          <div className="flex items-center justify-between"><h2 className="text-sm font-semibold text-white">Documents ({documents.length})</h2>
            {selected && <button className="text-xs text-accent-soft hover:underline" onClick={() => { setSelected(null); setChatKey(k => k + 1) }}>Use all documents</button>}</div>
          <DocumentList docs={documents} mode="single" selected={selected ? [selected] : []}
            onSelect={id => { setSelected(id === selected ? null : id); setChatKey(k => k + 1) }} onDelete={d => void remove(d)} empty="Upload a document to get started." />
        </aside>

        <section className="panel flex min-h-[32rem] min-w-0 flex-col p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div role="tablist" className="flex gap-1 rounded-xl bg-white/5 p-1">
              {([['chat', 'Chat with Documents', MessageSquareText], ['compare', 'Compare Documents', Scale]] as const).map(([id, label, Icon]) => (
                <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)}
                  className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm ${tab === id ? 'bg-accent text-white' : 'text-slate-300 hover:bg-white/5'}`}><Icon className="h-4 w-4" />{label}</button>))}
            </div>
            {tab === 'chat' && (
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span className="truncate rounded-full border border-white/10 px-2.5 py-1">Scope: {sel ? sel.filename : `all documents (${readyDocuments.length})`}</span>
                <button className="btn-ghost px-2.5 py-1 text-xs" onClick={() => setChatKey(k => k + 1)}><MessageSquarePlus className="h-3.5 w-3.5" />New chat</button>
              </div>)}
          </div>

          {tab === 'chat' ? (
            <div className="flex min-h-0 flex-1 flex-col gap-3">
              {nothingReady && <Notice tone="info">{documents.length ? 'Your documents are still being processed…' : 'Upload a document to start asking questions.'}</Notice>}
              <SuggestedQuestions documentId={sel?.id} disabled={nothingReady} onPick={q => chat.current?.ask(q)} />
              <ChatPanel key={`${workspaceId}:${selected ?? 'all'}:${chatKey}`} ref={chat} mode={mode} documentId={sel?.id}
                disabledReason={nothingReady ? 'No processed documents yet.' : null} />
            </div>
          ) : <ComparePanel docs={documents} />}
        </section>
      </div>
    </div>
  )
}
