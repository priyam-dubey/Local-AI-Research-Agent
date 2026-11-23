import DocumentList from '../components/DocumentList'
import UploadDropzone from '../components/UploadDropzone'
import { PageHeader } from '../components/ui'
import { deleteDocument } from '../lib/api'
import type { DocumentRecord } from '../lib/types'
import { useApp } from '../state/AppContext'

export default function DocumentsPage() {
  const { documents, workspaces, workspaceId, setWorkspaceId, refreshDocuments, refreshWorkspaces } = useApp()
  async function remove(d: DocumentRecord) {
    if (!confirm(`Remove “${d.filename}”?`)) return
    await deleteDocument(workspaceId, d.id); await refreshDocuments(); await refreshWorkspaces()
  }
  async function clearAll() {
    const removable = documents.filter(d => d.status !== 'queued' && d.status !== 'processing')
    if (!removable.length || !confirm(`Remove all ${removable.length} documents from this workspace?`)) return
    for (const d of removable) await deleteDocument(workspaceId, d.id)
    await refreshDocuments(); await refreshWorkspaces()
  }
  return (
    <div className="mx-auto max-w-4xl">
      <PageHeader title="Uploaded Documents" subtitle="Each workspace is a separate collection. Documents never mix between workspaces."
        right={<select aria-label="Workspace" className="input w-52" value={workspaceId} onChange={e => setWorkspaceId(e.target.value)}>{workspaces.map(w => <option key={w.id} value={w.id}>{w.name} ({w.document_count ?? 0})</option>)}</select>} />
      <div className="grid gap-4 md:grid-cols-2">
        <section className="panel p-5"><UploadDropzone /></section>
        <section className="panel p-5">
          <div className="mb-2 flex items-center justify-between"><h2 className="font-semibold text-white">Documents ({documents.length})</h2>
            {documents.length > 0 && <button onClick={() => void clearAll()} className="text-xs text-red-300 hover:underline">Clear All</button>}</div>
          <DocumentList docs={documents} onDelete={d => void remove(d)} empty="No documents in this workspace." />
        </section>
      </div>
    </div>
  )
}
