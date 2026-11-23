import { UploadCloud } from 'lucide-react'
import { useRef, useState, type DragEvent } from 'react'
import { uploadDocuments } from '../lib/api'
import type { RejectedFile } from '../lib/types'
import { useApp } from '../state/AppContext'
import { ErrorBanner } from './ui'

const EXTS = ['pdf', 'docx', 'txt', 'md', 'markdown', 'csv']
const MAX_MB = 50

/** Drag & drop / click upload for several files, with a real progress bar. Server re-validates everything. */
export default function UploadDropzone({ compact = false }: { compact?: boolean }) {
  const { workspaceId, refreshDocuments, refreshWorkspaces } = useApp()
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)
  const [names, setNames] = useState<string[]>([])
  const [problems, setProblems] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)

  async function send(files: File[]) {
    const bad: string[] = []
    const ok = files.filter(f => {
      const ext = f.name.split('.').pop()?.toLowerCase() ?? ''
      if (!EXTS.includes(ext)) { bad.push(`${f.name}: unsupported type`); return false }
      if (f.size > MAX_MB * 1024 * 1024) { bad.push(`${f.name}: larger than ${MAX_MB} MB`); return false }
      return true
    })
    setProblems(bad); setError(null)
    if (!ok.length) return
    setNames(ok.map(f => f.name)); setProgress(0)
    try {
      const res = await uploadDocuments(workspaceId, ok, setProgress)
      setProblems(p => [...p, ...res.rejected.map((r: RejectedFile) => `${r.filename}: ${r.error}`)])
      await refreshDocuments(); await refreshWorkspaces()
    } catch (e) { setError((e as Error).message) }
    finally { setProgress(null); setNames([]); if (input.current) input.current.value = '' }
  }

  const onDrop = (e: DragEvent) => { e.preventDefault(); setOver(false); void send(Array.from(e.dataTransfer.files)) }

  return (
    <div className="space-y-2">
      <div role="button" tabIndex={0} aria-label="Upload documents"
        onClick={() => input.current?.click()} onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
        onDragOver={e => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)} onDrop={onDrop}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed text-center transition ${compact ? 'gap-1 px-3 py-4' : 'gap-2 px-4 py-8'} ${over ? 'border-accent bg-accent/10' : 'border-white/15 bg-white/[0.03] hover:bg-white/5'}`}>
        <UploadCloud className={compact ? 'h-6 w-6 text-accent-soft' : 'h-9 w-9 text-accent-soft'} />
        <p className="text-sm text-slate-200">Drag &amp; drop files here, or click to browse</p>
        <p className="text-xs text-slate-500">Supports: PDF, DOCX, TXT, MD, CSV (max {MAX_MB} MB each)</p>
        <input ref={input} type="file" multiple hidden accept=".pdf,.docx,.txt,.md,.markdown,.csv" onChange={e => void send(Array.from(e.target.files ?? []))} />
      </div>
      {progress !== null && (
        <div className="rounded-xl border border-white/10 bg-white/5 p-3" aria-live="polite">
          <div className="mb-1 flex justify-between text-xs text-slate-300"><span className="truncate">Uploading {names.join(', ')}</span><span>{Math.round(progress * 100)}%</span></div>
          <div className="h-1.5 overflow-hidden rounded-full bg-white/10"><div className="h-full bg-accent transition-all" style={{ width: `${progress * 100}%` }} /></div>
        </div>
      )}
      {problems.length > 0 && <ErrorBanner onClose={() => setProblems([])}><ul className="space-y-0.5">{problems.map(p => <li key={p}>{p}</li>)}</ul></ErrorBanner>}
      {error && <ErrorBanner onClose={() => setError(null)}>{error}</ErrorBanner>}
    </div>
  )
}
