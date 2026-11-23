import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'
import type { ReactNode } from 'react'
import type { DocStatus, Provider } from '../lib/types'

export const Spinner = ({ className = 'h-4 w-4' }: { className?: string }) => <Loader2 className={`animate-spin ${className}`} />

export function ErrorBanner({ children, onClose }: { children: ReactNode; onClose?: () => void }) {
  return (
    <div role="alert" className="flex items-start gap-2 rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="flex-1">{children}</div>
      {onClose && <button onClick={onClose} className="text-red-300 hover:text-white" aria-label="Dismiss">×</button>}
    </div>
  )
}

export function Notice({ children, tone = 'warn' }: { children: ReactNode; tone?: 'warn' | 'info' }) {
  const c = tone === 'warn' ? 'border-amber-500/30 bg-amber-500/10 text-amber-200' : 'border-sky-500/30 bg-sky-500/10 text-sky-200'
  return <div className={`rounded-xl border px-3 py-2 text-sm ${c}`}>{children}</div>
}

export function StatusBadge({ status, error }: { status: DocStatus; error?: string | null }) {
  if (status === 'ready') return <span className="inline-flex items-center gap-1 text-emerald-400"><CheckCircle2 className="h-3 w-3" />Processed</span>
  if (status === 'failed') return <span title={error ?? ''} className="inline-flex items-center gap-1 text-red-400"><AlertCircle className="h-3 w-3" />Failed</span>
  return <span className="inline-flex items-center gap-1 text-amber-300"><Spinner className="h-3 w-3" />{status === 'queued' ? 'Queued' : 'Processing…'}</span>
}

export const ProviderBadge = ({ provider }: { provider: Provider }) => (
  <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-slate-300">
    <span className={`h-2 w-2 rounded-full ${provider === 'ollama' ? 'bg-emerald-400' : 'bg-sky-400'}`} />
    {provider === 'ollama' ? 'Local AI (Ollama)' : 'Gemini (Online)'}
  </span>
)

export function PageHeader({ title, subtitle, right }: { title: ReactNode; subtitle?: ReactNode; right?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-wrap items-start justify-between gap-3">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white md:text-3xl">{title}</h1>
        {subtitle && <p className="mt-1 text-slate-400">{subtitle}</p>}
      </div>
      {right}
    </header>
  )
}
