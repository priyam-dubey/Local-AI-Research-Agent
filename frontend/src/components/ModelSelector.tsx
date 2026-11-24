import { Check, Cloud, Cpu, Eye, EyeOff, ShieldCheck } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { deleteGeminiKey, setGeminiKey } from '../lib/api'
import type { Provider } from '../lib/types'
import { useApp } from '../state/AppContext'
import { ErrorBanner, Spinner } from './ui'

function Option({ id, active, icon, title, subtitle, onClick }:
  { id: Provider; active: boolean; icon: React.ReactNode; title: string; subtitle: string; onClick: (p: Provider) => void }) {
  return (
    <button type="button" onClick={() => onClick(id)} aria-pressed={active}
      className={`flex w-full items-center gap-3 rounded-xl border px-3 py-2.5 text-left transition ${active ? 'border-accent bg-accent/15' : 'border-white/10 bg-white/5 hover:bg-white/10'}`}>
      <span className="text-slate-200">{icon}</span>
      <span className="flex-1 leading-tight"><span className="block text-sm font-medium text-white">{title}</span><span className="block text-xs text-slate-400">{subtitle}</span></span>
      {active && <span className="grid h-5 w-5 place-items-center rounded-full bg-accent"><Check className="h-3 w-3 text-white" /></span>}
    </button>
  )
}

export default function ModelSelector() {
  const { provider, setProvider, status, refreshStatus, ollamaModel, setOllamaModel, geminiModel, setGeminiModel } = useApp()
  const [key, setKey] = useState('')
  const [show, setShow] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [warning, setWarning] = useState<string | null>(null)
  const configured = status?.gemini.key_configured ?? false

  async function saveKey(e: FormEvent) {
    e.preventDefault()
    if (!key.trim()) return
    setBusy(true); setError(null); setWarning(null)
    try {
      const r = await setGeminiKey(key.trim())
      setKey('')                                   // never keep the secret in UI state after it has been sent
      setWarning(r.warning)
      await refreshStatus()
    } catch (err) { setError((err as Error).message) }
    finally { setBusy(false) }
  }
  async function removeKey() { setBusy(true); try { await deleteGeminiKey(); await refreshStatus() } finally { setBusy(false) } }

  return (
    <div className="space-y-2">
      <Option id="ollama" active={provider === 'ollama'} icon={<Cpu className="h-5 w-5" />} title="Local Model (Ollama)" subtitle="Private · runs on your machine" onClick={setProvider} />
      {provider === 'ollama' && (
        <select aria-label="Ollama model" className="input" value={ollamaModel} onChange={e => setOllamaModel(e.target.value)}>
          <option value="">Default ({status?.ollama.model ?? 'mistral'})</option>
          {(status?.ollama.models ?? []).map(m => <option key={m} value={m}>{m}</option>)}
        </select>
      )}
      <Option id="gemini" active={provider === 'gemini'} icon={<Cloud className="h-5 w-5" />} title="Gemini (Online)" subtitle="Prompts & excerpts go to Google" onClick={setProvider} />
      {provider === 'gemini' && (
        <form onSubmit={saveKey} className="space-y-2">
          <label className="block text-xs font-medium text-slate-300" htmlFor="gkey">Gemini API Key</label>
          <div className="relative">
            <input id="gkey" className="input pr-9" type={show ? 'text' : 'password'} autoComplete="off" spellCheck={false}
              placeholder={configured ? '•••••••••• (saved on the server)' : 'Paste your API key'} value={key} onChange={e => setKey(e.target.value)} />
            <button type="button" onClick={() => setShow(s => !s)} className="absolute right-2 top-2 text-slate-400 hover:text-white" aria-label={show ? 'Hide key' : 'Show key'}>
              {show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>
          <div className="flex items-center gap-2">
            <button className="btn-primary px-3 py-1.5 text-xs" disabled={busy || !key.trim()}>{busy ? <Spinner className="h-3 w-3" /> : 'Validate & save'}</button>
            {configured && <>
              <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 text-xs text-emerald-300"><Check className="h-3 w-3" />Valid API Key</span>
              <button type="button" onClick={() => void removeKey()} className="text-xs text-slate-400 underline hover:text-white">Remove</button>
            </>}
          </div>
          <input aria-label="Gemini model" className="input" placeholder={`Model (default ${status?.gemini.model ?? 'gemini-2.5-flash'})`} value={geminiModel} onChange={e => setGeminiModel(e.target.value)} />
          <p className="flex items-start gap-1.5 text-[11px] leading-snug text-slate-500"><ShieldCheck className="mt-0.5 h-3 w-3 shrink-0" />The key is sent once to your backend and held in its memory only — never saved in this browser.</p>
          {warning && <p className="text-xs text-amber-300">{warning}</p>}
          {error && <ErrorBanner onClose={() => setError(null)}>{error}</ErrorBanner>}
        </form>
      )}
    </div>
  )
}
