import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { getSystemStatus, listDocuments, listWorkspaces } from '../lib/api'
import type { DocumentRecord, Provider, SystemStatus, Workspace } from '../lib/types'

// Only NON-secret preferences are kept in the browser. The Gemini key never is.
const LS = { provider: 'ai.provider', ollama: 'ai.ollamaModel', gemini: 'ai.geminiModel', workspace: 'ai.workspace' }
const read = (k: string, d: string) => { try { return localStorage.getItem(k) ?? d } catch { return d } }
const write = (k: string, v: string) => { try { localStorage.setItem(k, v) } catch { /* private mode */ } }

interface AppState {
  provider: Provider
  setProvider: (p: Provider) => void
  model: string | undefined          // model for the selected provider ('' -> backend default)
  ollamaModel: string
  setOllamaModel: (m: string) => void
  geminiModel: string
  setGeminiModel: (m: string) => void
  status: SystemStatus | null
  statusError: string | null
  refreshStatus: () => Promise<void>
  providerProblem: string | null
  workspaces: Workspace[]
  refreshWorkspaces: () => Promise<void>
  workspaceId: string
  setWorkspaceId: (id: string) => void
  documents: DocumentRecord[]
  readyDocuments: DocumentRecord[]
  refreshDocuments: () => Promise<void>
}

const Ctx = createContext<AppState | null>(null)
export const useApp = (): AppState => {
  const v = useContext(Ctx)
  if (!v) throw new Error('useApp must be used inside <AppProvider>')
  return v
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [provider, setProviderState] = useState<Provider>(read(LS.provider, 'ollama') === 'gemini' ? 'gemini' : 'ollama')
  const [ollamaModel, setOllamaModelState] = useState(read(LS.ollama, ''))
  const [geminiModel, setGeminiModelState] = useState(read(LS.gemini, ''))
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [workspaces, setWorkspaces] = useState<Workspace[]>([])
  const [workspaceId, setWorkspaceIdState] = useState(read(LS.workspace, 'default'))
  const [documents, setDocuments] = useState<DocumentRecord[]>([])

  const setProvider = (p: Provider) => { setProviderState(p); write(LS.provider, p) }
  const setOllamaModel = (m: string) => { setOllamaModelState(m); write(LS.ollama, m) }
  const setGeminiModel = (m: string) => { setGeminiModelState(m); write(LS.gemini, m) }
  const setWorkspaceId = (id: string) => { setWorkspaceIdState(id); write(LS.workspace, id) }

  const refreshStatus = useCallback(async () => {
    try { setStatus(await getSystemStatus()); setStatusError(null) }
    catch (e) { setStatusError((e as Error).message) }
  }, [])
  const refreshWorkspaces = useCallback(async () => {
    try {
      const ws = await listWorkspaces()
      setWorkspaces(ws)
      setWorkspaceIdState(cur => (ws.some(w => w.id === cur) ? cur : 'default'))
    } catch { /* surfaced through statusError */ }
  }, [])
  const refreshDocuments = useCallback(async () => {
    try { setDocuments(await listDocuments(workspaceId)) } catch { setDocuments([]) }
  }, [workspaceId])

  useEffect(() => { void refreshStatus(); void refreshWorkspaces(); const t = setInterval(() => void refreshStatus(), 15000); return () => clearInterval(t) }, [refreshStatus, refreshWorkspaces])
  useEffect(() => { setDocuments([]); void refreshDocuments() }, [refreshDocuments])

  // Poll while anything is still being processed.
  const busy = documents.some(d => d.status === 'queued' || d.status === 'processing')
  useEffect(() => {
    if (!busy) return
    const t = setInterval(() => void refreshDocuments(), 1200)
    return () => clearInterval(t)
  }, [busy, refreshDocuments])

  const providerProblem = useMemo(() => {
    if (statusError) return statusError
    if (!status) return null
    if (provider === 'ollama' && !status.ollama.available) return status.ollama.detail
    if (provider === 'gemini' && !status.gemini.key_configured) return 'Gemini is selected but no API key is configured. Add one under Model Selection.'
    return null
  }, [provider, status, statusError])

  const value: AppState = {
    provider, setProvider, ollamaModel, setOllamaModel, geminiModel, setGeminiModel,
    model: (provider === 'ollama' ? ollamaModel : geminiModel) || undefined,
    status, statusError, refreshStatus, providerProblem,
    workspaces, refreshWorkspaces, workspaceId, setWorkspaceId,
    documents, readyDocuments: documents.filter(d => d.status === 'ready'), refreshDocuments,
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
