import { createSSEParser } from './sse'
import type {
  AgentEvent, ChatAnswer, DocumentRecord, Provider, RejectedFile, ResearchHistoryItem,
  ResearchResult, ScopeMode, SystemStatus, Workspace,
} from './types'

export const API_BASE: string =
  (import.meta.env.VITE_API_URL as string | undefined) ?? (import.meta.env.DEV ? 'http://localhost:8000' : '')

export class ApiError extends Error {
  constructor(message: string, public code = 'error', public status = 0) { super(message) }
}

function toApiError(status: number, body: unknown): ApiError {
  const b = body as { error?: { code?: string; message?: string }; detail?: unknown } | null
  if (b?.error?.message) return new ApiError(b.error.message, b.error.code ?? 'error', status)
  if (typeof b?.detail === 'string') return new ApiError(b.detail, 'error', status)
  if (Array.isArray(b?.detail)) return new ApiError('Invalid request: ' + (b!.detail as { msg?: string }[]).map(d => d.msg).join('; '), 'validation', status)
  return new ApiError(`Request failed (HTTP ${status})`, 'error', status)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(API_BASE + path, init)
  } catch {
    throw new ApiError('Cannot reach the backend. Is it running on ' + (API_BASE || 'this server') + '?', 'network')
  }
  const body = await res.json().catch(() => null)
  if (!res.ok) throw toApiError(res.status, body)
  return body as T
}

const json = (method: string, body?: unknown): RequestInit => ({
  method, headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body),
})

// ---------------- system & settings ----------------
export const getSystemStatus = () => request<SystemStatus>('/system/status')
/** The key is sent once to the backend, which validates and keeps it in server memory. It is never stored in the browser. */
export const setGeminiKey = (apiKey: string) => request<{ valid: boolean; warning: string | null }>('/settings/gemini-key', json('POST', { api_key: apiKey }))
export const deleteGeminiKey = () => request<{ configured: boolean }>('/settings/gemini-key', { method: 'DELETE' })

// ---------------- workspaces & documents ----------------
export const listWorkspaces = () => request<Workspace[]>('/workspaces')
export const createWorkspace = (name: string) => request<Workspace>('/workspaces', json('POST', { name }))
export const deleteWorkspace = (id: string) => request<{ deleted: string }>(`/workspaces/${id}`, { method: 'DELETE' })
export const listDocuments = (ws: string) => request<DocumentRecord[]>(`/workspaces/${ws}/documents`)
export const deleteDocument = (ws: string, id: string) => request<{ deleted: string }>(`/workspaces/${ws}/documents/${id}`, { method: 'DELETE' })

/** Multi-file upload with real progress (fetch cannot report upload progress, XHR can). */
export function uploadDocuments(ws: string, files: File[], onProgress: (fraction: number) => void):
  Promise<{ documents: DocumentRecord[]; rejected: RejectedFile[] }> {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    files.forEach(f => form.append('files', f))
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/workspaces/${ws}/documents`)
    xhr.upload.onprogress = e => { if (e.lengthComputable) onProgress(e.loaded / e.total) }
    xhr.onerror = () => reject(new ApiError('Upload failed: cannot reach the backend.', 'network'))
    xhr.onload = () => {
      let body: unknown = null
      try { body = JSON.parse(xhr.responseText) } catch { /* non-JSON error page */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as { documents: DocumentRecord[]; rejected: RejectedFile[] })
      else reject(toApiError(xhr.status, body))
    }
    xhr.send(form)
  })
}

// ---------------- questions ----------------
export const getStoredQuestions = (ws: string, documentId = '') =>
  request<{ questions: string[] }>(`/workspaces/${ws}/questions?document_id=${encodeURIComponent(documentId)}`)
export const generateQuestions = (ws: string, body: { document_id?: string; count?: number; provider: Provider; model?: string }) =>
  request<{ questions: string[]; all: string[] }>(`/workspaces/${ws}/questions`, json('POST', body))

// ---------------- chat (SSE over POST) ----------------
export interface ChatRequest {
  question: string; conversation_id?: string; provider: Provider; model?: string
  scope: { mode: ScopeMode; document_ids: string[] }
}
export interface ChatHandlers { onStage: (stage: string) => void; onFinal: (a: ChatAnswer) => void; onError: (e: ApiError) => void }

export async function chatStream(ws: string, body: ChatRequest, h: ChatHandlers, signal?: AbortSignal): Promise<void> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}/workspaces/${ws}/chat/stream`, { ...json('POST', body), signal })
  } catch (e) {
    if ((e as Error).name === 'AbortError') return
    return h.onError(new ApiError('Cannot reach the backend.', 'network'))
  }
  if (!res.ok) return h.onError(toApiError(res.status, await res.json().catch(() => null)))
  if (!res.body) return h.onError(new ApiError('The server sent no data.'))

  let finished = false
  const parser = createSSEParser(ev => {
    const data = JSON.parse(ev.data)
    if (ev.event === 'stage') h.onStage(data.stage)
    else if (ev.event === 'final') { finished = true; h.onFinal(data as ChatAnswer) }
    else if (ev.event === 'error') { finished = true; h.onError(new ApiError(data.message, data.code)) }
  })
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      parser.feed(dec.decode(value, { stream: true }))
    }
  } catch (e) {
    if ((e as Error).name === 'AbortError') return
    return h.onError(new ApiError('The connection was interrupted.', 'network'))
  }
  if (!finished) h.onError(new ApiError('The answer stream ended unexpectedly.'))
}

// ---------------- research ----------------
export const startResearch = (body: { query: string; provider: Provider; model?: string; workspace_id?: string }) =>
  request<{ session_id: string; status: string; provider: string }>('/research', json('POST', body))
export const getResearchResult = (id: string) => request<ResearchResult>(`/research/${id}/result`)
export const researchHistory = () => request<ResearchHistoryItem[]>('/research')
export const reportPdfUrl = (id: string) => `${API_BASE}/research/${id}/report.pdf`

/** Live agent events. Calls onEnd when the backend sends [COMPLETE] / [ERROR]. */
export function streamResearch(id: string, onEvent: (e: AgentEvent) => void, onEnd: (status: 'complete' | 'error') => void): () => void {
  const es = new EventSource(`${API_BASE}/research/${id}/stream`)
  let ended = false
  es.onmessage = m => {
    const ev = JSON.parse(m.data) as AgentEvent
    if (ev.role === 'system' && (ev.content === '[COMPLETE]' || ev.content === '[ERROR]')) {
      ended = true
      es.close()
      onEnd(ev.content === '[COMPLETE]' ? 'complete' : 'error')
    } else onEvent(ev)
  }
  es.onerror = () => { if (!ended) { es.close(); onEnd('error') } }
  return () => es.close()
}
