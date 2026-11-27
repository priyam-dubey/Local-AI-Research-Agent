import { vi } from 'vitest'

type Handler = (url: URL, init?: RequestInit) => unknown | Response
/** Route-based fetch mock: keys are "METHOD /path". */
export function mockApi(overrides: Record<string, Handler>) {
  // the app asks /auth/status first; by default the mocked server is open (no password)
  const routes: Record<string, Handler> = { 'GET /auth/status': () => ({ required: false, authenticated: true }), ...overrides }
  const calls: { key: string; body: unknown }[] = []
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://localhost')
    const key = `${init?.method ?? 'GET'} ${url.pathname}`
    calls.push({ key, body: init?.body ? JSON.parse(String(init.body)) : undefined })
    const h = routes[key]
    if (!h) return new Response(JSON.stringify({ detail: `no mock for ${key}` }), { status: 404 })
    const out = h(url, init)
    return out instanceof Response ? out : new Response(JSON.stringify(out), { status: 200, headers: { 'Content-Type': 'application/json' } })
  }))
  return calls
}

export function sseResponse(events: { event: string; data: unknown }[], chunkSize = 23): Response {
  const wire = events.map(e => `event: ${e.event}\r\ndata: ${JSON.stringify(e.data)}\r\n\r\n`).join('')
  const enc = new TextEncoder()
  const stream = new ReadableStream({
    start(c) { for (let i = 0; i < wire.length; i += chunkSize) c.enqueue(enc.encode(wire.slice(i, i + chunkSize))); c.close() },
  })
  return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
}

export const STATUS = {
  ollama: { available: true, detail: 'Ready', model: 'mistral', models: ['mistral:latest', 'llama3:8b'] },
  gemini: { key_configured: false, model: 'gemini-2.5-flash' },
  vector_store: { engine: 'faiss', ready: true }, embeddings: { backend: 'hf', model: 'all-MiniLM-L6-v2', loaded: true, error: null }, default_provider: 'ollama',
}
export const DOC = (over = {}) => ({ id: 'd1', workspace_id: 'default', filename: 'Machine_Learning_Research.pdf', file_type: 'pdf', size_bytes: 2516582, status: 'ready', error: null, pages: 9, chunks: 30, created_at: 1, ...over })
export const ANSWER = {
  conversation_id: 'c1', message_id: 'm1', answer: 'The research used the MNIST dataset. [S1]', refused: false, grounded: true,
  sources: [{ id: 'S1', document_id: 'd1', filename: 'Machine_Learning_Research.pdf', page: 7, location: 'Page 7', excerpt: 'All experiments use the MNIST dataset', score: 0.8 }],
  warnings: [], standalone_question: 'What dataset was used?', mode: 'all', scope: { mode: 'all', document_ids: ['d1'] }, retrieval: { top_score: 0.8, term_coverage: 1 },
}
