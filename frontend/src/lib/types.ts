export type Provider = 'ollama' | 'gemini'
export type DocStatus = 'queued' | 'processing' | 'ready' | 'failed'
export type ScopeMode = 'document' | 'all' | 'compare'

export interface Workspace { id: string; name: string; created_at: number; document_count?: number }

export interface DocumentRecord {
  id: string; workspace_id: string; filename: string; file_type: string; size_bytes: number
  status: DocStatus; error: string | null; pages: number; chunks: number; created_at: number
}

export interface Source {
  id: string; document_id: string; filename: string; page: number | null
  location: string; excerpt: string; score: number
}

export interface ChatAnswer {
  conversation_id: string; message_id: string; answer: string; refused: boolean; grounded: boolean
  sources: Source[]; warnings: string[]; standalone_question: string; mode: ScopeMode
  scope: { mode: ScopeMode; document_ids: string[] }
  retrieval: { top_score: number; term_coverage: number }
}

export interface ChatMessage {
  id: string; role: 'user' | 'assistant'; content: string
  sources?: Source[]; refused?: boolean; warnings?: string[]
  pending?: boolean; stage?: string; error?: string
}

export interface SystemStatus {
  ollama: { available: boolean; detail: string; model: string; models: string[] }
  gemini: { key_configured: boolean; model: string }
  vector_store: { engine: string; ready: boolean }
  embeddings: { backend: string; model: string }
  default_provider: Provider
}

export interface ResearchHistoryItem {
  id: string; query: string; provider: Provider; model: string | null
  status: 'complete' | 'error'; quality: number; created_at: number; completed_at: number | null
}

export interface ResearchResult {
  session_id: string; status: string; provider: string; query: string
  final_report: string; citations: string[]; quality_score: number; iterations: number; plan: string[]
}

export interface AgentEvent { role: string; content: string }
export interface RejectedFile { filename: string; code: string; error: string }
