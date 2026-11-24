import { Send } from 'lucide-react'
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { chatStream } from '../lib/api'
import type { ChatMessage } from '../lib/types'
import { useApp } from '../state/AppContext'
import MessageBubble from './MessageBubble'
import { Notice } from './ui'

export interface ChatHandle { ask: (q: string) => void }
interface Props { mode: 'document' | 'all'; documentId?: string; disabledReason?: string | null }

const uid = () => Math.random().toString(36).slice(2, 10)

/** Conversation with follow-ups. Remount (change `key`) to start a fresh conversation. */
const ChatPanel = forwardRef<ChatHandle, Props>(function ChatPanel({ mode, documentId, disabledReason }, ref) {
  const { workspaceId, provider, model, providerProblem } = useApp()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const conv = useRef<string | undefined>(undefined)
  const abort = useRef<AbortController | null>(null)
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => () => abort.current?.abort(), [])
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [messages])

  function ask(question: string) {
    const q = question.trim()
    if (!q || busy || disabledReason) return
    const pid = uid()
    setMessages(m => [...m, { id: uid(), role: 'user', content: q }, { id: pid, role: 'assistant', content: '', pending: true, stage: 'retrieving' }])
    setInput(''); setBusy(true)
    const patch = (p: Partial<ChatMessage>) => setMessages(ms => ms.map(x => (x.id === pid ? { ...x, ...p } : x)))
    abort.current = new AbortController()
    void chatStream(workspaceId, {
      question: q, conversation_id: conv.current, provider, model,
      scope: { mode, document_ids: mode === 'document' && documentId ? [documentId] : [] },
    }, {
      onStage: s => patch({ stage: s }),
      onFinal: a => { conv.current = a.conversation_id; patch({ pending: false, content: a.answer, sources: a.sources, refused: a.refused, warnings: a.warnings }); setBusy(false) },
      onError: e => { patch({ pending: false, error: e.message }); setBusy(false) },
    }, abort.current.signal)
  }
  useImperativeHandle(ref, () => ({ ask }))

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex-1 space-y-5 overflow-y-auto px-1 py-2" role="log" aria-live="polite">
        {messages.length === 0 && <p className="py-10 text-center text-sm text-slate-500">Ask a question about {mode === 'document' ? 'the selected document' : 'your documents'}. Answers use only their content and cite the source.</p>}
        {messages.map(m => <MessageBubble key={m.id} m={m} />)}
        <div ref={bottom} />
      </div>
      <div className="space-y-2 pt-3">
        {(disabledReason || providerProblem) && <Notice>{disabledReason ?? providerProblem}</Notice>}
        <form onSubmit={e => { e.preventDefault(); ask(input) }} className="flex items-center gap-2 rounded-2xl border border-white/10 bg-ink-900/80 p-2">
          <input className="min-w-0 flex-1 bg-transparent px-2 text-sm text-white outline-none placeholder:text-slate-500" placeholder="Ask another question about your documents…"
            aria-label="Message" value={input} maxLength={2000} onChange={e => setInput(e.target.value)} disabled={!!disabledReason} />
          <button className="btn-primary h-10 w-10 !p-0" disabled={busy || !input.trim() || !!disabledReason} aria-label="Send"><Send className="h-4 w-4" /></button>
        </form>
      </div>
    </div>
  )
})
export default ChatPanel
