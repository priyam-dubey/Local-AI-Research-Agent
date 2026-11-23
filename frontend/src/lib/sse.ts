/** Minimal Server-Sent-Events parser for fetch() streams (EventSource cannot POST). */
export interface SSEEvent { event: string; data: string }

export function createSSEParser(onEvent: (e: SSEEvent) => void) {
  let buffer = ''
  return {
    feed(chunk: string) {
      buffer += chunk.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
      let idx: number
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const raw = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 2)
        let event = 'message'
        const data: string[] = []
        for (const line of raw.split('\n')) {
          if (!line || line.startsWith(':')) continue
          const c = line.indexOf(':')
          const field = c === -1 ? line : line.slice(0, c)
          const value = c === -1 ? '' : line.slice(c + 1).replace(/^ /, '')
          if (field === 'event') event = value
          else if (field === 'data') data.push(value)
        }
        if (data.length) onEvent({ event, data: data.join('\n') })
      }
    },
  }
}
