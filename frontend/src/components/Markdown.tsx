import ReactMarkdown, { defaultUrlTransform } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { linkifyCitations } from '../lib/citations'

interface Props { text: string; citations?: boolean; onCite?: (id: string) => void }

/** Safe Markdown (no raw HTML). Citation markers become clickable chips; other links open in a new tab. */
export default function Markdown({ text, citations = false, onCite }: Props) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        // only our own cite: scheme is added to the default allow-list (blocks javascript: etc.)
        urlTransform={u => (u.startsWith('cite:') ? u : defaultUrlTransform(u))}
        components={{
          a: ({ href, children }) =>
            href?.startsWith('cite:') ? (
              <button type="button" onClick={() => onCite?.(href.slice(5))}
                className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-md bg-accent/30 px-1 align-baseline text-[11px] font-semibold text-accent-soft hover:bg-accent/50">
                {children}
              </button>
            ) : (
              <a href={href} target="_blank" rel="noopener noreferrer nofollow">{children}</a>
            ),
        }}
      >
        {citations ? linkifyCitations(text) : text}
      </ReactMarkdown>
    </div>
  )
}
