import { Download } from 'lucide-react'
import { reportPdfUrl } from '../lib/api'
import type { ResearchResult } from '../lib/types'
import Markdown from './Markdown'

export default function ReportView({ result }: { result: ResearchResult }) {
  return (
    <article className="panel p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs text-slate-400">Provider: {result.provider} · Quality score: {(result.quality_score * 100).toFixed(0)}%</div>
        {result.final_report && <a className="btn-ghost text-xs" href={reportPdfUrl(result.session_id)} download><Download className="h-4 w-4" />Download PDF</a>}
      </div>
      {result.final_report ? <Markdown text={result.final_report} /> : <p className="text-sm text-slate-400">This session produced no report.</p>}
    </article>
  )
}
