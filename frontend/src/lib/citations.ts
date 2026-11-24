import type { Source } from './types'

/** Turn "[S1]" / "[S1, S2]" markers into markdown links "[1](cite:S1)" that the renderer shows as chips. */
export function linkifyCitations(md: string): string {
  return md.replace(/\[\s*(S\d+(?:\s*[,;]\s*S?\d+)*)\s*\]/g, (_m, inner: string) =>
    (inner.match(/\d+/g) ?? []).map(n => `[${n}](cite:S${n})`).join(' '))
}

export function sourceLabel(s: Pick<Source, 'filename' | 'page' | 'location'>): string {
  const where = s.page ? `Page ${s.page}` : s.location
  return where ? `${s.filename}, ${where}` : s.filename
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}
