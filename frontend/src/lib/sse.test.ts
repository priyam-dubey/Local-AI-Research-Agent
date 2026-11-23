import { describe, expect, it } from 'vitest'
import { linkifyCitations, sourceLabel } from './citations'
import { createSSEParser, type SSEEvent } from './sse'

describe('SSE parser', () => {
  it('parses named events split across arbitrary chunk boundaries', () => {
    const got: SSEEvent[] = []
    const p = createSSEParser(e => got.push(e))
    const wire = 'event: stage\ndata: {"stage":"retrieving"}\n\nevent: final\ndata: {"answer":"hi"}\n\n'
    for (let i = 0; i < wire.length; i += 7) p.feed(wire.slice(i, i + 7))
    expect(got).toEqual([
      { event: 'stage', data: '{"stage":"retrieving"}' },
      { event: 'final', data: '{"answer":"hi"}' },
    ])
  })
  it('handles CRLF, comments and multi-line data', () => {
    const got: SSEEvent[] = []
    createSSEParser(e => got.push(e)).feed(': ping\r\n\r\ndata: a\r\ndata: b\r\n\r\n')
    expect(got).toEqual([{ event: 'message', data: 'a\nb' }])
  })
})

describe('citations', () => {
  it('turns markers into cite links and keeps other brackets', () => {
    expect(linkifyCitations('MNIST was used. [S1] See [link] and [S2, S3].'))
      .toBe('MNIST was used. [1](cite:S1) See [link] and [2](cite:S2) [3](cite:S3).')
  })
  it('labels sources with page or location', () => {
    expect(sourceLabel({ filename: 'a.pdf', page: 7, location: 'Page 7' })).toBe('a.pdf, Page 7')
    expect(sourceLabel({ filename: 'a.md', page: null, location: 'Section: Methods' })).toBe('a.md, Section: Methods')
  })
})
