import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import App from '../App'
import { ANSWER, DOC, mockApi, sseResponse, STATUS } from '../test-utils'

const WS = [{ id: 'default', name: 'My Documents', created_at: 1, document_count: 2 }]

function setup(extra: Record<string, () => unknown> = {}, docs = [DOC(), DOC({ id: 'd2', filename: 'Project_Report.docx', file_type: 'docx', size_bytes: 1887436 })]) {
  window.location.hash = '#/assistant'
  return mockApi({
    'GET /system/status': () => STATUS, 'GET /workspaces': () => WS, 'GET /workspaces/default/documents': () => docs,
    'GET /workspaces/default/questions': () => ({ questions: ['What is the main objective of this research?', 'What methodology was used?'] }),
    'POST /workspaces/default/chat/stream': () => sseResponse([
      { event: 'stage', data: { stage: 'retrieving' } }, { event: 'stage', data: { stage: 'generating' } },
      { event: 'stage', data: { stage: 'verifying' } }, { event: 'final', data: ANSWER }]) as unknown as object,
    ...extra,
  })
}

describe('Document Assistant', () => {
  it('lists documents with status and shows stored suggested questions', async () => {
    setup()
    render(<App />)
    expect(await screen.findByText('Machine_Learning_Research.pdf')).toBeInTheDocument()
    expect(screen.getByText('Project_Report.docx')).toBeInTheDocument()
    expect(screen.getAllByText('Processed').length).toBe(2)
    expect(await screen.findByRole('button', { name: 'What methodology was used?' })).toBeInTheDocument()
  })

  it('clicking a suggested question asks it and renders a streamed, cited answer', async () => {
    const calls = setup()
    render(<App />)
    await userEvent.click(await screen.findByRole('button', { name: 'What methodology was used?' }))

    expect(await screen.findByText(/The research used the MNIST dataset/)).toBeInTheDocument()
    expect(screen.getByText('Machine_Learning_Research.pdf, Page 7')).toBeInTheDocument()
    expect(screen.getByText(/Answer based only on the uploaded documents/)).toBeInTheDocument()
    const req = calls.find(c => c.key === 'POST /workspaces/default/chat/stream')!.body as { provider: string; scope: { mode: string }; question: string }
    expect(req).toMatchObject({ question: 'What methodology was used?', provider: 'ollama', scope: { mode: 'all', document_ids: [] } })
  })

  it('selecting a document scopes questions to it; typing a question and sending works', async () => {
    const calls = setup()
    render(<App />)
    await userEvent.click(await screen.findByText('Project_Report.docx'))
    await screen.findByText(/Scope: Project_Report.docx/)
    await userEvent.type(screen.getByLabelText('Message'), 'What is the budget?')
    await userEvent.click(screen.getByRole('button', { name: 'Send' }))
    await waitFor(() => expect(calls.some(c => c.key === 'POST /workspaces/default/chat/stream')).toBe(true))
    expect(calls.find(c => c.key === 'POST /workspaces/default/chat/stream')!.body).toMatchObject({ scope: { mode: 'document', document_ids: ['d2'] } })
  })

  it('surfaces backend errors (e.g. Ollama down) in the chat instead of failing silently', async () => {
    setup({ 'POST /workspaces/default/chat/stream': () => new Response(JSON.stringify({ error: { code: 'provider_unavailable', message: 'Cannot reach Ollama at http://localhost:11434. Start it with `ollama serve`.' } }), { status: 503 }) as unknown as object })
    render(<App />)
    await userEvent.type(await screen.findByLabelText('Message'), 'hello there')
    await userEvent.click(screen.getByRole('button', { name: 'Send' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Cannot reach Ollama')
  })

  it('shows an error event from the stream', async () => {
    setup({ 'POST /workspaces/default/chat/stream': () => sseResponse([{ event: 'stage', data: { stage: 'retrieving' } }, { event: 'error', data: { code: 'rate_limited', message: 'Gemini rate limit or quota reached.' } }]) as unknown as object })
    render(<App />)
    await userEvent.type(await screen.findByLabelText('Message'), 'hello there')
    await userEvent.click(screen.getByRole('button', { name: 'Send' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('rate limit')
  })

  it('blocks asking when nothing is processed yet and shows processing state', async () => {
    setup({}, [DOC({ status: 'processing' })])
    render(<App />)
    expect(await screen.findByText(/still being processed/)).toBeInTheDocument()
    expect(screen.getByLabelText('Message')).toBeDisabled()
    expect(screen.getAllByText(/Processing/).length).toBeGreaterThan(0)
  })

  it('compare tab needs two selected documents and renders a cited comparison', async () => {
    const calls = setup({ 'POST /workspaces/default/chat/stream': () => sseResponse([{ event: 'final', data: { ...ANSWER, mode: 'compare', answer: '## Differences\n- A uses MNIST. [S1]' } }]) as unknown as object })
    render(<App />)
    await userEvent.click(await screen.findByRole('tab', { name: /Compare Documents/ }))
    const btn = screen.getByRole('button', { name: /Compare/ })
    expect(btn).toBeDisabled()
    const list = screen.getAllByRole('checkbox')
    await userEvent.click(list[0]); await userEvent.click(list[1])
    await userEvent.click(screen.getByRole('button', { name: /Compare 2 documents/ }))
    expect(await screen.findByRole('heading', { name: 'Differences' })).toBeInTheDocument()
    expect(calls.find(c => c.key === 'POST /workspaces/default/chat/stream')!.body).toMatchObject({ scope: { mode: 'compare', document_ids: ['d1', 'd2'] } })
  })
})
