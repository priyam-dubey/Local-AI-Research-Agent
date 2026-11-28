import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { AppProvider } from '../state/AppContext'
import { ANSWER, mockApi, STATUS } from '../test-utils'
import Markdown from './Markdown'
import { AnswerView } from './MessageBubble'
import ModelSelector from './ModelSelector'
import Sidebar from './Sidebar'

describe('Markdown', () => {
  it('renders citation markers as chips and calls back with the source id', async () => {
    const got: string[] = []
    render(<Markdown text={'MNIST was used. [S1] And more [S2, S3].'} citations onCite={id => got.push(id)} />)
    await userEvent.click(screen.getByRole('button', { name: '1' }))
    await userEvent.click(screen.getByRole('button', { name: '3' }))
    expect(got).toEqual(['S1', 'S3'])
  })

  it('does not render javascript: links or raw HTML from model/document text', () => {
    const { container } = render(<Markdown text={'[click](javascript:alert(1)) <img src=x onerror=alert(1)> <script>alert(1)</script> [ok](https://example.com)'} />)
    expect(container.innerHTML).not.toContain('javascript:')
    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
    const ok = screen.getByRole('link', { name: 'ok' })
    expect(ok).toHaveAttribute('target', '_blank')
    expect(ok.getAttribute('rel')).toContain('noopener')
  })
})

describe('AnswerView', () => {
  it('shows the grounded badge and a traceable source (file + page)', async () => {
    render(<AnswerView m={{ content: ANSWER.answer, sources: ANSWER.sources, refused: false, warnings: [] }} />)
    expect(screen.getByText(/Answer based only on the uploaded documents/)).toBeInTheDocument()
    expect(screen.getByText('Machine_Learning_Research.pdf, Page 7')).toBeInTheDocument()
    await userEvent.click(screen.getByText('Machine_Learning_Research.pdf, Page 7'))
    expect(screen.getByText(/All experiments use the MNIST dataset/)).toBeInTheDocument()   // excerpt revealed
  })

  it('shows "not found" instead of the grounded badge when the answer was refused, plus warnings', () => {
    render(<AnswerView m={{ content: "I couldn't find this information in the uploaded document.", sources: [], refused: true, warnings: ['1 statement(s) were removed'] }} />)
    expect(screen.getByText(/nothing was guessed/)).toBeInTheDocument()
    expect(screen.queryByText(/Answer based only/)).toBeNull()
    expect(screen.getByText(/1 statement\(s\) were removed/)).toBeInTheDocument()
  })
})

describe('ModelSelector', () => {
  it('keeps the Gemini key out of browser storage and clears it after sending', async () => {
    let saved = false
    const calls = mockApi({
      'GET /system/status': () => ({ ...STATUS, gemini: { ...STATUS.gemini, key_configured: saved } }),
      'GET /workspaces': () => [], 'GET /workspaces/default/documents': () => [],
      'POST /settings/gemini-key': () => { saved = true; return { valid: true, warning: null } },
    })
    const SECRET = 'AIzaSyUITESTUITESTUITESTUITESTUITEST123'
    render(<AppProvider><ModelSelector /></AppProvider>)
    await userEvent.click(await screen.findByRole('button', { name: /Gemini \(Online\)/ }))
    const input = screen.getByLabelText('Gemini API Key') as HTMLInputElement
    expect(input.type).toBe('password')
    await userEvent.type(input, SECRET)
    await userEvent.click(screen.getByRole('button', { name: /Validate & save/ }))

    await waitFor(() => expect(screen.getByText('Valid API Key')).toBeInTheDocument())
    expect(calls.find(c => c.key === 'POST /settings/gemini-key')?.body).toEqual({ api_key: SECRET })
    expect(input.value).toBe('')                                                        // cleared from the UI
    expect(JSON.stringify({ ...localStorage })).not.toContain(SECRET)                  // never persisted in the browser
    expect(JSON.stringify({ ...sessionStorage })).not.toContain(SECRET)
    expect(localStorage.getItem('ai.provider')).toBe('gemini')                         // only the non-secret choice is remembered
  })

  it('shows a readable error for an invalid key', async () => {
    mockApi({
      'GET /system/status': () => STATUS, 'GET /workspaces': () => [], 'GET /workspaces/default/documents': () => [],
      'POST /settings/gemini-key': () => new Response(JSON.stringify({ error: { code: 'invalid_api_key', message: 'The Gemini API key is invalid or not permitted.' } }), { status: 401 }),
    })
    render(<AppProvider><ModelSelector /></AppProvider>)
    await userEvent.click(await screen.findByRole('button', { name: /Gemini \(Online\)/ }))
    await userEvent.type(screen.getByLabelText('Gemini API Key'), 'not-a-real-key-123')
    await userEvent.click(screen.getByRole('button', { name: /Validate & save/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('invalid or not permitted')
  })

  it('lists installed Ollama models for the local option', async () => {
    mockApi({ 'GET /system/status': () => STATUS, 'GET /workspaces': () => [], 'GET /workspaces/default/documents': () => [] })
    render(<AppProvider><ModelSelector /></AppProvider>)
    expect(await screen.findByRole('option', { name: 'llama3:8b' })).toBeInTheDocument()
  })
})


describe('Sidebar system status', () => {
  const mount = () => render(<MemoryRouter><AppProvider><Sidebar /></AppProvider></MemoryRouter>)
  const api = (embeddings: object, ollamaUp = true) => mockApi({
    'GET /system/status': () => ({ ...STATUS, ollama: { ...STATUS.ollama, available: ollamaUp }, embeddings: { ...STATUS.embeddings, ...embeddings } }),
    'GET /workspaces': () => [], 'GET /workspaces/default/documents': () => [],
  })

  it('shows the embedding model loading, then ready', async () => {
    api({ loaded: false })
    mount()
    expect(await screen.findByText('Loading…')).toBeInTheDocument()
  })

  it('shows a load failure (e.g. offline first run) with the reason', async () => {
    api({ loaded: false, error: 'Could not load embedding model. Check your internet connection.' })
    mount()
    expect(await screen.findByText('Error')).toBeInTheDocument()
    expect(screen.getByText(/internet connection/)).toBeInTheDocument()
  })

  it('shows Ollama offline when it is not running', async () => {
    api({}, false)
    mount()
    expect(await screen.findByText('Offline')).toBeInTheDocument()
  })
})
