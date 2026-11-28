import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import App from '../App'
import { mockApi, STATUS } from '../test-utils'

const WS = [{ id: 'default', name: 'My Documents', created_at: 1, document_count: 0 }]
const app = { 'GET /system/status': () => STATUS, 'GET /workspaces': () => WS, 'GET /workspaces/default/documents': () => [], 'GET /research': () => [] }

describe('password-protected deployment', () => {
  it('opens straight into the app when no password is required', async () => {
    mockApi({ 'GET /auth/status': () => ({ required: false, authenticated: true }), ...app })
    render(<App />)
    expect(await screen.findByText(/Welcome to your AI Research/)).toBeInTheDocument()
    expect(screen.queryByText('Sign out')).toBeNull()
  })

  it('shows the login page, rejects a wrong password, then signs in', async () => {
    let authed = false
    const calls = mockApi({
      'GET /auth/status': () => ({ required: true, authenticated: authed }),
      'POST /auth/login': (_u, init) => {
        const { password } = JSON.parse(String(init?.body))
        if (password !== 'right') return new Response(JSON.stringify({ error: { code: 'invalid_password', message: 'Incorrect password.' } }), { status: 401 })
        authed = true
        return { authenticated: true }
      },
      ...app,
    })
    render(<App />)
    expect(await screen.findByText(/password-protected/)).toBeInTheDocument()
    expect(calls.some(c => c.key === 'GET /system/status')).toBe(false)                // nothing else is requested before sign-in

    await userEvent.type(screen.getByLabelText('Password'), 'wrong')
    await userEvent.click(screen.getByRole('button', { name: 'Sign in' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Incorrect password')

    await userEvent.clear(screen.getByLabelText('Password'))
    await userEvent.type(screen.getByLabelText('Password'), 'right')
    await userEvent.click(screen.getByRole('button', { name: 'Sign in' }))
    expect(await screen.findByText(/Welcome to your AI Research/)).toBeInTheDocument()
    expect(screen.getByText('Sign out')).toBeInTheDocument()
  })

  it('sends the user back to the login page when the session expires mid-use', async () => {
    mockApi({ 'GET /auth/status': () => ({ required: true, authenticated: true }), ...app,
      'GET /workspaces/default/documents': () => new Response(JSON.stringify({ error: { code: 'auth_required', message: 'Please sign in to continue.' } }), { status: 401 }) })
    render(<App />)
    expect(await screen.findByText(/password-protected/)).toBeInTheDocument()
  })
})
