import { createContext, useCallback, useContext, useEffect, useState, type FormEvent, type ReactNode } from 'react'
import { BrainCircuit, Lock } from 'lucide-react'
import { authEvents, getAuthStatus, login, logout } from '../lib/api'
import { ErrorBanner, Spinner } from '../components/ui'

interface AuthState { required: boolean; signOut: () => Promise<void> }
const Ctx = createContext<AuthState>({ required: false, signOut: async () => undefined })
export const useAuth = () => useContext(Ctx)

function LoginScreen({ onDone }: { onDone: () => void }) {
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit(e: FormEvent) {
    e.preventDefault()
    setBusy(true); setError(null)
    try { await login(password); setPassword(''); onDone() }
    catch (err) { setError((err as Error).message) }
    finally { setBusy(false) }
  }
  return (
    <div className="grid h-full place-items-center p-6">
      <form onSubmit={submit} className="panel w-full max-w-sm space-y-4 p-6">
        <div className="flex items-center gap-3"><BrainCircuit className="h-9 w-9 text-accent-soft" /><h1 className="text-lg font-semibold text-white">Local AI Research Agent</h1></div>
        <p className="flex items-center gap-2 text-sm text-slate-400"><Lock className="h-4 w-4" />This instance is password-protected.</p>
        <label htmlFor="pw" className="block text-sm font-medium text-slate-300">Password</label>
        <input id="pw" type="password" autoFocus autoComplete="current-password" className="input" value={password} onChange={e => setPassword(e.target.value)} />
        {error && <ErrorBanner>{error}</ErrorBanner>}
        <button className="btn-primary w-full" disabled={busy || !password}>{busy ? <Spinner /> : 'Sign in'}</button>
      </form>
    </div>
  )
}

/** Shows the login page when the backend requires a password and there is no valid session. */
export function AuthGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<{ required: boolean; authenticated: boolean } | null>(null)
  const [offline, setOffline] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try { setState(await getAuthStatus()); setOffline(null) } catch (e) { setOffline((e as Error).message) }
  }, [])
  useEffect(() => { void refresh() }, [refresh])
  useEffect(() => {                                   // a 401 anywhere (expired session) sends the user back to login
    const onExpired = () => setState(s => (s ? { ...s, authenticated: false } : s))
    authEvents.addEventListener('auth-required', onExpired)
    return () => authEvents.removeEventListener('auth-required', onExpired)
  }, [])

  const signOut = useCallback(async () => { await logout().catch(() => undefined); await refresh() }, [refresh])

  if (offline) return <div className="grid h-full place-items-center p-6"><ErrorBanner>{offline}</ErrorBanner></div>
  if (!state) return <div className="grid h-full place-items-center"><Spinner className="h-6 w-6" /></div>
  if (state.required && !state.authenticated) return <LoginScreen onDone={() => void refresh()} />
  return <Ctx.Provider value={{ required: state.required, signOut }}>{children}</Ctx.Provider>
}
