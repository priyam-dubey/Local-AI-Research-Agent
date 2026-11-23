import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import { AppProvider } from './state/AppContext'
import Dashboard from './pages/Dashboard'
import DocumentAssistant from './pages/DocumentAssistant'
import DocumentsPage from './pages/DocumentsPage'
import History from './pages/History'
import Research from './pages/Research'
import Settings from './pages/Settings'

export default function App() {
  return (
    <AppProvider>
      <HashRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="research" element={<Research />} />
            <Route path="assistant" element={<DocumentAssistant />} />
            <Route path="documents" element={<DocumentsPage />} />
            <Route path="history" element={<History />} />
            <Route path="history/:id" element={<History />} />
            <Route path="settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </HashRouter>
    </AppProvider>
  )
}
