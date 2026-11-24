import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'

export default function Layout() {
  return (
    <div className="flex h-full">
      <Sidebar />
      <main className="h-full min-w-0 flex-1 overflow-y-auto p-6 md:p-8"><Outlet /></main>
    </div>
  )
}
