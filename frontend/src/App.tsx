import { useState } from 'react'
import { LandingPage } from './pages/LandingPage'
import { WorkspacePage } from './pages/WorkspacePage'

export type View = 'landing' | 'workspace'

const NAV_ITEMS: { view: View; label: string }[] = [
  { view: 'landing', label: 'Overview' },
  { view: 'workspace', label: 'Workspace' },
]

function App() {
  const [view, setView] = useState<View>('landing')

  return (
    <div className="flex min-h-svh flex-col">
      <header className="sticky top-0 z-10 border-b border-ink-700/80 bg-ink-950/85 backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
          <button
            type="button"
            onClick={() => setView('landing')}
            className="flex items-center gap-2.5 rounded-lg text-left outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
            aria-label="LUMENOIR home"
          >
            <span
              aria-hidden="true"
              className="h-2.5 w-2.5 rounded-full bg-accent shadow-[0_0_14px_2px] shadow-accent/40"
            />
            <span className="text-sm font-medium tracking-[0.32em] text-mist-50">
              LUMENOIR
            </span>
          </button>

          <nav aria-label="Primary" className="flex items-center gap-1">
            {NAV_ITEMS.map((item) => (
              <button
                key={item.view}
                type="button"
                onClick={() => setView(item.view)}
                aria-current={view === item.view ? 'page' : undefined}
                className={`rounded-full px-3 py-1.5 text-sm transition-colors outline-none focus-visible:ring-2 focus-visible:ring-accent/60 ${
                  view === item.view
                    ? 'bg-ink-800 text-mist-50'
                    : 'text-mist-400 hover:text-mist-50'
                }`}
              >
                {item.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="flex-1">
        {view === 'landing' ? (
          <LandingPage onOpenWorkspace={() => setView('workspace')} />
        ) : (
          <WorkspacePage />
        )}
      </main>

      <footer className="border-t border-ink-700/80">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-1 px-4 py-6 text-xs text-mist-500 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
          <p>LUMENOIR · Low-light image restoration</p>
          <p>Foundation build — enhancement engine not implemented yet.</p>
        </div>
      </footer>
    </div>
  )
}

export default App
