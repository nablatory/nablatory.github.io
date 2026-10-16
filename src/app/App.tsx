import { useEffect, useState } from 'react';
import { findModel } from '../models/catalog';
import { engine } from '../runtime/client';
import type { Capabilities } from '../runtime/protocol';
import { href, onLinkClick, useRoute, type Route } from './router';
import { ExplorePage } from './pages/ExplorePage';
import { ModelPage } from './pages/ModelPage';
import { ComparePage } from './pages/ComparePage';
import { AboutPage } from './pages/AboutPage';
import { MissingPage } from './pages/MissingPage';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { GlossaryLayer } from '../ui/Term';

type Theme = 'system' | 'light' | 'dark';

function readTheme(): Theme {
  try {
    const t = localStorage.getItem('nablatory-theme');
    return t === 'light' || t === 'dark' ? t : 'system';
  } catch {
    return 'system';
  }
}

function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(readTheme);
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'system') delete root.dataset.theme;
    else root.dataset.theme = theme;
    try {
      if (theme === 'system') localStorage.removeItem('nablatory-theme');
      else localStorage.setItem('nablatory-theme', theme);
    } catch {
      // Storage unavailable (private window): the choice lasts for this page only.
    }
  }, [theme]);
  return [theme, setTheme];
}

function EngineStatus() {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [busy, setBusy] = useState(0);
  useEffect(() => {
    engine
      .request({ type: 'capabilities' })
      .then(setCaps)
      .catch(() => undefined);
    return engine.subscribe(() => setBusy(engine.busy));
  }, []);
  const title = caps
    ? [
        'All computation runs in a Web Worker on this device, on the CPU in 32-bit floating point.',
        caps.webgpu
          ? `WebGPU adapter detected${caps.adapter ? ` (${caps.adapter})` : ''}; it is used only for the matrix-multiply benchmark on the How it works page.`
          : 'WebGPU is not available in this browser.',
        caps.cores ? `${caps.cores} logical cores reported.` : '',
      ].join(' ')
    : 'Starting the computation worker';
  return (
    <span className={`engine-status${busy ? ' is-busy' : ''}`} title={title}>
      <span className="engine-dot" aria-hidden="true" />
      <span className="engine-label">{busy ? 'Computing' : 'On your device'}</span>
    </span>
  );
}

function ThemeSwitch({ theme, onChange }: { theme: Theme; onChange: (t: Theme) => void }) {
  const next: Record<Theme, Theme> = { system: 'light', light: 'dark', dark: 'system' };
  const label: Record<Theme, string> = {
    system: 'Theme follows the system',
    light: 'Light theme',
    dark: 'Dark theme',
  };
  return (
    <button
      type="button"
      className="btn btn-quiet btn-icon theme-switch"
      onClick={() => onChange(next[theme])}
      title={`${label[theme]}. Click for ${label[next[theme]].toLowerCase()}.`}
      aria-label={label[theme]}
    >
      <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
        {theme === 'light' && (
          <>
            <circle cx="8" cy="8" r="3.2" fill="none" stroke="currentColor" strokeWidth="1.4" />
            <path
              d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinecap="round"
            />
          </>
        )}
        {theme === 'dark' && (
          <path
            d="M13.5 9.8A6 6 0 0 1 6.2 2.5a6 6 0 1 0 7.3 7.3z"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.4"
            strokeLinejoin="round"
          />
        )}
        {theme === 'system' && (
          <>
            <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.4" />
            <path d="M8 2a6 6 0 0 1 0 12z" fill="currentColor" />
          </>
        )}
      </svg>
    </button>
  );
}

const NAV: { path: string; label: string; page: Route['page'] }[] = [
  { path: '/', label: 'Models', page: 'explore' },
  { path: '/compare', label: 'Compare', page: 'compare' },
  { path: '/about', label: 'How it works', page: 'about' },
];

export function App() {
  const route = useRoute();
  const [theme, setTheme] = useTheme();
  const model = route.page === 'model' ? findModel(route.id) : undefined;

  useEffect(() => {
    if (route.page !== 'model')
      document.title =
        route.page === 'explore'
          ? 'Nablatory'
          : `${route.page === 'compare' ? 'Compare' : route.page === 'about' ? 'How it works' : 'Not found'} · Nablatory`;
  }, [route.page]);

  return (
    <>
      <a className="skip" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <a
          className="wordmark"
          href={href('/')}
          onClick={(e) => onLinkClick(e, '/')}
          aria-label="Nablatory, home"
        >
          <span className="wordmark-nabla" aria-hidden="true">
            ∇
          </span>
          <span>Nablatory</span>
        </a>
        <nav className="topnav" aria-label="Sections">
          {NAV.map((n) => (
            <a
              key={n.path}
              href={href(n.path)}
              onClick={(e) => onLinkClick(e, n.path)}
              aria-current={
                route.page === n.page || (n.page === 'explore' && route.page === 'model')
                  ? 'page'
                  : undefined
              }
            >
              {n.label}
            </a>
          ))}
        </nav>
        <div className="topbar-end">
          <EngineStatus />
          <ThemeSwitch theme={theme} onChange={setTheme} />
        </div>
      </header>
      <ErrorBoundary label="This page" resetKey={route.page === 'model' ? route.id : route}>
        {route.page === 'explore' && <ExplorePage />}
        {route.page === 'model' &&
          (model ? (
            <ModelPage key={model.id} model={model} view={route.view} />
          ) : (
            <MissingPage path={`/m/${route.id}`} />
          ))}
        {route.page === 'compare' && <ComparePage params={route.params} />}
        {route.page === 'about' && <AboutPage />}
        {route.page === 'missing' && <MissingPage path={route.path} />}
      </ErrorBoundary>
      <footer className="site-foot">
        <span className="site-foot-mark">
          <span aria-hidden="true">∇</span> Nablatory
        </span>
        <span>Everything is computed on your device. Nothing is uploaded.</span>
        <span className="site-foot-links">
          <a href={href('/about')} onClick={(e) => onLinkClick(e, '/about')}>
            How it works
          </a>
          <span>MIT licence</span>
        </span>
      </footer>
      <GlossaryLayer />
    </>
  );
}
