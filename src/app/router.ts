import { useSyncExternalStore } from 'react';

const base = import.meta.env.BASE_URL.replace(/\/$/, '');

export type ModelView = 'guide' | 'lab';

export type Route =
  | { page: 'explore' }
  | { page: 'model'; id: string; view: ModelView }
  | { page: 'compare'; params: URLSearchParams }
  | { page: 'about' }
  | { page: 'missing'; path: string };

export function parse(pathname: string, search: string): Route {
  const path = pathname.startsWith(base) ? pathname.slice(base.length) : pathname;
  const parts = path.split('/').filter(Boolean);
  if (parts.length === 0 || (parts.length === 1 && parts[0] === 'explore'))
    return { page: 'explore' };
  if (parts[0] === 'm' && parts[1])
    return {
      page: 'model',
      id: decodeURIComponent(parts[1]),
      view: parts[2] === 'lab' ? 'lab' : 'guide',
    };
  if (parts[0] === 'compare') return { page: 'compare', params: new URLSearchParams(search) };
  if (parts[0] === 'about') return { page: 'about' };
  return { page: 'missing', path };
}

export function href(path: string): string {
  return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}

/** Address of a model page at the given level. */
export function modelPath(id: string, view: ModelView): string {
  return view === 'lab' ? `/m/${id}/lab` : `/m/${id}`;
}

const VIEW_KEY = 'nablatory-view';

/** The level the visitor used last (Guide by default). */
export function preferredView(): ModelView {
  try {
    return localStorage.getItem(VIEW_KEY) === 'lab' ? 'lab' : 'guide';
  } catch {
    return 'guide';
  }
}

export function rememberView(view: ModelView): void {
  try {
    localStorage.setItem(VIEW_KEY, view);
  } catch {
    // Storage unavailable: the choice lasts for this page only.
  }
}

const listeners = new Set<() => void>();

export function navigate(
  path: string,
  opts: { replace?: boolean; keepScroll?: boolean } = {},
): void {
  const url = href(path);
  if (opts.replace) history.replaceState(null, '', url);
  else history.pushState(null, '', url);
  for (const l of listeners) l();
  if (!opts.replace && !opts.keepScroll) window.scrollTo(0, 0);
}

window.addEventListener('popstate', () => {
  for (const l of listeners) l();
});

let cached: { key: string; route: Route } | null = null;
function snapshot(): Route {
  const key = location.pathname + location.search;
  if (cached?.key !== key) cached = { key, route: parse(location.pathname, location.search) };
  return cached.route;
}

export function useRoute(): Route {
  return useSyncExternalStore((fn) => {
    listeners.add(fn);
    return () => listeners.delete(fn);
  }, snapshot);
}

/** Same-origin link handled by the client router. */
export function onLinkClick(e: React.MouseEvent<HTMLAnchorElement>, path: string): void {
  if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey)
    return;
  e.preventDefault();
  navigate(path);
}
