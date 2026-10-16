/**
 * GitHub Pages serves files, not routes. After `vite build`, copy index.html
 * into every route directory so that direct loads of /m/<id>/, /m/<id>/lab/,
 * /compare/ and /about/ return HTTP 200, and to 404.html for anything else.
 */
import { copyFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { MODELS } from '../src/models/catalog';

const dist = join(import.meta.dirname, '..', 'dist');
const index = join(dist, 'index.html');
if (!existsSync(index)) throw new Error('dist/index.html not found; run vite build first');

const routes = [
  'explore',
  'compare',
  'about',
  ...MODELS.flatMap((m) => [`m/${m.id}`, `m/${m.id}/lab`]),
];
for (const r of routes) {
  mkdirSync(join(dist, r), { recursive: true });
  copyFileSync(index, join(dist, r, 'index.html'));
}
copyFileSync(index, join(dist, '404.html'));
writeFileSync(join(dist, '.nojekyll'), '');
console.log(`emitted ${routes.length} routes + 404.html`);
