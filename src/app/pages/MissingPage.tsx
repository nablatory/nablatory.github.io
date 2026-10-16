import { MODELS } from '../../models/catalog';
import { href, onLinkClick } from '../router';

export function MissingPage({ path }: { path: string }) {
  const slug = path.split('/').filter(Boolean).pop() ?? '';
  const near = MODELS.filter((m) => slug && (m.id.includes(slug) || slug.includes(m.id))).slice(
    0,
    4,
  );
  return (
    <main className="page page-narrow" id="main">
      <h1 className="page-h">Nothing is at {path}</h1>
      <p className="prose">
        The address may be mistyped, or the model may have been renamed.{' '}
        <a href={href('/')} onClick={(e) => onLinkClick(e, '/')}>
          The catalog
        </a>{' '}
        lists every model on the site.
      </p>
      {near.length > 0 && (
        <ul className="plain-list">
          {near.map((m) => (
            <li key={m.id}>
              <a href={href(`/m/${m.id}`)} onClick={(e) => onLinkClick(e, `/m/${m.id}`)}>
                {m.name}
              </a>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
