import { useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

export function Tex({
  tex,
  block = false,
  className,
}: {
  tex: string;
  block?: boolean;
  className?: string;
}) {
  const html = useMemo(
    () =>
      katex.renderToString(tex, {
        displayMode: block,
        throwOnError: false,
        strict: 'ignore',
        output: 'html',
      }),
    [tex, block],
  );
  return (
    <span
      className={className ? `tex ${className}` : 'tex'}
      data-block={block || undefined}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
