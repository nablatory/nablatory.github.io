import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { GLOSSARY, annotate, findTerm } from './glossary';

/**
 * Attributes that give any element (HTML or SVG) a glossary explanation on hover, keyboard focus
 * or tap. `id` names a glossary entry; `title` and `note` add or replace the heading and add a line
 * specific to this place (a setting's own help, a block's shape).
 */
export function termAttrs(
  id: string | undefined,
  opts: { title?: string; note?: string } = {},
): Record<string, string> {
  if (!id && !opts.note) return {};
  const a: Record<string, string> = { 'data-term': id ?? '' };
  if (opts.title) a['data-term-title'] = opts.title;
  if (opts.note) a['data-term-note'] = opts.note;
  return a;
}

/** A word or phrase with a dotted underline that explains itself. */
export function Term({
  id,
  title,
  note,
  children,
}: {
  id?: string;
  title?: string;
  note?: string;
  children: ReactNode;
}) {
  const term = id ?? (typeof children === 'string' ? findTerm(children) : undefined);
  if (!term && !note) return <>{children}</>;
  return (
    <span className="term" tabIndex={0} {...termAttrs(term, { title, note })}>
      {children}
    </span>
  );
}

/** Running text with its key terms explained on hover. */
export function Prose({ text, limit }: { text: string; limit?: number }) {
  return (
    <>
      {annotate(text, limit).map((p, i) =>
        p.term ? (
          <span key={i} className="term" tabIndex={0} data-term={p.term}>
            {p.text}
          </span>
        ) : (
          p.text
        ),
      )}
    </>
  );
}

interface Shown {
  el: Element;
  id: string;
  title?: string;
  note?: string;
}

const TIP_ID = 'glossary-tip';

/** One shared popover for every [data-term] element on the page. Mount once. */
export function GlossaryLayer() {
  const [shown, setShown] = useState<Shown | null>(null);
  const timer = useRef(0);
  const lastPointer = useRef('mouse');

  useEffect(() => {
    const termOf = (t: EventTarget | null) =>
      t instanceof Element ? t.closest('[data-term]') : null;
    const read = (el: Element): Shown => ({
      el,
      id: el.getAttribute('data-term') ?? '',
      title: el.getAttribute('data-term-title') ?? undefined,
      note: el.getAttribute('data-term-note') ?? undefined,
    });
    const show = (el: Element, delay: number) => {
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setShown(read(el)), delay);
    };
    const hide = () => {
      window.clearTimeout(timer.current);
      setShown(null);
    };
    const onOver = (e: PointerEvent) => {
      lastPointer.current = e.pointerType;
      if (e.pointerType === 'touch') return;
      const el = termOf(e.target);
      if (el) show(el, 260);
    };
    const onOut = (e: PointerEvent) => {
      const el = termOf(e.target);
      if (el && termOf(e.relatedTarget) !== el) hide();
    };
    const onDown = (e: PointerEvent) => {
      lastPointer.current = e.pointerType;
    };
    const onClick = (e: MouseEvent) => {
      const el = termOf(e.target);
      // A tap on a plain term (not a control) toggles its explanation.
      if (lastPointer.current === 'touch' && el?.classList.contains('term')) {
        setShown((s) => (s?.el === el ? null : read(el)));
        return;
      }
      if (!el) hide();
    };
    const onFocus = (e: FocusEvent) => {
      const el = termOf(e.target);
      if (el && (e.target as Element).matches?.(':focus-visible')) show(el, 0);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') hide();
    };
    document.addEventListener('pointerover', onOver);
    document.addEventListener('pointerout', onOut);
    document.addEventListener('pointerdown', onDown, true);
    document.addEventListener('click', onClick);
    document.addEventListener('focusin', onFocus);
    document.addEventListener('focusout', hide);
    document.addEventListener('keydown', onKey);
    window.addEventListener('scroll', hide, true);
    return () => {
      window.clearTimeout(timer.current);
      document.removeEventListener('pointerover', onOver);
      document.removeEventListener('pointerout', onOut);
      document.removeEventListener('pointerdown', onDown, true);
      document.removeEventListener('click', onClick);
      document.removeEventListener('focusin', onFocus);
      document.removeEventListener('focusout', hide);
      document.removeEventListener('keydown', onKey);
      window.removeEventListener('scroll', hide, true);
    };
  }, []);

  useEffect(() => {
    if (!shown) return;
    const el = shown.el;
    const prev = el.getAttribute('aria-describedby');
    el.setAttribute('aria-describedby', TIP_ID);
    return () => {
      if (prev) el.setAttribute('aria-describedby', prev);
      else el.removeAttribute('aria-describedby');
    };
  }, [shown]);

  if (!shown || !shown.el.isConnected) return null;
  const entry = GLOSSARY[shown.id];
  if (!entry && !shown.note) return null;
  return createPortal(<Popover shown={shown} />, document.body);
}

function Popover({ shown }: { shown: Shown }) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const entry = GLOSSARY[shown.id];
  useLayoutEffect(() => {
    const box = ref.current;
    if (!box) return;
    const r = shown.el.getBoundingClientRect();
    const w = box.offsetWidth;
    const h = box.offsetHeight;
    const vw = document.documentElement.clientWidth;
    const vh = window.innerHeight;
    const left = Math.max(12, Math.min(vw - w - 12, r.left + r.width / 2 - w / 2));
    const below = r.bottom + 8;
    const top = below + h <= vh - 8 || r.top - 8 - h < 8 ? below : r.top - 8 - h;
    setPos({ left, top });
  }, [shown]);
  const title = shown.title ?? entry?.term;
  return (
    <div
      ref={ref}
      id={TIP_ID}
      role="tooltip"
      className="gloss"
      style={pos ? { left: pos.left, top: pos.top } : { left: 0, top: 0, visibility: 'hidden' }}
    >
      {title && (
        <p className="gloss-term">
          {title}
          {shown.title && entry && entry.term !== shown.title && (
            <span className="gloss-kind">{entry.term}</span>
          )}
        </p>
      )}
      {entry && <p className="gloss-what">{entry.what}</p>}
      {entry?.why && <p className="gloss-why">{entry.why}</p>}
      {shown.note && <p className="gloss-note">{shown.note}</p>}
    </div>
  );
}
