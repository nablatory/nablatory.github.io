import {
  useEffect,
  useId,
  useState,
  type ReactNode,
  type ButtonHTMLAttributes,
  type CSSProperties,
} from 'react';
import { fmt, parseNumber } from './format';
import { termAttrs } from './Term';

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'plain' | 'primary' | 'quiet';
  size?: 'sm' | 'md';
};

export function Button({ variant = 'plain', size = 'md', className, ...rest }: ButtonProps) {
  return (
    <button
      type="button"
      className={`btn btn-${variant} btn-${size} ${className ?? ''}`}
      {...rest}
    />
  );
}

export function Field({
  label,
  hint,
  term,
  children,
  inline,
}: {
  label: ReactNode;
  hint?: ReactNode;
  /** Glossary entry explained on hover over the label, together with a text hint. */
  term?: string;
  children: ReactNode;
  inline?: boolean;
}) {
  // A plain-text hint joins the glossary explanation; any other hint keeps its ⓘ.
  const note = typeof hint === 'string' ? hint : undefined;
  const explained = !!term || !!note;
  return (
    <div className={inline ? 'field field-inline' : 'field'}>
      <span className="field-label">
        {explained ? (
          <span
            className="term"
            tabIndex={0}
            {...termAttrs(term, { note, title: typeof label === 'string' ? label : undefined })}
          >
            {label}
          </span>
        ) : (
          label
        )}
        {hint && !note && <Tip>{hint}</Tip>}
      </span>
      {children}
    </div>
  );
}

/** A small ⓘ that reveals an explanation on hover, keyboard focus or tap. */
export function Tip({
  children,
  label = 'More information',
}: {
  children: ReactNode;
  label?: string;
}) {
  const id = useId();
  const [open, setOpen] = useState(false);
  return (
    <span className={`tip${open ? ' is-open' : ''}`}>
      <button
        type="button"
        className="tip-btn"
        aria-label={label}
        aria-describedby={id}
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        onBlur={() => setOpen(false)}
      >
        <svg viewBox="0 0 12 12" width="12" height="12" aria-hidden="true">
          <circle cx="6" cy="6" r="5.25" fill="none" stroke="currentColor" strokeWidth="1" />
          <path
            d="M6 5.2v3.3M6 3.4v.1"
            stroke="currentColor"
            strokeWidth="1.2"
            strokeLinecap="round"
          />
        </svg>
      </button>
      <span className="tip-body" role="tooltip" id={id}>
        {children}
      </span>
    </span>
  );
}

/** A row of tabs. The caller renders the panel with id `panel-${id}` and role="tabpanel". */
export function Tabs<T extends string>({
  tabs,
  value,
  onChange,
  label,
}: {
  tabs: { id: T; label: string; count?: number }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="tabs" role="tablist" aria-label={label}>
      {tabs.map((t) => (
        <button
          key={t.id}
          type="button"
          role="tab"
          id={`tab-${t.id}`}
          aria-selected={value === t.id}
          aria-controls={`panel-${t.id}`}
          className={value === t.id ? 'is-on' : ''}
          onClick={() => onChange(t.id)}
        >
          {t.label}
          {t.count ? <span className="tab-count">{t.count}</span> : null}
        </button>
      ))}
    </div>
  );
}

export function Select<T extends string>({
  value,
  options,
  onChange,
  disabled,
  ariaLabel,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  disabled?: boolean;
  ariaLabel?: string;
}) {
  return (
    <span className="select">
      <select
        value={value}
        disabled={disabled}
        aria-label={ariaLabel}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      <svg viewBox="0 0 10 6" aria-hidden="true">
        <path d="M1 1l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.3" />
      </svg>
    </span>
  );
}

/**
 * Exact numeric entry. Accepts scientific notation; ↑/↓ step (×10 with Shift,
 * or multiplicatively when `log`). Commits on Enter or blur.
 */
export function NumberInput({
  value,
  onCommit,
  min,
  max,
  step = 1,
  log,
  integer,
  disabled,
  ariaLabel,
  width,
}: {
  value: number;
  onCommit: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  log?: boolean;
  integer?: boolean;
  disabled?: boolean;
  ariaLabel?: string;
  width?: number;
}) {
  const [text, setText] = useState(fmtInput(value));
  const [error, setError] = useState(false);
  useEffect(() => setText(fmtInput(value)), [value]);

  const clamp = (v: number) => {
    let x = integer ? Math.round(v) : v;
    if (min !== undefined) x = Math.max(min, x);
    if (max !== undefined) x = Math.min(max, x);
    return x;
  };
  const commit = (raw: string) => {
    const v = parseNumber(raw);
    if (v === null) {
      setError(true);
      return;
    }
    setError(false);
    const c = clamp(v);
    setText(fmtInput(c));
    if (c !== value) onCommit(c);
  };
  return (
    <input
      className={`num-input${error ? ' is-error' : ''}`}
      style={width ? { width } : undefined}
      inputMode="decimal"
      aria-label={ariaLabel}
      aria-invalid={error || undefined}
      disabled={disabled}
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={(e) => commit(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === 'Enter') commit((e.target as HTMLInputElement).value);
        if (e.key === 'Escape') setText(fmtInput(value));
        if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
          e.preventDefault();
          const dir = e.key === 'ArrowUp' ? 1 : -1;
          const cur = parseNumber(text) ?? value;
          const next = log
            ? cur * Math.pow(e.shiftKey ? 10 : Math.SQRT2, dir)
            : cur + dir * step * (e.shiftKey ? 10 : 1);
          const c = clamp(next);
          setText(fmtInput(c));
          onCommit(c);
        }
      }}
    />
  );
}

/** Free-text entry with validation. Commits on Enter or blur; Escape restores. */
export function TextInput({
  value,
  onCommit,
  validate,
  maxLength,
  suggestions,
  disabled,
  ariaLabel,
}: {
  value: string;
  onCommit: (v: string) => void;
  validate?: (s: string) => string | null;
  maxLength?: number;
  suggestions?: string[];
  disabled?: boolean;
  ariaLabel?: string;
}) {
  const [text, setText] = useState(value);
  const [error, setError] = useState<string | null>(null);
  const listId = useId();
  const errId = useId();
  useEffect(() => setText(value), [value]);
  const commit = (raw: string) => {
    const msg = validate?.(raw) ?? null;
    setError(msg);
    if (!msg && raw !== value) onCommit(raw);
  };
  return (
    <>
      <input
        className={`text-input${error ? ' is-error' : ''}`}
        type="text"
        spellCheck={false}
        autoComplete="off"
        maxLength={maxLength}
        list={suggestions?.length ? listId : undefined}
        aria-label={ariaLabel}
        aria-invalid={!!error || undefined}
        aria-describedby={error ? errId : undefined}
        disabled={disabled}
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          if (error) setError(validate?.(e.target.value) ?? null);
        }}
        onBlur={(e) => commit(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commit((e.target as HTMLInputElement).value);
          if (e.key === 'Escape') {
            setText(value);
            setError(null);
          }
        }}
      />
      {suggestions?.length ? (
        <datalist id={listId}>
          {suggestions.map((s) => (
            <option key={s} value={s} />
          ))}
        </datalist>
      ) : null}
      {error && (
        <span id={errId} className="field-hint is-bad" role="alert">
          {error}
        </span>
      )}
    </>
  );
}

function fmtInput(v: number): string {
  return fmt(v, 6).replace('−', '-');
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
  size = 'md',
}: {
  value: T;
  options: { value: T; label: ReactNode; title?: string }[];
  onChange: (v: T) => void;
  ariaLabel: string;
  size?: 'sm' | 'md';
}) {
  return (
    <div className={`segmented segmented-${size}`} role="radiogroup" aria-label={ariaLabel}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          title={o.title}
          className={o.value === value ? 'is-on' : ''}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <span className="toggle">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <label htmlFor={id}>{label}</label>
    </span>
  );
}

export function Range({
  value,
  min,
  max,
  step,
  onChange,
  ariaLabel,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  ariaLabel: string;
}) {
  return (
    <input
      className="range"
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      aria-label={ariaLabel}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  );
}

export function Disclosure({
  title,
  children,
  defaultOpen = false,
  meta,
  className,
}: {
  title: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  meta?: ReactNode;
  className?: string;
}) {
  return (
    <details className={`disclosure ${className ?? ''}`} open={defaultOpen || undefined}>
      <summary>
        <svg viewBox="0 0 8 8" aria-hidden="true">
          <path d="M2 1l3.5 3L2 7" fill="none" stroke="currentColor" strokeWidth="1.3" />
        </svg>
        <span className="disclosure-title">{title}</span>
        {meta && <span className="disclosure-meta">{meta}</span>}
      </summary>
      <div className="disclosure-body">{children}</div>
    </details>
  );
}

/** Small marker distinguishing measured (●) from analytic (○) quantities. */
export function Provenance({ kind }: { kind: 'measured' | 'analytic' }) {
  return (
    <span
      className={`prov prov-${kind}`}
      title={kind === 'measured' ? 'Measured in this browser' : 'Calculated from the shapes'}
    >
      {kind === 'measured' ? '●' : '○'}
      <span className="visually-hidden">{kind}</span>
    </span>
  );
}

export function Spinner({ label }: { label: string }) {
  return (
    <span className="spinner" role="status">
      <span className="spinner-bar" aria-hidden="true" />
      <span className="spinner-label">{label}</span>
    </span>
  );
}

/**
 * A titled surface. `help` is revealed by a small "?" under the title (works
 * with touch and keyboard); `toolbar` sits under the title row; `actions`
 * sit at the right of the title; `span` is the width in a 12-column grid.
 */
export function Card({
  title,
  subtitle,
  help,
  actions,
  toolbar,
  children,
  span,
  className,
  as: Tag = 'section',
}: {
  title?: ReactNode;
  /** A short line under the title (what the axes or rows are). */
  subtitle?: ReactNode;
  help?: ReactNode;
  actions?: ReactNode;
  toolbar?: ReactNode;
  children: ReactNode;
  span?: number;
  className?: string;
  as?: 'section' | 'figure' | 'div';
}) {
  const [showHelp, setShowHelp] = useState(false);
  const helpId = useId();
  const style = span ? ({ '--span': span } as CSSProperties) : undefined;
  return (
    <Tag className={`card ${className ?? ''}`} style={style} data-span={span}>
      {(title || actions) && (
        <header className="card-head">
          <div className="card-title-row">
            {title && <h3 className="card-title">{title}</h3>}
            {help && (
              <button
                type="button"
                className={`card-help-btn${showHelp ? ' is-on' : ''}`}
                aria-expanded={showHelp}
                aria-controls={helpId}
                aria-label={showHelp ? 'Hide explanation' : 'Explain this view'}
                onClick={() => setShowHelp(!showHelp)}
              >
                ?
              </button>
            )}
          </div>
          {actions && <div className="card-actions">{actions}</div>}
          {subtitle && <p className="card-sub">{subtitle}</p>}
        </header>
      )}
      {help && showHelp && (
        <p className="card-help" id={helpId}>
          {help}
        </p>
      )}
      {toolbar && <div className="card-toolbar">{toolbar}</div>}
      <div className="card-body">{children}</div>
    </Tag>
  );
}

/** A headed group of cards inside a tab. */
export function Section({
  title,
  desc,
  actions,
  children,
  id,
}: {
  title: ReactNode;
  desc?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  id?: string;
}) {
  return (
    <section className="lab-section" id={id}>
      <header className="lab-section-head">
        <div>
          <h2 className="lab-section-title">{title}</h2>
          {desc && <p className="lab-section-desc">{desc}</p>}
        </div>
        {actions && <div className="lab-section-actions">{actions}</div>}
      </header>
      <div className="card-grid">{children}</div>
    </section>
  );
}

/** Says what is missing and what to do about it. */
export function EmptyState({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <p>{children}</p>
      {action && <div className="empty-action">{action}</div>}
    </div>
  );
}

/** A row of labelled numbers. */
export function Readout({
  items,
  big,
  className,
}: {
  items: { label: ReactNode; value: ReactNode; title?: string }[];
  big?: boolean;
  className?: string;
}) {
  return (
    <dl className={`readouts${big ? ' readouts-big' : ''} ${className ?? ''}`}>
      {items.map((it, i) => (
        <div key={i} title={it.title}>
          <dt>{it.label}</dt>
          <dd className="num">{it.value}</dd>
        </div>
      ))}
    </dl>
  );
}
