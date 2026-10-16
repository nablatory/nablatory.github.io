import { useMemo, useState } from 'react';
import { stats } from '../engine/tensor';
import { Heatmap } from './Heatmap';
import { ScaleBar } from './ScaleBar';
import { Histogram } from './Histogram';
import { rangeScale, symmetricScale, type Scale } from './colormap';
import { fmt, fmtShape } from '../ui/format';
import { Segmented } from '../ui/controls';

export interface TensorLike {
  shape: number[];
  data: ArrayLike<number>;
}

/** Leading indices select a 2-D slice over the last two axes. */
function sliceInfo(shape: number[], lead: number[]) {
  const r = shape.length;
  const rows = r >= 2 ? shape[r - 2] : 1;
  const cols = r >= 1 ? shape[r - 1] : 1;
  let offset = 0;
  let stride = rows * cols;
  for (let i = r - 3; i >= 0; i--) {
    offset += (lead[i] ?? 0) * stride;
    stride *= shape[i];
  }
  return { rows, cols, offset };
}

/** Values per line when one long row is wrapped: a divisor of n near 32 when there is one. */
function wrapWidth(n: number): number {
  let best = 32;
  for (let d = 16; d <= 48; d++)
    if (n % d === 0 && (n % best !== 0 || Math.abs(d - 32) < Math.abs(best - 32))) best = d;
  return best;
}

export function flatToIndex(shape: number[], flat: number): number[] {
  const idx: number[] = [];
  for (let i = shape.length - 1; i >= 0; i--) {
    idx.unshift(flat % shape[i]);
    flat = Math.floor(flat / shape[i]);
  }
  return idx;
}

export function TensorView({
  tensor,
  label,
  kind = 'value',
  previous,
  onPick,
  marked,
  maxWidth = 320,
  compact = false,
  scaleOverride,
}: {
  tensor: TensorLike;
  label: string;
  kind?: 'value' | 'grad' | 'update' | 'diff';
  /** Same-shape tensor from the previous computation; enables Δ view. */
  previous?: TensorLike | null;
  onPick?: (flat: number, value: number) => void;
  /** Flat indices to outline (edited elements). */
  marked?: ReadonlySet<number>;
  maxWidth?: number;
  compact?: boolean;
  scaleOverride?: Scale;
}) {
  const { shape } = tensor;
  const [lead, setLead] = useState<number[]>(() =>
    shape.slice(0, Math.max(0, shape.length - 2)).map(() => 0),
  );
  const [mode, setMode] = useState<'map' | 'numbers' | 'histogram'>('map');
  const [showDiff, setShowDiff] = useState(false);
  const [hover, setHover] = useState<{ idx: number[]; v: number; prev?: number } | null>(null);

  const leadSafe = shape
    .slice(0, Math.max(0, shape.length - 2))
    .map((d, i) => Math.min(lead[i] ?? 0, d - 1));
  const { rows, cols, offset } = sliceInfo(shape, leadSafe);
  // A single long row (a vector) is drawn on several lines so every element keeps a visible cell.
  const wrap = rows === 1 && cols > 32 ? wrapWidth(cols) : 0;
  const at = (r: number, c: number) => (wrap ? [0, r * wrap + c] : [r, c]);
  const canDiff = !!previous && previous.data.length === tensor.data.length;
  const diffing = canDiff && showDiff;

  const shown = useMemo(() => {
    if (!diffing) return tensor.data;
    const d = new Float32Array(tensor.data.length);
    for (let i = 0; i < d.length; i++) d[i] = tensor.data[i] - previous!.data[i];
    return d;
  }, [diffing, tensor.data, previous]);

  const sliceData = useMemo(
    () => Array.prototype.slice.call(shown, offset, offset + rows * cols) as number[],
    [shown, offset, rows, cols],
  );
  const scale = useMemo(
    () =>
      scaleOverride && !diffing
        ? scaleOverride
        : kind === 'value' && !diffing && isNonNegative(tensor.data)
          ? rangeScale(tensor.data, 'sequential')
          : symmetricScale(sliceData),
    [scaleOverride, diffing, kind, tensor.data, sliceData],
  );
  const st = useMemo(() => stats(tensor.data), [tensor.data]);
  const diffStats = useMemo(
    () => (canDiff ? stats(Array.from(tensor.data, (v, i) => v - previous!.data[i])) : null),
    [canDiff, tensor.data, previous],
  );

  const markedSlice = useMemo(() => {
    if (!marked?.size) return undefined;
    const s = new Set<number>();
    for (const k of marked) if (k >= offset && k < offset + rows * cols) s.add(k - offset);
    return s;
  }, [marked, offset, rows, cols]);

  const scaleLabel = diffing
    ? 'new − previous'
    : kind === 'grad'
      ? '∂L/∂·'
      : kind === 'update'
        ? 'Δθ'
        : 'value';

  if (shape.length === 0) {
    return (
      <div className="tensor-view">
        <div className="tensor-scalar num">{fmt(tensor.data[0], 6)}</div>
        {canDiff && <div className="muted">previous {fmt(previous!.data[0], 6)}</div>}
      </div>
    );
  }

  return (
    <div className={`tensor-view${compact ? ' is-compact' : ''}`}>
      {!compact && (
        <div className="tensor-head">
          <span className="tensor-kind" data-kind={diffing ? 'diff' : kind}>
            {kind === 'grad' && !diffing ? '∇ ' : ''}
            {label}
          </span>
          <span className="tensor-shape">{fmtShape(shape)}</span>
        </div>
      )}
      <dl className="tensor-stats">
        <div>
          <dt>min</dt>
          <dd>{fmt(st.min)}</dd>
        </div>
        <div>
          <dt>max</dt>
          <dd>{fmt(st.max)}</dd>
        </div>
        <div>
          <dt>mean</dt>
          <dd>{fmt(st.mean)}</dd>
        </div>
        <div>
          <dt>std</dt>
          <dd>{fmt(st.std)}</dd>
        </div>
        <div>
          <dt>‖·‖₂</dt>
          <dd>{fmt(st.l2)}</dd>
        </div>
        {st.nonFinite > 0 && (
          <div className="is-bad">
            <dt>non-finite</dt>
            <dd>{st.nonFinite}</dd>
          </div>
        )}
        {diffStats && (
          <div>
            <dt>max |Δ|</dt>
            <dd>{fmt(diffStats.absMax)}</dd>
          </div>
        )}
      </dl>
      <div className="tensor-controls">
        {leadSafe.map((v, i) => (
          <label key={i} className="axis-picker">
            <span>{axisName(shape.length, i)}</span>
            <input
              type="number"
              min={0}
              max={shape[i] - 1}
              value={v}
              onChange={(e) => {
                const next = [...leadSafe];
                next[i] = Math.max(0, Math.min(shape[i] - 1, Number(e.target.value) || 0));
                setLead(next);
              }}
              aria-label={`${axisName(shape.length, i)} index`}
            />
            <span className="muted">/ {shape[i]}</span>
          </label>
        ))}
        {!compact && (
          <Segmented
            ariaLabel="Display"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'map', label: 'Map' },
              { value: 'numbers', label: 'Numbers', title: 'Exact values of this slice' },
              { value: 'histogram', label: 'Histogram' },
            ]}
          />
        )}
        {canDiff && (
          <label className="diff-toggle">
            <input
              type="checkbox"
              checked={showDiff}
              onChange={(e) => setShowDiff(e.target.checked)}
            />{' '}
            Δ previous
          </label>
        )}
      </div>
      {mode === 'map' && (
        <div className="tensor-map">
          <Heatmap
            data={shown}
            offset={offset}
            rows={wrap ? Math.ceil(cols / wrap) : rows}
            cols={wrap || cols}
            count={wrap ? cols : undefined}
            scale={scale}
            maxWidth={maxWidth}
            marked={markedSlice}
            label={`${label} slice ${leadSafe.join(',')}`}
            onHover={(vr, vc, v) => {
              const [r, c] = at(vr, vc);
              setHover({
                idx: [...leadSafe, ...(shape.length >= 2 ? [r] : []), c],
                v,
                prev: previous && !diffing ? previous.data[offset + r * cols + c] : undefined,
              });
            }}
            onLeave={() => setHover(null)}
            onPick={
              onPick
                ? (vr, vc) => {
                    const [r, c] = at(vr, vc);
                    onPick(offset + r * cols + c, tensor.data[offset + r * cols + c]);
                  }
                : undefined
            }
          />
          {wrap > 0 && (
            <p className="tensor-note">
              One row of {cols} values, drawn {wrap} per line.
            </p>
          )}
          <div className="tensor-readout">
            <ScaleBar scale={scale} label={scaleLabel} />
            <span className="readout num" aria-live="polite">
              {hover ? (
                <>
                  [{hover.idx.join(', ')}] = {fmt(hover.v, 6)}
                  {hover.prev !== undefined && hover.prev !== hover.v && (
                    <span className="muted"> (was {fmt(hover.prev, 6)})</span>
                  )}
                </>
              ) : onPick ? (
                'Click an element to edit it'
              ) : (
                ' '
              )}
            </span>
          </div>
        </div>
      )}
      {mode === 'numbers' && (
        <NumberGrid
          data={shown}
          offset={offset}
          rows={rows}
          cols={cols}
          onPick={onPick ? (k) => onPick(offset + k, tensor.data[offset + k]) : undefined}
          marked={markedSlice}
        />
      )}
      {mode === 'histogram' && (
        <Histogram
          series={[{ data: shown, color: kind === 'grad' ? 'var(--grad)' : 'var(--ink-2)', label }]}
          width={maxWidth}
          symmetric={kind !== 'value' || diffing}
        />
      )}
    </div>
  );
}

function NumberGrid({
  data,
  offset,
  rows,
  cols,
  onPick,
  marked,
}: {
  data: ArrayLike<number>;
  offset: number;
  rows: number;
  cols: number;
  onPick?: (k: number) => void;
  marked?: ReadonlySet<number>;
}) {
  const maxCells = 256;
  const r = Math.min(rows, Math.max(1, Math.floor(maxCells / cols)));
  const c = Math.min(cols, 16);
  return (
    <div className="number-grid-wrap">
      <table className="number-grid num">
        <tbody>
          {Array.from({ length: r }, (_, i) => (
            <tr key={i}>
              {Array.from({ length: c }, (_, j) => {
                const k = i * cols + j;
                const v = data[offset + k];
                return (
                  <td
                    key={j}
                    className={`${v < 0 ? 'is-neg' : ''}${marked?.has(k) ? ' is-marked' : ''}`}
                  >
                    {onPick ? (
                      <button type="button" onClick={() => onPick(k)}>
                        {fmt(v, 3)}
                      </button>
                    ) : (
                      fmt(v, 3)
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {(r < rows || c < cols) && (
        <p className="muted">
          Showing {r}×{c} of {rows}×{cols}.
        </p>
      )}
    </div>
  );
}

function axisName(rank: number, i: number): string {
  if (rank === 4) return ['batch', 'channel'][i];
  if (rank === 3) return 'index 0';
  return `axis ${i}`;
}

function isNonNegative(d: ArrayLike<number>): boolean {
  for (let i = 0; i < d.length; i++) if (d[i] < 0) return false;
  return true;
}
