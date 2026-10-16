import { createContext, useContext, useMemo, useState, type ReactNode } from 'react';
import type { ViewSpec, BatchMeta, ProcedureResult } from '../models/types';
import type { Trace } from '../runtime/trace';
import { Heatmap } from './Heatmap';
import { ScaleBar } from './ScaleBar';
import { LineChart } from './LineChart';
import { rangeScale, symmetricScale, type Scale } from './colormap';
import { fmt, fmtShape } from '../ui/format';
import { Card, EmptyState, Segmented } from '../ui/controls';

export interface Resolved {
  shape: number[];
  data: ArrayLike<number>;
  /** Previous value of the same tensor, when shapes agree. */
  previous?: ArrayLike<number>;
}

export interface ViewContext {
  trace: Trace | null;
  previous: Trace | null;
  procedures: Record<string, { result: ProcedureResult }>;
}

export function resolve(ref: string, ctx: ViewContext): Resolved | null {
  if (ref.startsWith('proc:')) {
    const [, id, name] = ref.split(':');
    const t = ctx.procedures[id]?.result.tensors[name];
    return t ? { shape: t.shape, data: t.data } : null;
  }
  const t = ctx.trace;
  if (!t) return null;
  const find = (tr: Trace) => {
    if (ref.startsWith('out:')) {
      const id = tr.outputs[ref.slice(4)];
      return tr.nodes.find((n) => n.id === id);
    }
    if (ref.startsWith('in:')) return tr.nodes.find((n) => n.key === `input:${ref.slice(3)}`);
    return undefined;
  };
  const n = find(t);
  if (!n) return null;
  const p = ctx.previous ? find(ctx.previous) : undefined;
  return {
    shape: n.shape,
    data: n.value,
    previous: p && p.value.length === n.value.length ? p.value : undefined,
  };
}

export function metaOf(spec: ViewSpec, ctx: ViewContext): BatchMeta | undefined {
  const src = 'source' in spec ? spec.source : '';
  if (typeof src === 'string' && src.startsWith('proc:'))
    return ctx.procedures[src.split(':')[1]]?.result.meta;
  return ctx.trace?.meta;
}

export const CLASS_COLORS = [
  '#3f6fb0',
  '#c2463c',
  '#2f8f6f',
  '#c08a1e',
  '#7a5aa6',
  '#3a9bb5',
  '#9a6236',
  '#6f8a2c',
  '#b4508a',
  '#5e6a73',
];

/** Controls injected by the page: a procedure's arguments and Run button, and a Run button for empty views. */
export const ViewExtras = createContext<{
  toolbar?: ReactNode;
  runner?: ReactNode;
  /** Appended to the view's explanation (e.g. what the procedure behind it does). */
  help?: ReactNode;
  /** Width in the 12-column grid, replacing the view's own. */
  span?: number;
}>({});

function Frame({
  title,
  note,
  children,
  actions,
  span = 6,
}: {
  title: string;
  note?: ReactNode;
  children: ReactNode;
  actions?: ReactNode;
  span?: number;
}) {
  const extras = useContext(ViewExtras);
  // A long title keeps its first part; the parenthesised reading guide goes underneath.
  const cut = title.length > 44 ? title.indexOf(' (') : -1;
  const main = cut > 0 && title.endsWith(')') ? title.slice(0, cut) : title;
  const sub = main !== title ? title.slice(cut + 2, -1) : undefined;
  const help =
    note && extras.help ? (
      <>
        {note} {extras.help}
      </>
    ) : (
      (note ?? extras.help)
    );
  return (
    <Card
      as="figure"
      className="view"
      title={main}
      subtitle={sub && sub[0].toUpperCase() + sub.slice(1)}
      help={help}
      actions={actions}
      toolbar={extras.toolbar}
      span={extras.span ?? span}
    >
      {children}
    </Card>
  );
}

function Missing({ title, what, span }: { title: string; what: string; span?: number }) {
  const extras = useContext(ViewExtras);
  return (
    <Frame title={title} span={span}>
      <EmptyState action={extras.runner}>{what}</EmptyState>
    </Frame>
  );
}

/** One image (first in batch) drawn in greyscale or RGB. */
function ImageCanvas({
  shape,
  data,
  offset = 0,
  range,
  size = 168,
  label,
}: {
  shape: number[];
  data: ArrayLike<number>;
  offset?: number;
  range?: [number, number];
  size?: number;
  label: string;
}) {
  const [C, H, W] = shape.length >= 3 ? shape.slice(-3) : [1, shape[0], shape[1]];
  const scale: Scale = useMemo(() => {
    if (range) return { mode: 'grey', min: range[0], max: range[1] };
    const slice = Array.prototype.slice.call(data, offset, offset + C * H * W) as number[];
    return rangeScale(slice, 'grey');
  }, [range, data, offset, C, H, W]);
  if (C === 3)
    return (
      <RgbCanvas data={data} offset={offset} H={H} W={W} scale={scale} size={size} label={label} />
    );
  return (
    <Heatmap
      data={data}
      offset={offset}
      rows={H}
      cols={W}
      scale={scale}
      maxWidth={size}
      maxHeight={size}
      label={label}
      gridLines={false}
    />
  );
}

function RgbCanvas({
  data,
  offset,
  H,
  W,
  scale,
  size,
  label,
}: {
  data: ArrayLike<number>;
  offset: number;
  H: number;
  W: number;
  scale: Scale;
  size: number;
  label: string;
}) {
  const cell = Math.max(1, Math.floor(size / Math.max(H, W)));
  return (
    <canvas
      className="heatmap"
      width={W}
      height={H}
      style={{ width: W * cell, height: H * cell, imageRendering: 'pixelated' }}
      role="img"
      aria-label={label}
      ref={(c) => {
        if (!c) return;
        const img = new ImageData(W, H);
        for (let i = 0; i < H * W; i++)
          for (let ch = 0; ch < 3; ch++) {
            const v = (data[offset + ch * H * W + i] - scale.min) / (scale.max - scale.min);
            img.data[i * 4 + ch] = Math.round(Math.max(0, Math.min(1, v)) * 255);
          }
        for (let i = 0; i < H * W; i++) img.data[i * 4 + 3] = 255;
        c.getContext('2d')!.putImageData(img, 0, 0);
      }}
    />
  );
}

function ImageView({
  spec,
  ctx,
}: {
  spec: Extract<ViewSpec, { kind: 'image' }>;
  ctx: ViewContext;
}) {
  const r = resolve(spec.source, ctx);
  if (!r)
    return <Missing span={4} title={spec.title} what={spec.unavailable ?? 'Not computed yet.'} />;
  const range = spec.range;
  return (
    <Frame
      span={4}
      title={spec.title}
      note={
        range
          ? `Greyscale: ${fmt(range[0])} → black, ${fmt(range[1])} → white.`
          : 'Greyscale: tensor minimum → black, maximum → white.'
      }
    >
      <ImageCanvas shape={r.shape} data={r.data} range={range} label={spec.title} />
      {spec.caption && <p className="view-caption">{spec.caption}</p>}
    </Frame>
  );
}

function ImageGridView({
  spec,
  ctx,
}: {
  spec: Extract<ViewSpec, { kind: 'imageGrid' }>;
  ctx: ViewContext;
}) {
  const r = resolve(spec.source, ctx);
  if (!r) return <Missing span={6} title={spec.title} what="Not computed yet." />;
  const [N, C, H, W] = r.shape.length === 4 ? r.shape : [r.shape[0], 1, r.shape[1], r.shape[2]];
  const per = C * H * W;
  const cols = spec.columns ?? Math.min(N, 8);
  const size = Math.max(H, Math.min(84, Math.floor(560 / cols)));
  return (
    <Frame
      span={6}
      title={spec.title}
      note={
        spec.range
          ? `Greyscale: ${fmt(spec.range[0])} → black, ${fmt(spec.range[1])} → white. ${N} samples.`
          : `${N} samples, each scaled to its own range.`
      }
    >
      <div className="image-grid" style={{ gridTemplateColumns: `repeat(${cols}, auto)` }}>
        {Array.from({ length: N }, (_, i) => (
          <ImageCanvas
            key={i}
            shape={[C, H, W]}
            data={r.data}
            offset={i * per}
            range={spec.range}
            size={size}
            label={`${spec.title} ${i}`}
          />
        ))}
      </div>
    </Frame>
  );
}

function FeatureMapsView({
  spec,
  ctx,
  onSelect,
}: {
  spec: Extract<ViewSpec, { kind: 'featureMaps' }>;
  ctx: ViewContext;
  onSelect?: (ref: string) => void;
}) {
  const r = resolve(spec.source, ctx);
  const [norm, setNorm] = useState<'shared' | 'each'>('shared');
  const [delta, setDelta] = useState(false);
  const fmSpan = r && r.shape[1] <= 8 ? 6 : 12;
  if (!r)
    return (
      <Missing span={fmSpan} title={spec.title} what={spec.unavailable ?? 'Not computed yet.'} />
    );
  const [, C, H, W] = r.shape;
  const per = H * W;
  const useDelta = delta && !!r.previous;
  const data = useDelta
    ? Float32Array.from({ length: C * per }, (_, i) => r.data[i] - r.previous![i])
    : r.data;
  const first = Array.prototype.slice.call(data, 0, C * per) as number[];
  const nonNeg = !useDelta && first.every((v) => v >= 0);
  const shared: Scale = nonNeg ? rangeScale(first, 'sequential') : symmetricScale(first);
  const size = Math.max(36, Math.min(96, Math.floor(480 / Math.min(C, 8))));
  return (
    <Frame
      span={fmSpan}
      title={spec.title}
      actions={
        <span className="view-actions">
          <Segmented
            ariaLabel="Colour normalisation"
            value={norm}
            onChange={setNorm}
            options={[
              { value: 'shared', label: 'Shared scale' },
              { value: 'each', label: 'Per map' },
            ]}
          />
          {r.previous && (
            <label className="diff-toggle">
              <input type="checkbox" checked={delta} onChange={(e) => setDelta(e.target.checked)} />{' '}
              Δ
            </label>
          )}
        </span>
      }
      note={
        <>
          {C} maps of {H}×{W}, first example.{' '}
          {useDelta
            ? 'Colour shows new − previous activation.'
            : nonNeg
              ? 'Colour: activation value (all ≥ 0).'
              : 'Colour: signed activation value.'}{' '}
          {norm === 'each' && 'Each map uses its own symmetric range, so maps are not comparable.'}
        </>
      }
    >
      {norm === 'shared' && (
        <ScaleBar scale={shared} label={useDelta ? 'Δ activation' : 'activation'} />
      )}
      <div className="fmap-grid">
        {Array.from({ length: C }, (_, c) => {
          const slice = Array.prototype.slice.call(data, c * per, (c + 1) * per) as number[];
          const s =
            norm === 'shared'
              ? shared
              : nonNeg
                ? rangeScale(slice, 'sequential')
                : symmetricScale(slice);
          return (
            <button
              key={c}
              type="button"
              className="fmap"
              onClick={() => onSelect?.(spec.source)}
              title={`Channel ${c}`}
            >
              <Heatmap
                data={data}
                offset={c * per}
                rows={H}
                cols={W}
                scale={s}
                maxWidth={size}
                maxHeight={size}
                label={`${spec.title} channel ${c}`}
                gridLines={false}
              />
              <span className="fmap-label">{c}</span>
            </button>
          );
        })}
      </div>
    </Frame>
  );
}

function ProbsView({
  spec,
  ctx,
}: {
  spec: Extract<ViewSpec, { kind: 'probs' }>;
  ctx: ViewContext;
}) {
  const r = resolve(spec.source, ctx);
  if (!r)
    return <Missing span={4} title={spec.title} what={spec.unavailable ?? 'Not computed yet.'} />;
  const K = r.shape[r.shape.length - 1];
  const names =
    spec.classNames ??
    ctx.trace?.meta?.classNames ??
    Array.from({ length: K }, (_, i) => String(i));
  let arg = 0;
  for (let k = 1; k < K; k++) if (r.data[k] > r.data[arg]) arg = k;
  return (
    <Frame
      span={4}
      title={spec.title}
      note={
        r.previous ? 'Bars: current. Tick: previous computation.' : 'First example of the batch.'
      }
    >
      <div className="probs" role="list">
        {Array.from({ length: K }, (_, k) => {
          const v = r.data[k];
          const pv = r.previous?.[k];
          return (
            <div key={k} className={`prob-row${k === arg ? ' is-top' : ''}`} role="listitem">
              <span className="prob-name">{names[k]}</span>
              <span className="prob-track">
                <span
                  className="prob-bar"
                  style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%` }}
                />
                {pv !== undefined && Math.abs(pv - v) > 1e-9 && (
                  <span
                    className="prob-prev"
                    style={{ left: `${Math.max(0, Math.min(1, pv)) * 100}%` }}
                  />
                )}
              </span>
              <span className="prob-value">{fmt(v, 3)}</span>
            </div>
          );
        })}
      </div>
    </Frame>
  );
}

function AttentionView({
  spec,
  ctx,
}: {
  spec: Extract<ViewSpec, { kind: 'attention' }>;
  ctx: ViewContext;
}) {
  const r = resolve(spec.source, ctx);
  const [head, setHead] = useState(0);
  const [hover, setHover] = useState<{ q: number; k: number; v: number } | null>(null);
  const attnSpan = r && r.shape[r.shape.length - 1] <= 24 ? 6 : 12;
  if (!r)
    return (
      <Missing span={attnSpan} title={spec.title} what={spec.unavailable ?? 'Not computed yet.'} />
    );
  const T = r.shape[r.shape.length - 1];
  const Tq = r.shape[r.shape.length - 2];
  const heads = r.shape.length >= 3 ? r.shape[r.shape.length - 3] : 1;
  const h = Math.min(head, heads - 1);
  const offset = h * Tq * T;
  const meta = metaOf(spec, ctx);
  const labels = (key = 'tokens') =>
    Array.isArray(meta?.[key]) ? (meta![key] as unknown[]).map(String) : undefined;
  const colTokens = labels(spec.keyTokens);
  const rowTokens = labels(spec.queryTokens);
  const cell = Math.max(8, Math.min(28, Math.floor(300 / T)));
  const scale: Scale = { mode: 'sequential', min: 0, max: 1 };
  return (
    <Frame
      span={attnSpan}
      title={spec.title}
      actions={
        heads > 1 ? (
          <span className="view-actions head-picker" role="radiogroup" aria-label="Attention head">
            {Array.from({ length: heads }, (_, i) => (
              <button
                key={i}
                type="button"
                role="radio"
                aria-checked={i === h}
                className={i === h ? 'is-on' : ''}
                onClick={() => setHead(i)}
              >
                h{i}
              </button>
            ))}
          </span>
        ) : undefined
      }
      note="Rows: query position; columns: key position; colour: softmax weight in [0, 1]. Weights show where information is read from, not why the output changed."
    >
      <div className="attn">
        <div className="attn-matrix">
          {colTokens && (
            <div
              className="attn-cols"
              style={{
                paddingLeft: rowTokens ? 56 : 0,
                gridTemplateColumns: `repeat(${T}, ${cell}px)`,
              }}
            >
              {colTokens.slice(0, T).map((t, i) => (
                <span key={i} className={hover?.k === i ? 'is-hot' : ''}>
                  {t}
                </span>
              ))}
            </div>
          )}
          <div className="attn-body">
            {rowTokens && (
              <div className="attn-rows" style={{ gridTemplateRows: `repeat(${Tq}, ${cell}px)` }}>
                {rowTokens.slice(0, Tq).map((t, i) => (
                  <span key={i} className={hover?.q === i ? 'is-hot' : ''}>
                    {t}
                  </span>
                ))}
              </div>
            )}
            <Heatmap
              data={r.data}
              offset={offset}
              rows={Tq}
              cols={T}
              cell={cell}
              scale={scale}
              label={`${spec.title} head ${h}`}
              onHover={(q, k, v) => setHover({ q, k, v })}
              onLeave={() => setHover(null)}
            />
          </div>
        </div>
        <div className="attn-side">
          <ScaleBar scale={scale} label="attention weight" width={120} />
          <p className="readout num">
            {hover ? `α[${hover.q}, ${hover.k}] = ${fmt(hover.v, 5)}` : ' '}
          </p>
        </div>
      </div>
    </Frame>
  );
}

function SeriesView({
  spec,
  ctx,
}: {
  spec: Extract<ViewSpec, { kind: 'series' }>;
  ctx: ViewContext;
}) {
  const [dim, setDim] = useState(0);
  const resolved = spec.sources.map((s) => ({ s, r: resolve(s.ref, ctx) }));
  const xs = spec.x ? resolve(spec.x, ctx) : null;
  const first = resolved.find((x) => x.r)?.r;
  if (!first)
    return <Missing span={6} title={spec.title} what={spec.unavailable ?? 'Not computed yet.'} />;
  const D = first.shape.length >= 2 ? first.shape[first.shape.length - 1] : 1;
  const d = Math.min(dim, D - 1);
  const colors = ['var(--ink)', 'var(--pos)', 'var(--neg)', 'var(--cmp)', 'var(--grad)'];
  const series = resolved
    .filter((x) => x.r)
    .map(({ s, r }, i) => {
      const Dd = r!.shape.length >= 2 ? r!.shape[r!.shape.length - 1] : 1;
      const T = r!.shape.length >= 2 ? r!.shape[r!.shape.length - 2] : r!.shape[0];
      const dd = Math.min(d, Dd - 1);
      return {
        id: s.ref,
        label: s.label,
        color: colors[i % colors.length],
        dashed: i > 0,
        points: Array.from({ length: T }, (_, t) => ({
          x: xs && t < xs.data.length ? xs.data[t] : t,
          y: r!.data[t * Dd + dd],
        })),
      };
    });
  return (
    <Frame
      span={6}
      title={spec.title}
      note={spec.note}
      actions={
        D > 1 ? (
          <label className="axis-picker">
            <span>dimension</span>
            <input
              type="number"
              min={0}
              max={D - 1}
              value={d}
              onChange={(e) => setDim(Number(e.target.value) || 0)}
            />
            <span className="muted">/ {D}</span>
          </label>
        ) : undefined
      }
    >
      <LineChart series={series} xLabel={spec.xLabel ?? 't'} height={170} logY={spec.logY} />
    </Frame>
  );
}

function ScatterView({
  spec,
  ctx,
}: {
  spec: Extract<ViewSpec, { kind: 'scatter' }>;
  ctx: ViewContext;
}) {
  const r = resolve(spec.source, ctx);
  const color = spec.colorBy ? resolve(spec.colorBy, ctx) : null;
  const field = spec.field ? resolve(spec.field, ctx) : null;
  if (!r)
    return <Missing span={6} title={spec.title} what={spec.unavailable ?? 'Not computed yet.'} />;
  const N = r.shape[0];
  const pts = Array.from({ length: N }, (_, i) => [r.data[i * 2], r.data[i * 2 + 1]] as const);
  const fixed = extentOf(spec.extent, ctx);
  let [x0, x1, y0, y1] = fixed ?? [Infinity, -Infinity, Infinity, -Infinity];
  if (!fixed) {
    for (const [x, y] of pts) {
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
    }
    const px = (x1 - x0) * 0.08 || 1;
    const py = (y1 - y0) * 0.08 || 1;
    x0 -= px;
    x1 += px;
    y0 -= py;
    y1 += py;
  }
  const S = 300;
  const sx = (x: number) => ((x - x0) / (x1 - x0)) * S;
  const sy = (y: number) => S - ((y - y0) / (y1 - y0)) * S;
  const fieldScale = field
    ? isNonNeg(field.data)
      ? rangeScale(field.data, 'sequential')
      : symmetricScale(field.data)
    : null;
  return (
    <Frame
      span={6}
      title={spec.title}
      note={
        field
          ? 'Background: model output evaluated on a grid (computed, not interpolated).'
          : `${N} points.`
      }
    >
      <div className="scatter" style={{ width: S, height: S }}>
        {field && fieldScale && (
          <div className="scatter-field">
            <Heatmap
              data={field.data}
              rows={field.shape[0]}
              cols={field.shape[1]}
              scale={fieldScale}
              cell={S / field.shape[1]}
              label="decision field"
              gridLines={false}
            />
          </div>
        )}
        <svg width={S} height={S} className="scatter-svg" role="img" aria-label={spec.title}>
          {pts.map(([x, y], i) => (
            <circle
              key={i}
              cx={sx(x)}
              cy={sy(y)}
              r={2.6}
              fill={
                color ? CLASS_COLORS[Math.round(color.data[i]) % CLASS_COLORS.length] : 'var(--ink)'
              }
              stroke="var(--plot)"
              strokeWidth={0.6}
            />
          ))}
        </svg>
      </div>
      {field && fieldScale && <ScaleBar scale={fieldScale} label="field value" />}
    </Frame>
  );
}

const NO_EDGES: [number, number][] = [];

function GraphView({
  spec,
  ctx,
}: {
  spec: Extract<ViewSpec, { kind: 'graph' }>;
  ctx: ViewContext;
}) {
  const r = resolve(spec.source, ctx);
  const meta = metaOf(spec, ctx);
  const edges = meta?.edges ?? NO_EDGES;
  const classNames = spec.classNames ?? meta?.classNames;
  const N = r?.shape[0] ?? 0;
  const pos = useMemo(() => springLayout(N, edges), [N, edges]);
  if (!r)
    return <Missing span={4} title={spec.title} what={spec.unavailable ?? 'Not computed yet.'} />;
  const C = r.shape[1] ?? 1;
  const colorRef = spec.colorBy && spec.colorBy !== 'argmax' ? resolve(spec.colorBy, ctx) : null;
  const cls = Array.from({ length: N }, (_, i) => {
    if (colorRef) return Math.round(colorRef.data[i]);
    let b = 0;
    for (let c = 1; c < C; c++) if (r.data[i * C + c] > r.data[i * C + b]) b = c;
    return b;
  });
  const S = 320;
  return (
    <Frame
      span={4}
      title={spec.title}
      note={`${N} nodes, ${edges.length} edges. Node colour: ${colorRef ? 'given labels' : 'arg max of the output row'}. Layout is a fixed spring embedding for display only.`}
    >
      <svg width={S} height={S} className="graph-svg" role="img" aria-label={spec.title}>
        {edges.map(([a, b], i) => (
          <line
            key={i}
            x1={pos[a][0] * S}
            y1={pos[a][1] * S}
            x2={pos[b][0] * S}
            y2={pos[b][1] * S}
            stroke="var(--rule-strong)"
            strokeWidth={0.8}
          />
        ))}
        {pos.map(([x, y], i) => (
          <g key={i}>
            <circle
              cx={x * S}
              cy={y * S}
              r={6}
              fill={CLASS_COLORS[cls[i] % CLASS_COLORS.length]}
              stroke="var(--plot)"
              strokeWidth={1.2}
            />
            <title>{`node ${i}: class ${classNames?.[cls[i]] ?? cls[i]}`}</title>
          </g>
        ))}
      </svg>
      {classNames && (
        <div className="legend-row">
          {classNames.map((n, i) => (
            <span key={n} className="legend-item">
              <span
                className="swatch"
                style={{ background: CLASS_COLORS[i % CLASS_COLORS.length] }}
              />
              {n}
            </span>
          ))}
        </div>
      )}
    </Frame>
  );
}

function springLayout(n: number, edges: [number, number][]): [number, number][] {
  // Deterministic Fruchterman–Reingold for display.
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const p = Array.from({ length: n }, () => [rnd(), rnd()] as [number, number]);
  const k = Math.sqrt(1 / Math.max(n, 1));
  for (let it = 0; it < 220; it++) {
    const t = 0.1 * (1 - it / 220);
    const d = p.map(() => [0, 0]);
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const dx = p[i][0] - p[j][0];
        const dy = p[i][1] - p[j][1];
        const dist = Math.max(1e-3, Math.hypot(dx, dy));
        const f = (k * k) / dist;
        d[i][0] += (dx / dist) * f;
        d[i][1] += (dy / dist) * f;
        d[j][0] -= (dx / dist) * f;
        d[j][1] -= (dy / dist) * f;
      }
    for (const [a, b] of edges) {
      const dx = p[a][0] - p[b][0];
      const dy = p[a][1] - p[b][1];
      const dist = Math.max(1e-3, Math.hypot(dx, dy));
      const f = (dist * dist) / k;
      d[a][0] -= (dx / dist) * f;
      d[a][1] -= (dy / dist) * f;
      d[b][0] += (dx / dist) * f;
      d[b][1] += (dy / dist) * f;
    }
    for (let i = 0; i < n; i++) {
      const len = Math.max(1e-9, Math.hypot(d[i][0], d[i][1]));
      p[i][0] += (d[i][0] / len) * Math.min(len, t);
      p[i][1] += (d[i][1] / len) * Math.min(len, t);
    }
  }
  const xs = p.map((q) => q[0]);
  const ys = p.map((q) => q[1]);
  const [ax, bx, ay, by] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  return p.map(([x, y]) => [
    0.06 + 0.88 * ((x - ax) / (bx - ax || 1)),
    0.06 + 0.88 * ((y - ay) / (by - ay || 1)),
  ]);
}

function MatrixView({
  spec,
  ctx,
}: {
  spec: Extract<ViewSpec, { kind: 'matrix' }>;
  ctx: ViewContext;
}) {
  const r = resolve(spec.source, ctx);
  const [hover, setHover] = useState<string>(' ');
  if (!r)
    return <Missing span={6} title={spec.title} what={spec.unavailable ?? 'Not computed yet.'} />;
  const rows = r.shape.length >= 2 ? r.shape[r.shape.length - 2] : 1;
  const cols = r.shape[r.shape.length - 1];
  const s = isNonNeg(r.data) ? rangeScale(r.data, 'sequential') : symmetricScale(r.data);
  return (
    <Frame span={6} title={spec.title} note={`${fmtShape(r.shape)}; first ${rows}×${cols} slice.`}>
      <Heatmap
        data={r.data}
        rows={rows}
        cols={cols}
        scale={s}
        maxWidth={420}
        label={spec.title}
        onHover={(i, j, v) => setHover(`[${i}, ${j}] = ${fmt(v, 5)}`)}
      />
      <div className="tensor-readout">
        <ScaleBar scale={s} />
        <span className="readout num">{hover}</span>
      </div>
    </Frame>
  );
}

function VectorView({
  spec,
  ctx,
}: {
  spec: Extract<ViewSpec, { kind: 'vector' }>;
  ctx: ViewContext;
}) {
  const r = resolve(spec.source, ctx);
  if (!r)
    return <Missing span={4} title={spec.title} what={spec.unavailable ?? 'Not computed yet.'} />;
  const n = Math.min(r.shape[r.shape.length - 1] ?? 1, 64);
  let m = 0;
  for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(r.data[i]));
  return (
    <Frame span={4} title={spec.title}>
      <div className="vector-bars">
        {Array.from({ length: n }, (_, i) => {
          const v = r.data[i];
          return (
            <div key={i} className="vbar-row">
              <span className="prob-name">{spec.labels?.[i] ?? i}</span>
              <span className="vbar-track">
                <span
                  className={`vbar ${v < 0 ? 'is-neg' : 'is-pos'}`}
                  style={{
                    width: `${(Math.abs(v) / (m || 1)) * 50}%`,
                    [v < 0 ? 'right' : 'left']: '50%',
                  }}
                />
              </span>
              <span className="prob-value">{fmt(v, 3)}</span>
            </div>
          );
        })}
      </div>
    </Frame>
  );
}

function TokensView({
  spec,
  ctx,
}: {
  spec: Extract<ViewSpec, { kind: 'tokens' }>;
  ctx: ViewContext;
}) {
  const r = resolve(spec.source, ctx);
  const [pos, setPos] = useState<number | null>(null);
  if (!r)
    return <Missing span={12} title={spec.title} what={spec.unavailable ?? 'Not computed yet.'} />;
  const meta = metaOf(spec, ctx);
  const vocab = (meta?.vocab as string[] | undefined) ?? [];
  const tokens = meta?.tokens ?? [];
  const V = r.shape[r.shape.length - 1];
  const T = r.shape[r.shape.length - 2];
  const k = spec.topK ?? 5;
  const p = pos ?? T - 1;
  const row = Array.from({ length: V }, (_, v) => ({ v, p: r.data[p * V + v] }))
    .sort((a, b) => b.p - a.p)
    .slice(0, k);
  const show = (t: string) => (t === ' ' ? '␣' : t === '\n' ? '⏎' : t);
  return (
    <Frame
      span={12}
      title={spec.title}
      note={
        spec.note ??
        'Select a position to see the model’s distribution over the next token at that position.'
      }
    >
      <div className="token-strip">
        {tokens.slice(0, T).map((t, i) => (
          <button
            key={i}
            type="button"
            className={i === p ? 'is-on' : ''}
            onClick={() => setPos(i)}
          >
            {show(t)}
          </button>
        ))}
      </div>
      <div className="probs">
        {row.map(({ v, p: pv }) => (
          <div key={v} className="prob-row">
            <span className="prob-name">{show(vocab[v] ?? String(v))}</span>
            <span className="prob-track">
              <span className="prob-bar" style={{ width: `${pv * 100}%` }} />
            </span>
            <span className="prob-value">{fmt(pv, 3)}</span>
          </div>
        ))}
      </div>
    </Frame>
  );
}

function TextView({ spec, ctx }: { spec: Extract<ViewSpec, { kind: 'text' }>; ctx: ViewContext }) {
  const [, id] = spec.source.split(':');
  const text = ctx.procedures[id]?.result.text;
  if (!text) return <Missing span={6} title={spec.title} what="Not generated yet." />;
  return (
    <Frame span={6} title={spec.title}>
      <pre className="generated">{text}</pre>
    </Frame>
  );
}

function RolloutView({
  spec,
  ctx,
}: {
  spec: Extract<ViewSpec, { kind: 'rollout' }>;
  ctx: ViewContext;
}) {
  const pred = resolve(spec.predicted, ctx);
  const obs = spec.observed ? resolve(spec.observed, ctx) : null;
  const rew = spec.reward ? resolve(spec.reward, ctx) : null;
  const rewObs = spec.rewardObserved ? resolve(spec.rewardObserved, ctx) : null;
  if (!pred) return <Missing span={12} title={spec.title} what="Not computed yet." />;
  const [T, C, H, W] =
    pred.shape.length === 4 ? pred.shape : [pred.shape[0], 1, pred.shape[1], pred.shape[2]];
  const per = C * H * W;
  const size = Math.max(H * 2, 48);
  const strip = (r: Resolved, label: string) => (
    <div className="rollout-row">
      <span className="rollout-label">{label}</span>
      {Array.from({ length: T }, (_, t) => (
        <ImageCanvas
          key={t}
          shape={[C, H, W]}
          data={r.data}
          offset={t * per}
          range={[0, 1]}
          size={size}
          label={`${label} t=${t}`}
        />
      ))}
    </div>
  );
  return (
    <Frame
      span={12}
      title={spec.title}
      note="Predicted frames are decoded from the model’s latent state; observed frames come from the environment simulator. Greyscale 0 → black, 1 → white."
    >
      <div className="rollout">
        <div className="rollout-row rollout-times">
          <span className="rollout-label" />
          {Array.from({ length: T }, (_, t) => (
            <span key={t} style={{ width: size }}>
              t={t}
            </span>
          ))}
        </div>
        {obs && strip(obs, 'observed')}
        {strip(pred, 'predicted')}
      </div>
      {rew && (
        <LineChart
          height={120}
          xLabel="t"
          series={[
            {
              id: 'r',
              label: 'predicted reward',
              color: 'var(--ink)',
              points: Array.from(rew.data, (y, x) => ({ x, y })),
            },
            ...(rewObs
              ? [
                  {
                    id: 'ro',
                    label: 'observed reward',
                    color: 'var(--pos)',
                    dashed: true,
                    points: Array.from(rewObs.data, (y, x) => ({ x, y })),
                  },
                ]
              : []),
          ]}
        />
      )}
    </Frame>
  );
}

function PhaseView({
  spec,
  ctx,
}: {
  spec: Extract<ViewSpec, { kind: 'phase' }>;
  ctx: ViewContext;
}) {
  const f = resolve(spec.field, ctx);
  const tr = spec.trajectories ? resolve(spec.trajectories, ctx) : null;
  const ext = extentOf(spec.extent, ctx);
  const cls = spec.colorBy ? resolve(spec.colorBy, ctx) : null;
  if (!f || !ext)
    return <Missing span={6} title={spec.title} what={spec.unavailable ?? 'Not computed yet.'} />;
  const [GH, GW] = f.shape;
  const [x0, x1, y0, y1] = ext;
  const S = 320;
  const sx = (x: number) => ((x - x0) / (x1 - x0)) * S;
  const sy = (y: number) => S - ((y - y0) / (y1 - y0)) * S;
  let m = 0;
  for (let i = 0; i < GH * GW; i++) m = Math.max(m, Math.hypot(f.data[i * 2], f.data[i * 2 + 1]));
  const arrows = [];
  for (let i = 0; i < GH; i++)
    for (let j = 0; j < GW; j++) {
      const x = x0 + ((j + 0.5) / GW) * (x1 - x0);
      const y = y0 + ((i + 0.5) / GH) * (y1 - y0);
      const u = f.data[(i * GW + j) * 2];
      const v = f.data[(i * GW + j) * 2 + 1];
      const len = Math.hypot(u, v) / (m || 1);
      const L = (S / GW) * 0.42 * len;
      const a = Math.atan2(-v, u);
      arrows.push(
        <line
          key={`${i},${j}`}
          x1={sx(x)}
          y1={sy(y)}
          x2={sx(x) + L * Math.cos(a)}
          y2={sy(y) + L * Math.sin(a)}
          stroke="var(--ink-3)"
          strokeWidth={1}
        />,
      );
    }
  const trajs = tr
    ? Array.from({ length: tr.shape[0] }, (_, k) => {
        const T = tr.shape[1];
        return Array.from({ length: T }, (_, t) => [
          tr.data[(k * T + t) * 2],
          tr.data[(k * T + t) * 2 + 1],
        ]);
      })
    : [];
  return (
    <Frame
      span={6}
      title={spec.title}
      note={
        spec.note ??
        'Arrows: learned dynamics f(h) on a grid, length ∝ ‖f‖ (normalised to the largest). Curves: trajectories integrated by the model’s solver.'
      }
    >
      <svg width={S} height={S} className="phase-svg" role="img" aria-label={spec.title}>
        {arrows}
        {trajs.map((pts, k) => (
          <path
            key={k}
            d={pts.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x)} ${sy(y)}`).join(' ')}
            fill="none"
            stroke={CLASS_COLORS[(cls ? Math.round(cls.data[k]) : k) % CLASS_COLORS.length]}
            strokeWidth={1.6}
          />
        ))}
      </svg>
    </Frame>
  );
}

/** Resolve a fixed or computed extent. */
function extentOf(
  e: [number, number, number, number] | string | undefined,
  ctx: ViewContext,
): [number, number, number, number] | null {
  if (e === undefined) return null;
  if (typeof e !== 'string') return e;
  const r = resolve(e, ctx);
  return r && r.data.length >= 4 ? [r.data[0], r.data[1], r.data[2], r.data[3]] : null;
}

function isNonNeg(d: ArrayLike<number>): boolean {
  for (let i = 0; i < d.length; i++) if (d[i] < 0) return false;
  return true;
}

export function OutputView({
  spec,
  ctx,
  onSelectRef,
  toolbar,
  runner,
  help,
  span,
}: {
  spec: ViewSpec;
  ctx: ViewContext;
  onSelectRef?: (ref: string) => void;
  /** A procedure's arguments and Run button, shown under the title. */
  toolbar?: ReactNode;
  /** A Run button for the empty state. */
  runner?: ReactNode;
  help?: ReactNode;
  span?: number;
}) {
  return (
    <ViewExtras.Provider value={{ toolbar, runner, help, span }}>
      <ViewBody spec={spec} ctx={ctx} onSelectRef={onSelectRef} />
    </ViewExtras.Provider>
  );
}

function ViewBody({
  spec,
  ctx,
  onSelectRef,
}: {
  spec: ViewSpec;
  ctx: ViewContext;
  onSelectRef?: (ref: string) => void;
}) {
  switch (spec.kind) {
    case 'image':
      return <ImageView spec={spec} ctx={ctx} />;
    case 'imageGrid':
      return <ImageGridView spec={spec} ctx={ctx} />;
    case 'featureMaps':
      return <FeatureMapsView spec={spec} ctx={ctx} onSelect={onSelectRef} />;
    case 'probs':
      return <ProbsView spec={spec} ctx={ctx} />;
    case 'attention':
      return <AttentionView spec={spec} ctx={ctx} />;
    case 'series':
      return <SeriesView spec={spec} ctx={ctx} />;
    case 'scatter':
      return <ScatterView spec={spec} ctx={ctx} />;
    case 'graph':
      return <GraphView spec={spec} ctx={ctx} />;
    case 'matrix':
      return <MatrixView spec={spec} ctx={ctx} />;
    case 'vector':
      return <VectorView spec={spec} ctx={ctx} />;
    case 'tokens':
      return <TokensView spec={spec} ctx={ctx} />;
    case 'text':
      return <TextView spec={spec} ctx={ctx} />;
    case 'rollout':
      return <RolloutView spec={spec} ctx={ctx} />;
    case 'phase':
      return <PhaseView spec={spec} ctx={ctx} />;
  }
}

/** Every tensor reference a view reads. */
export function viewRefs(spec: ViewSpec): string[] {
  const refs: (string | undefined)[] = [];
  switch (spec.kind) {
    case 'series':
      refs.push(...spec.sources.map((s) => s.ref), spec.x);
      break;
    case 'scatter':
      refs.push(spec.source, spec.colorBy, spec.field);
      if (typeof spec.extent === 'string') refs.push(spec.extent);
      break;
    case 'graph':
      refs.push(spec.source, spec.colorBy === 'argmax' ? undefined : spec.colorBy);
      break;
    case 'rollout':
      refs.push(spec.predicted, spec.observed, spec.reward, spec.rewardObserved);
      break;
    case 'phase':
      refs.push(spec.field, spec.trajectories, spec.colorBy);
      if (typeof spec.extent === 'string') refs.push(spec.extent);
      break;
    default:
      refs.push(spec.source);
  }
  return refs.filter((r): r is string => !!r);
}

/** The procedure whose result a view shows, if any. */
export function procedureOf(spec: ViewSpec): string | null {
  const r = viewRefs(spec).find((x) => x.startsWith('proc:'));
  return r ? r.split(':')[1] : null;
}
