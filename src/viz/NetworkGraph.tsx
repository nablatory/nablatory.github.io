import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import type { Module, ModuleEdge, ModuleGraph } from './architecture';
import { describeModule } from './architecture';
import { COMPACT, FULL, networkLayout, textWidth, visibleOps } from './networkLayout';
import type { Point } from './graphLayout';
import { colorIndex, lutFor, palette, type Scale } from './colormap';
import { fmt, fmtCount } from '../ui/format';
import { termAttrs } from '../ui/Term';
import { opTerm } from '../ui/glossary';

export interface ModuleDiff {
  maxAbs: number;
  rel: number;
}

export type GraphMode = 'values' | 'gradients';

/** Re-render when the colour theme changes (the tiles are drawn from CSS custom properties). */
export function useThemeKey(): number {
  const [key, setKey] = useState(0);
  useEffect(() => {
    const mq = matchMedia('(prefers-color-scheme: dark)');
    const fn = () => setKey((k) => k + 1);
    mq.addEventListener('change', fn);
    const obs = new MutationObserver(fn);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => {
      mq.removeEventListener('change', fn);
      obs.disconnect();
    };
  }, []);
  return key;
}

const shapeText = (s: number[]) => (s.length ? s.join('×') : 'scalar');

/** Rows × columns of the picture drawn for a tensor (first example, up to four channels). */
interface Tile {
  url: string;
  rows: number;
  cols: number;
  stretch: boolean;
}

const MAX_SIDE = 64;

/**
 * Draws one block's output (or its gradient) as a small picture, on its own scale:
 * signed data on the diverging map, non-negative data on the sequential one.
 */
function makeTile(data: ArrayLike<number>, shape: number[], signed: boolean): Tile | null {
  if (data.length <= 1 || typeof document === 'undefined') return null;
  let rows: number;
  let cols: number;
  let get: (r: number, c: number) => number;
  const n = shape.length;
  if (n === 4) {
    const [, C, H, W] = shape;
    const k = Math.min(C, 4);
    const gx = k > 1 ? 2 : 1;
    const gy = k > 2 ? 2 : 1;
    rows = gy * H + (gy - 1);
    cols = gx * W + (gx - 1);
    get = (r, c) => {
      const ty = Math.floor(r / (H + 1));
      const tx = Math.floor(c / (W + 1));
      const y = r - ty * (H + 1);
      const x = c - tx * (W + 1);
      const ch = ty * gx + tx;
      if (y >= H || x >= W || ch >= k) return NaN;
      return data[(ch * H + y) * W + x];
    };
  } else if (n === 3) {
    const [, T, D] = shape;
    rows = T;
    cols = D;
    get = (r, c) => data[r * D + c];
  } else if (n === 2 && shape[0] > 1) {
    const [N, D] = shape;
    rows = Math.min(N, 32);
    cols = D;
    get = (r, c) => data[r * D + c];
  } else {
    const D = n === 2 ? shape[1] : data.length;
    cols = Math.ceil(Math.sqrt(D));
    rows = Math.ceil(D / cols);
    get = (r, c) => (r * cols + c < D ? data[r * cols + c] : NaN);
  }
  // Block means when a side is longer than MAX_SIDE.
  const fy = Math.ceil(rows / MAX_SIDE);
  const fx = Math.ceil(cols / MAX_SIDE);
  const R = Math.ceil(rows / fy);
  const Cn = Math.ceil(cols / fx);
  const vals = new Float64Array(R * Cn);
  let lo = Infinity;
  let hi = -Infinity;
  for (let r = 0; r < R; r++)
    for (let c = 0; c < Cn; c++) {
      let s = 0;
      let k = 0;
      for (let y = r * fy; y < Math.min(rows, (r + 1) * fy); y++)
        for (let x = c * fx; x < Math.min(cols, (c + 1) * fx); x++) {
          const v = get(y, x);
          if (Number.isFinite(v)) {
            s += v;
            k++;
          }
        }
      const v = k ? s / k : NaN;
      vals[r * Cn + c] = v;
      if (Number.isFinite(v)) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    }
  if (!Number.isFinite(lo)) return null;
  const diverging = signed || lo < 0;
  const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  const scale: Scale = diverging
    ? { mode: 'diverging', min: -m, max: m }
    : { mode: 'sequential', min: 0, max: hi || 1 };
  const p = palette();
  const lut = lutFor(p, scale.mode);
  const canvas = document.createElement('canvas');
  canvas.width = Cn;
  canvas.height = R;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  const img = ctx.createImageData(Cn, R);
  for (let i = 0; i < vals.length; i++) {
    const ci = colorIndex(vals[i], scale);
    const o = i * 4;
    if (ci < 0) {
      img.data[o + 3] = 0;
      continue;
    }
    img.data[o] = lut[ci * 3];
    img.data[o + 1] = lut[ci * 3 + 1];
    img.data[o + 2] = lut[ci * 3 + 2];
    img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const aspect = R / Cn;
  return { url: canvas.toDataURL(), rows: R, cols: Cn, stretch: aspect > 4 || aspect < 0.25 };
}

/** Smooth path through the edge points: vertical S-curves between layers, straight inside them. */
function forwardPath(pts: Point[]): string {
  let d = `M${pts[0].x} ${pts[0].y}`;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1];
    const b = pts[i];
    if (Math.abs(a.x - b.x) < 0.5) d += ` L${b.x} ${b.y}`;
    else {
      const k = (b.y - a.y) * 0.5;
      d += ` C${a.x} ${a.y + k}, ${b.x} ${b.y - k}, ${b.x} ${b.y}`;
    }
  }
  return d;
}

/** Out to the lane on the right, up (or down), back into the target, with rounded corners. */
function feedbackPath(pts: Point[]): string {
  const [a, b, c, e] = pts;
  const r = Math.min(10, Math.abs(b.y - c.y) / 2);
  const dir = c.y < b.y ? -1 : 1;
  return [
    `M${a.x} ${a.y}`,
    `L${b.x - r} ${b.y}`,
    `Q${b.x} ${b.y}, ${b.x} ${b.y + dir * r}`,
    `L${c.x} ${c.y - dir * r}`,
    `Q${c.x} ${c.y}, ${c.x - r} ${c.y}`,
    `L${e.x + 2} ${e.y}`,
  ].join(' ');
}

function l2(d?: ArrayLike<number>): number {
  if (!d) return 0;
  let s = 0;
  for (let i = 0; i < d.length; i++) s += d[i] * d[i];
  return Math.sqrt(s);
}

/** The text, shortened with an ellipsis to fit `px` at the given size. */
function clip(s: string, px: number, size: number, weight = 400, mono = false): string {
  if (textWidth(s, size, weight, mono) <= px) return s;
  let lo = 0;
  let hi = s.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (textWidth(`${s.slice(0, mid).trimEnd()}…`, size, weight, mono) <= px) lo = mid;
    else hi = mid - 1;
  }
  return `${s.slice(0, Math.max(1, lo)).trimEnd()}…`;
}

/** The blocks upstream and downstream of one block (through forward edges). */
function lineage(graph: ModuleGraph, name: string): Set<string> {
  const up = new Map<string, string[]>();
  const down = new Map<string, string[]>();
  for (const e of graph.edges) {
    if (e.feedback) continue;
    (up.get(e.to) ?? up.set(e.to, []).get(e.to)!).push(e.from);
    (down.get(e.from) ?? down.set(e.from, []).get(e.from)!).push(e.to);
  }
  const seen = new Set([name]);
  for (const map of [up, down]) {
    const stack = [name];
    while (stack.length) {
      const n = stack.pop()!;
      for (const x of map.get(n) ?? [])
        if (!seen.has(x)) {
          seen.add(x);
          stack.push(x);
        }
    }
  }
  return seen;
}

export function NetworkGraph({
  graph,
  selectedModule,
  selectedNode,
  onSelectModule,
  onSelectNode,
  expanded,
  onToggleExpand,
  mode,
  diffs,
  blockNote,
}: {
  graph: ModuleGraph;
  selectedModule: string | null;
  selectedNode: number | null;
  onSelectModule: (name: string) => void;
  onSelectNode: (id: number) => void;
  expanded: ReadonlySet<string>;
  onToggleExpand: (name: string) => void;
  mode: GraphMode;
  diffs: Map<string, ModuleDiff> | null;
  /** A model-specific sentence about a block, shown in its hover explanation. */
  blockNote?: (m: Module) => string | undefined;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [avail, setAvail] = useState(0);
  const [hover, setHover] = useState<string | null>(null);
  const theme = useThemeKey();
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setAvail(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Full-size blocks when they fit; smaller blocks, then a slightly reduced drawing, when not;
  // past that the drawing scrolls sideways inside its frame.
  const { metrics, layout } = useMemo(() => {
    const full = networkLayout(graph, FULL, expanded);
    if (avail === 0 || (avail >= 560 && full.width <= avail))
      return { metrics: FULL, layout: full };
    return { metrics: COMPACT, layout: networkLayout(graph, COMPACT, expanded) };
  }, [graph, expanded, avail]);
  const fit = avail > 0 ? Math.min(1, avail / layout.width) : 1;
  const scale = fit < 0.78 ? 1 : fit;
  // A drawing wider than its frame opens centred, where the inputs usually are.
  const drawnWidth = Math.round(layout.width * scale);
  useEffect(() => {
    const el = wrapRef.current;
    if (el && el.scrollWidth > el.clientWidth)
      el.scrollLeft = (el.scrollWidth - el.clientWidth) / 2;
  }, [drawnWidth, avail]);

  const tiles = useMemo(() => {
    const out = new Map<string, Tile | null>();
    for (const m of graph.modules) {
      const src = mode === 'gradients' ? m.output.grad : m.output.value;
      out.set(m.name, src ? makeTile(src, m.output.shape, mode === 'gradients') : null);
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph, mode, theme]);

  // Edge widths: values passed (log scale), or the gradient norm at the tensor (log, rescaled).
  const gradRange = useMemo(() => {
    const vals = graph.edges.map((e) => l2(e.tensor.grad)).filter((v) => v > 0);
    if (!vals.length) return null;
    return [Math.log10(Math.min(...vals)), Math.log10(Math.max(...vals))] as const;
  }, [graph]);
  const edgeWidth = (e: ModuleEdge) => {
    if (mode === 'values')
      return Math.min(3.6, 1 + 0.55 * Math.log10(Math.max(1, e.tensor.value.length)));
    const g = l2(e.tensor.grad);
    if (!gradRange || g <= 0) return 1;
    const [lo, hi] = gradRange;
    return 1 + 3 * (hi > lo ? (Math.log10(g) - lo) / (hi - lo) : 1);
  };

  const lit = hover ? lineage(graph, layout.ghosts.get(hover) ?? hover) : null;
  const edgeOf = new Map(graph.edges.map((e) => [`${e.from}→${e.to}`, e]));
  for (const [id, src] of layout.ghosts) {
    const to = id.slice(id.indexOf('→') + 1);
    const e = edgeOf.get(`${src}→${to}`);
    if (e) edgeOf.set(`${id}→${to}`, e);
  }
  const maxW = metrics.w;
  const T = metrics.tile;

  return (
    <div className="net-scroll" ref={wrapRef}>
      <svg
        className={`net net-${mode}${lit ? ' has-hover' : ''}`}
        width={layout.width * scale}
        height={layout.height * scale}
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        role="group"
        aria-label="Computation graph"
      >
        <defs>
          <marker
            id="net-arrow"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="7"
            markerHeight="7"
            markerUnits="userSpaceOnUse"
            orient="auto-start-reverse"
          >
            <path d="M0.5 0.8L7 4L0.5 7.2z" className="net-arrowhead" />
          </marker>
        </defs>
        <g className="net-edges">
          {layout.edges.map((pe) => {
            const e = edgeOf.get(`${pe.from}→${pe.to}`);
            if (!e) return null;
            const on = !lit || (lit.has(layout.ghosts.get(pe.from) ?? pe.from) && lit.has(pe.to));
            const noGrad = mode === 'gradients' && (e.stopGrad || !e.tensor.grad);
            const cls = `net-edge${pe.feedback ? ' is-feedback' : ''}${e.stopGrad ? ' is-sg' : ''}${on ? '' : ' is-dim'}${noGrad ? ' no-grad' : ''}`;
            if (pe.feedback) {
              const [, b, c] = pe.points;
              return (
                <g key={`${pe.from}→${pe.to}`} className={cls}>
                  <path
                    d={feedbackPath(pe.points)}
                    strokeWidth={edgeWidth(e)}
                    markerEnd={mode === 'values' ? 'url(#net-arrow)' : undefined}
                    markerStart={mode === 'gradients' ? 'url(#net-arrow)' : undefined}
                  />
                  <text
                    x={b.x + 12}
                    y={(b.y + c.y) / 2}
                    transform={`rotate(90 ${b.x + 12} ${(b.y + c.y) / 2})`}
                    textAnchor="middle"
                    className="net-lane-label"
                  >
                    previous step
                  </text>
                </g>
              );
            }
            return (
              <path
                key={`${pe.from}→${pe.to}`}
                className={cls}
                d={forwardPath(pe.points)}
                strokeWidth={edgeWidth(e)}
                markerEnd={mode === 'values' ? 'url(#net-arrow)' : undefined}
                markerStart={mode === 'gradients' && !noGrad ? 'url(#net-arrow)' : undefined}
              />
            );
          })}
        </g>
        <g className="net-nodes">
          {[...layout.ghosts].map(([id, src]) => {
            const p = layout.nodes.get(id);
            const m = graph.byName.get(src);
            if (!p || !m) return null;
            const dim = lit && !lit.has(src);
            return (
              <g
                key={id}
                className={`net-ghost${dim ? ' is-dim' : ''}${selectedModule === src ? ' is-selected' : ''}`}
                transform={`translate(${p.x} ${p.y})`}
                onClick={() => onSelectModule(src)}
                onPointerEnter={() => setHover(id)}
                onPointerLeave={() => setHover((h) => (h === id ? null : h))}
                {...termAttrs('input', {
                  title: m.label,
                  note: `The same input as the block “${m.label}” at the top, used again here.`,
                })}
              >
                <rect width={p.w} height={p.h} rx={p.h / 2} className="net-ghost-box" />
                <path
                  d={`M10 ${p.h / 2 - 4}v4h6M14 ${p.h / 2 - 2}l2 2-2 2`}
                  className="net-ghost-glyph"
                />
                <text x={22} y={p.h / 2 + 4} className="net-ghost-text">
                  {m.label}
                </text>
              </g>
            );
          })}
          {graph.modules.map((m) => {
            const p = layout.nodes.get(m.name);
            if (!p) return null;
            const sel = selectedModule === m.name;
            const dim = lit && !lit.has(m.name);
            const isExp = expanded.has(m.name);
            const diff = diffs?.get(m.name);
            const tile = tiles.get(m.name) ?? null;
            const desc = describeModule(m);
            const label = m.label;
            const grad = l2(m.output.grad);
            const note = [
              `${desc} → ${shapeText(m.output.shape)}`,
              m.paramCount ? `${fmtCount(m.paramCount)} parameters` : '',
              mode === 'gradients'
                ? m.output.grad
                  ? `‖∂L/∂y‖ = ${fmt(grad, 3)}`
                  : 'not on a path to the loss: no gradient'
                : '',
              diff && diff.maxAbs > 0 ? `changed by up to ${fmt(diff.maxAbs, 2)}` : '',
              m.residual
                ? 'Its input is added back to what it computes (a residual connection)'
                : '',
              blockNote?.(m) ?? '',
            ]
              .filter(Boolean)
              .join('. ');
            const common = {
              className: `net-node role-${m.role}${sel ? ' is-selected' : ''}${dim ? ' is-dim' : ''}`,
              transform: `translate(${p.x} ${p.y})`,
              tabIndex: 0,
              role: 'button',
              'aria-pressed': sel,
              'aria-label': `${label}: ${desc}, output ${shapeText(m.output.shape)}`,
              onClick: () => onSelectModule(m.name),
              onDoubleClick: () => m.ops.length > 1 && onToggleExpand(m.name),
              onPointerEnter: () => setHover(m.name),
              onPointerLeave: () => setHover((h) => (h === m.name ? null : h)),
              onKeyDown: (e: KeyboardEvent) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onSelectModule(m.name);
                }
                if (e.key === '+' || e.key === '-') onToggleExpand(m.name);
              },
              ...termAttrs(m.category, { title: label, note }),
            };

            if (m.role === 'junction' && !isExp) {
              const r = metrics.join / 2;
              return (
                <g key={m.name} {...common}>
                  <circle cx={r} cy={r} r={r} className="net-join" />
                  <JoinGlyph kind={m.category} r={r} />
                  <text x={metrics.join + 8} y={r - 2} className="net-name">
                    {clip(label, p.w - metrics.join - 8, 12.5, 600)}
                  </text>
                  <text x={metrics.join + 8} y={r + 12} className="net-desc">
                    {clip(desc, p.w - metrics.join - 8, 11)}
                  </text>
                  {diff && diff.maxAbs > 0 && mode === 'values' && (
                    <circle cx={r + r * 0.72} cy={r - r * 0.72} r={3.5} className="net-changed" />
                  )}
                </g>
              );
            }

            const head = m.role === 'objective' ? metrics.hObjective : metrics.h;
            const hasTile = m.role !== 'objective';
            const tx = hasTile ? 8 + T + 10 : 14;
            const expandable = m.ops.length > 1;
            // The name shares its row with the + button; the two lines below it run the full width.
            const nameW = maxW - tx - (expandable ? 24 : 10);
            const lineW = maxW - tx - 10;
            const y0 = head / 2;
            // A loss block shows its value (the same forward pass in both modes).
            const scalar =
              m.role === 'objective' && m.output.value.length === 1
                ? fmt(m.output.value[0], 4)
                : null;
            const scalarRight = p.w - (expandable ? 30 : 14);
            const scalarW = scalar ? textWidth(scalar, 13, 500, true) + 12 : 0;
            const shape = shapeText(m.output.shape);
            const withParams = m.paramCount ? `${shape} · ${fmtCount(m.paramCount)} params` : shape;
            const meta =
              mode === 'gradients'
                ? m.output.grad
                  ? `‖∂L/∂y‖ ${fmt(grad, 2)}`
                  : 'no gradient'
                : textWidth(withParams, 10.5, 400, true) <= lineW
                  ? withParams
                  : shape;
            return (
              <g key={m.name} {...common}>
                <rect width={p.w} height={p.h} rx={8} className="net-box" />
                {m.residual && (
                  <path
                    d={`M0 12C-12 14 -12 ${head - 14} -1 ${head - 12}`}
                    className="net-residual"
                    markerEnd={mode === 'values' ? 'url(#net-arrow)' : undefined}
                    markerStart={mode === 'gradients' ? 'url(#net-arrow)' : undefined}
                  />
                )}
                {diff && diff.maxAbs > 0 && mode === 'values' && (
                  <rect x={0} y={8} width={3} height={head - 16} rx={1.5} className="net-changed" />
                )}
                {hasTile && (
                  <g transform={`translate(8 ${(head - T) / 2})`}>
                    <rect width={T} height={T} rx={3} className="net-tile-bg" />
                    {tile ? (
                      <image
                        href={tile.url}
                        x={1}
                        y={1}
                        width={T - 2}
                        height={T - 2}
                        preserveAspectRatio={tile.stretch ? 'none' : 'xMidYMid meet'}
                        className="net-tile"
                      />
                    ) : (
                      <text x={T / 2} y={T / 2 + 4} textAnchor="middle" className="net-tile-text">
                        {mode === 'gradients'
                          ? m.output.grad
                            ? m.output.grad.length === 1
                              ? fmt(m.output.grad[0], 2)
                              : ''
                            : '—'
                          : m.output.value.length === 1
                            ? fmt(m.output.value[0], 2)
                            : ''}
                      </text>
                    )}
                    <rect width={T} height={T} rx={3} className="net-tile-frame" />
                  </g>
                )}
                {m.role === 'objective' && (
                  <text x={14} y={y0 + 5} className="net-loss-glyph">
                    ℒ
                  </text>
                )}
                <text
                  x={m.role === 'objective' ? 36 : tx}
                  y={m.role === 'objective' ? y0 - 3 : y0 - 9}
                  className="net-name"
                >
                  {clip(
                    label,
                    m.role === 'objective' ? scalarRight - scalarW - 36 : nameW,
                    13,
                    600,
                  )}
                </text>
                <text
                  x={m.role === 'objective' ? 36 : tx}
                  y={m.role === 'objective' ? y0 + 12 : y0 + 5}
                  className="net-desc"
                >
                  {clip(desc, m.role === 'objective' ? scalarRight - scalarW - 36 : lineW, 11.5)}
                </text>
                {m.role !== 'objective' && (
                  <text x={tx} y={y0 + 19} className="net-meta">
                    {clip(meta, lineW, 10.5, 400, true)}
                  </text>
                )}
                {scalar !== null && (
                  <text x={scalarRight} y={y0 + 5} textAnchor="end" className="net-scalar">
                    {scalar}
                  </text>
                )}
                {expandable && (
                  <g
                    className="net-expand"
                    transform={`translate(${p.w - 22} 6)`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleExpand(m.name);
                    }}
                  >
                    <rect width={16} height={16} rx={4} />
                    <path
                      d={isExp ? 'M4.5 8h7' : 'M4.5 8h7M8 4.5v7'}
                      stroke="currentColor"
                      strokeWidth="1.3"
                    />
                    <title>
                      {isExp ? 'Hide operations' : `List its ${m.ops.length} operations`}
                    </title>
                  </g>
                )}
                {isExp &&
                  visibleOps(m).map((n, i) => (
                    <g
                      key={n.id}
                      transform={`translate(8 ${head + 2 + i * metrics.opRow})`}
                      className={`net-op${selectedNode === n.id ? ' is-selected' : ''}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectNode(n.id);
                      }}
                      {...termAttrs(n.op === 'param' ? 'parameters' : opTerm(n.op), {
                        title: n.op === 'param' ? n.param : n.op,
                      })}
                    >
                      <rect width={p.w - 16} height={metrics.opRow - 2} rx={3} />
                      <text x={6} y={12}>
                        {clip(n.op === 'param' ? (n.param ?? 'param') : n.op, p.w - 90, 11)}
                      </text>
                      <text x={p.w - 22} y={12} textAnchor="end" className="net-op-dim">
                        {shapeText(n.shape)}
                      </text>
                    </g>
                  ))}
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}

function JoinGlyph({ kind, r }: { kind: string; r: number }) {
  const k = r * 0.42;
  const d =
    kind === 'concat'
      ? `M${r - 3} ${r - k}V${r + k}M${r + 3} ${r - k}V${r + k}`
      : kind === 'mul'
        ? `M${r - k * 0.75} ${r - k * 0.75}L${r + k * 0.75} ${r + k * 0.75}M${r + k * 0.75} ${r - k * 0.75}L${r - k * 0.75} ${r + k * 0.75}`
        : `M${r - k} ${r}H${r + k}M${r} ${r - k}V${r + k}`;
  return <path d={d} className="net-join-glyph" />;
}

/** What the widths, tiles and marks of the graph mean. */
export function NetworkLegend({
  mode,
  feedback,
  stopGrad,
  residual,
}: {
  mode: GraphMode;
  feedback: boolean;
  stopGrad: boolean;
  residual: boolean;
}) {
  return (
    <ul className="net-legend" aria-label="Legend">
      <li>
        <span className="net-legend-tile" aria-hidden="true">
          <span />
        </span>
        {mode === 'values'
          ? 'Tile: the block’s output for the first example (up to four channels), on its own scale'
          : 'Tile: ∂L/∂y at the block’s output, on its own scale'}
      </li>
      <li>
        <svg width="34" height="12" aria-hidden="true">
          <path d="M1 3H33" className="net-legend-line" strokeWidth="1" />
          <path d="M1 9H33" className="net-legend-line" strokeWidth="3" />
        </svg>
        {mode === 'values'
          ? 'Line width: number of values passed (log scale)'
          : 'Line width: ‖∂L/∂y‖ of the tensor passed (log scale); arrows point backwards'}
      </li>
      {feedback && (
        <li>
          <svg width="28" height="10" aria-hidden="true">
            <path d="M1 5H27" className="net-legend-line" strokeDasharray="4 3" strokeWidth="1.4" />
          </svg>
          State passed from one time step to the next
        </li>
      )}
      {stopGrad && (
        <li>
          <svg width="28" height="10" aria-hidden="true">
            <path d="M1 5H27" className="net-legend-line" strokeDasharray="1 4" strokeWidth="1.6" />
          </svg>
          Value passed on with its gradient stopped
        </li>
      )}
      {residual && (
        <li>
          <svg width="14" height="18" aria-hidden="true">
            <path d="M12 2C2 3 2 15 12 16" className="net-legend-line" strokeWidth="1.4" />
          </svg>
          Input added back inside the block (residual connection)
        </li>
      )}
      <li>
        <span className="net-legend-plus" aria-hidden="true">
          +
        </span>
        Lists a block’s operations
      </li>
    </ul>
  );
}
