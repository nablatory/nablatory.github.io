import { useEffect, useMemo, useRef, useState } from 'react';
import { fmt } from '../ui/format';

export interface LineSeries {
  id: string;
  label: string;
  color: string;
  points: { x: number; y: number }[];
  dashed?: boolean;
  /** Draw points as markers (sparse series such as validation loss). */
  markers?: boolean;
}

/** Evenly spaced "nice" ticks; integer spacing when `integer`. */
function niceTicks(
  lo: number,
  hi: number,
  n = 4,
  integer = false,
): { ticks: number[]; step: number } {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  let step = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= n) ?? mag * 10;
  if (integer) step = Math.max(1, Math.round(step));
  const out: number[] = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * span; v += step)
    out.push(+v.toPrecision(12));
  return { ticks: out, step };
}

/** Tick label with the precision of the tick spacing, so neighbours never repeat. */
function fmtTick(v: number, step: number): string {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-4) return fmt(v, 2);
  const decimals = Math.max(0, Math.min(8, -Math.floor(Math.log10(step) + 1e-9)));
  const s = v.toFixed(decimals);
  return (decimals > 0 ? s.replace(/\.?0+$/, '') : s).replace('-', '−');
}

/** Ticks for a log axis given log10 bounds: decades, or 1–2–5 steps when the span is short. */
function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  const mults = hi - lo > 2.5 ? [1] : hi - lo > 1.2 ? [1, 3] : [1, 2, 5];
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++)
    for (const m of mults) {
      const v = m * Math.pow(10, e);
      const l = Math.log10(v);
      if (l >= lo && l <= hi) out.push(+v.toPrecision(6));
    }
  if (out.length < 2)
    return niceTicks(Math.pow(10, lo), Math.pow(10, hi), 3).ticks.filter((v) => v > 0);
  return out;
}

/** Width of the element, tracked with a ResizeObserver. */
function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export function LineChart({
  series,
  width,
  height = 180,
  xLabel,
  logY = false,
  yLabel,
}: {
  series: LineSeries[];
  /** Fixed width; fills the container when omitted. */
  width?: number;
  height?: number;
  xLabel?: string;
  yLabel?: string;
  logY?: boolean;
}) {
  const [hoverX, setHoverX] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [wrapRef, measured] = useWidth<HTMLDivElement>();
  const W0 = width ?? (measured || 480);
  // Axis names sit outside the plot: the y name above it, the x name under the ticks.
  const m = { l: 52, r: 12, t: yLabel ? 24 : 10, b: xLabel ? 38 : 24 };
  const W = Math.max(40, W0 - m.l - m.r);
  const H = height - m.t - m.b;

  const dom = useMemo(() => {
    let x0 = Infinity,
      x1 = -Infinity,
      y0 = Infinity,
      y1 = -Infinity;
    let integerX = true;
    for (const s of series)
      for (const p of s.points) {
        const y = logY ? Math.log10(Math.max(p.y, 1e-12)) : p.y;
        if (!Number.isFinite(y)) continue;
        if (!Number.isInteger(p.x)) integerX = false;
        x0 = Math.min(x0, p.x);
        x1 = Math.max(x1, p.x);
        y0 = Math.min(y0, y);
        y1 = Math.max(y1, y);
      }
    if (!Number.isFinite(x0)) return null;
    if (x1 === x0) x1 = x0 + 1;
    if (y1 === y0) {
      const pad = Math.abs(y0) * 0.05 || 0.5;
      y1 += pad;
      y0 -= pad;
    }
    const pad = (y1 - y0) * 0.06;
    return { x0, x1, y0: y0 - pad, y1: y1 + pad, integerX };
  }, [series, logY]);

  if (!dom)
    return (
      <div className="chart-empty muted" style={{ height }} ref={wrapRef}>
        No data yet
      </div>
    );
  const sx = (x: number) => m.l + ((x - dom.x0) / (dom.x1 - dom.x0)) * W;
  const sy = (y: number) => {
    const v = logY ? Math.log10(Math.max(y, 1e-12)) : y;
    return m.t + H - ((v - dom.y0) / (dom.y1 - dom.y0)) * H;
  };
  const yTicks = logY
    ? logTicks(dom.y0, dom.y1).map((v) => ({ v, label: fmt(v, 3) }))
    : (() => {
        const { ticks, step } = niceTicks(dom.y0, dom.y1, Math.max(2, Math.round(H / 45)));
        return ticks.map((v) => ({ v, label: fmtTick(v, step) }));
      })();
  const { ticks: xt, step: xStep } = niceTicks(
    dom.x0,
    dom.x1,
    Math.max(2, Math.round(W / 90)),
    dom.integerX,
  );

  const nearest = (s: LineSeries, x: number) => {
    let best: { x: number; y: number } | null = null;
    for (const p of s.points) if (!best || Math.abs(p.x - x) < Math.abs(best.x - x)) best = p;
    return best;
  };

  return (
    <div className="linechart" ref={wrapRef}>
      <svg
        ref={svgRef}
        width={W0}
        height={height}
        role="img"
        aria-label={`${series.map((s) => s.label).join(', ')} versus ${xLabel ?? 'x'}`}
        onPointerMove={(e) => {
          const r = svgRef.current!.getBoundingClientRect();
          const px = e.clientX - r.left;
          if (px < m.l || px > m.l + W) return setHoverX(null);
          setHoverX(dom.x0 + ((px - m.l) / W) * (dom.x1 - dom.x0));
        }}
        onPointerLeave={() => setHoverX(null)}
      >
        {yTicks.map(({ v, label }) => (
          <g key={`y${v}`}>
            <line
              x1={m.l}
              x2={m.l + W}
              y1={sy(v)}
              y2={sy(v)}
              stroke="var(--line)"
              strokeWidth={1}
            />
            <text x={m.l - 8} y={sy(v) + 3.5} textAnchor="end" className="axis-text">
              {label}
            </text>
          </g>
        ))}
        <line
          x1={m.l}
          x2={m.l + W}
          y1={m.t + H}
          y2={m.t + H}
          stroke="var(--line-strong)"
          strokeWidth={1}
        />
        {xt.map((v) => (
          <text key={`x${v}`} x={sx(v)} y={m.t + H + 15} textAnchor="middle" className="axis-text">
            {fmtTick(v, xStep)}
          </text>
        ))}
        {series.map((s) =>
          s.markers ? (
            <g key={s.id}>
              {s.points.map((p, i) => (
                <circle
                  key={i}
                  cx={sx(p.x)}
                  cy={sy(p.y)}
                  r={2.8}
                  fill="var(--surface)"
                  stroke={s.color}
                  strokeWidth={1.5}
                />
              ))}
            </g>
          ) : (
            <path
              key={s.id}
              d={s.points
                .map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`)
                .join(' ')}
              fill="none"
              stroke={s.color}
              strokeWidth={1.6}
              strokeDasharray={s.dashed ? '5 4' : undefined}
              strokeLinejoin="round"
            />
          ),
        )}
        {hoverX !== null && (
          <line
            x1={sx(hoverX)}
            x2={sx(hoverX)}
            y1={m.t}
            y2={m.t + H}
            stroke="var(--ink-3)"
            strokeWidth={1}
          />
        )}
        {xLabel && (
          <text x={m.l + W / 2} y={height - 4} textAnchor="middle" className="axis-text axis-label">
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text x={4} y={13} className="axis-text axis-label">
            {yLabel}
            {logY ? ', log scale' : ''}
          </text>
        )}
      </svg>
      <div className="chart-legend">
        {series.map((s) => {
          const p = hoverX !== null ? nearest(s, hoverX) : s.points[s.points.length - 1];
          return (
            <span key={s.id} className="legend-item">
              <svg width="16" height="8" aria-hidden="true">
                {s.markers ? (
                  <circle
                    cx="8"
                    cy="4"
                    r="2.8"
                    fill="var(--surface)"
                    stroke={s.color}
                    strokeWidth="1.5"
                  />
                ) : (
                  <line
                    x1="0"
                    x2="16"
                    y1="4"
                    y2="4"
                    stroke={s.color}
                    strokeWidth="1.8"
                    strokeDasharray={s.dashed ? '5 4' : undefined}
                  />
                )}
              </svg>
              <span className="legend-label">{s.label}</span>
              {p && (
                <span className="legend-value num">
                  {fmt(p.y)}
                  <span className="muted">
                    {' '}
                    at {xLabel ?? 'x'} {fmt(p.x, 5)}
                  </span>
                </span>
              )}
            </span>
          );
        })}
      </div>
    </div>
  );
}
