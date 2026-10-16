import { useMemo } from 'react';
import { fmt } from '../ui/format';

export interface HistSeries {
  data: ArrayLike<number>;
  color: string;
  label: string;
}

/** Overlaid histograms on a shared bin grid (counts normalised to fractions). */
export function Histogram({
  series,
  bins = 40,
  width = 300,
  height = 90,
  symmetric = false,
}: {
  series: HistSeries[];
  bins?: number;
  width?: number;
  height?: number;
  symmetric?: boolean;
}) {
  const h = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of series)
      for (let i = 0; i < s.data.length; i++) {
        const v = s.data[i];
        if (!Number.isFinite(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    if (!Number.isFinite(lo)) return null;
    if (symmetric) {
      const m = Math.max(Math.abs(lo), Math.abs(hi));
      lo = -m;
      hi = m;
    }
    if (hi === lo) {
      hi += 0.5;
      lo -= 0.5;
    }
    const counts = series.map((s) => {
      const c = new Float64Array(bins);
      let n = 0;
      for (let i = 0; i < s.data.length; i++) {
        const v = s.data[i];
        if (!Number.isFinite(v)) continue;
        const b = Math.min(bins - 1, Math.floor(((v - lo) / (hi - lo)) * bins));
        c[b]++;
        n++;
      }
      for (let b = 0; b < bins; b++) c[b] /= n || 1;
      return c;
    });
    const peak = Math.max(...counts.map((c) => Math.max(...c)), 1e-9);
    return { lo, hi, counts, peak };
  }, [series, bins, symmetric]);

  if (!h) return <div className="muted">No finite values</div>;
  const pad = 14;
  const bw = width / bins;
  const plotH = height - pad;
  const zeroX = h.lo < 0 && h.hi > 0 ? ((0 - h.lo) / (h.hi - h.lo)) * width : null;
  return (
    <svg
      className="histogram"
      width={width}
      height={height}
      role="img"
      aria-label={`Histogram of ${series.map((s) => s.label).join(', ')}`}
    >
      {h.counts.map((c, si) => {
        let d = `M0 ${plotH}`;
        for (let b = 0; b < bins; b++) {
          const y = plotH - (c[b] / h.peak) * (plotH - 2);
          d += ` L${b * bw} ${y} L${(b + 1) * bw} ${y}`;
        }
        d += ` L${width} ${plotH}`;
        const s = series[si];
        return (
          <path
            key={si}
            d={d}
            fill={series.length === 1 ? s.color : 'none'}
            fillOpacity={0.28}
            stroke={s.color}
            strokeWidth={1.2}
          />
        );
      })}
      {zeroX !== null && (
        <line
          x1={zeroX}
          x2={zeroX}
          y1={0}
          y2={plotH}
          stroke="var(--rule-strong)"
          strokeDasharray="2 2"
        />
      )}
      <line x1={0} x2={width} y1={plotH} y2={plotH} stroke="var(--rule-strong)" />
      <text x={0} y={height - 2} className="axis-text">
        {fmt(h.lo, 3)}
      </text>
      <text x={width} y={height - 2} textAnchor="end" className="axis-text">
        {fmt(h.hi, 3)}
      </text>
    </svg>
  );
}
