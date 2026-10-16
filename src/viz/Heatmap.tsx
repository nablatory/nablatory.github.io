import { useEffect, useRef, useState } from 'react';
import { colorIndex, lutFor, palette, type Scale } from './colormap';

export interface HeatmapProps {
  data: ArrayLike<number>;
  offset?: number;
  rows: number;
  cols: number;
  scale: Scale;
  /** CSS pixels per cell; defaults to fit `maxWidth`. */
  cell?: number;
  maxWidth?: number;
  maxHeight?: number;
  selected?: [number, number] | null;
  /** Flat indices (within this slice) to outline, e.g. user edits. */
  marked?: ReadonlySet<number>;
  onHover?: (r: number, c: number, v: number) => void;
  onLeave?: () => void;
  onPick?: (r: number, c: number, v: number) => void;
  label: string;
  /** Draw thin separators between cells when cells are large enough. */
  gridLines?: boolean;
  /** Number of cells holding values (row-major); cells past it are left empty. */
  count?: number;
}

/** Pixel-exact canvas rendering of a 2-D slice. One cell = one tensor element. */
export function Heatmap({
  data,
  offset = 0,
  rows,
  cols,
  scale,
  cell,
  maxWidth = 320,
  maxHeight = 320,
  selected,
  marked,
  onHover,
  onLeave,
  onPick,
  label,
  gridLines = true,
  count,
}: HeatmapProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [theme, setTheme] = useState(0);
  const size =
    cell ?? Math.max(1, Math.min(32, Math.floor(Math.min(maxWidth / cols, maxHeight / rows))));
  const w = cols * size;
  const h = rows * size;

  useEffect(() => {
    const mq = matchMedia('(prefers-color-scheme: dark)');
    const fn = () => setTheme((t) => t + 1);
    mq.addEventListener('change', fn);
    const obs = new MutationObserver(fn);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => {
      mq.removeEventListener('change', fn);
      obs.disconnect();
    };
  }, []);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const p = palette();
    const lut = lutFor(p, scale.mode);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d')!;
    const img = new ImageData(cols, rows);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const v = data[offset + r * cols + c];
        const i = colorIndex(v, scale);
        const o = (r * cols + c) * 4;
        if (count !== undefined && r * cols + c >= count) {
          img.data[o + 3] = 0;
          continue;
        }
        if (i < 0) {
          // Non-finite values are drawn as a hatched mid-grey.
          const hatch = (r + c) % 2 === 0;
          img.data[o] = img.data[o + 1] = img.data[o + 2] = hatch ? 128 : 96;
        } else {
          img.data[o] = lut[i * 3];
          img.data[o + 1] = lut[i * 3 + 1];
          img.data[o + 2] = lut[i * 3 + 2];
        }
        img.data[o + 3] = 255;
      }
    const off = new OffscreenCanvas(cols, rows);
    off.getContext('2d')!.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(off, 0, 0, w, h);
    if (gridLines && size >= 10) {
      ctx.strokeStyle = p.rule;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1 / dpr;
      ctx.beginPath();
      for (let c = 1; c < cols; c++) {
        ctx.moveTo(c * size, 0);
        ctx.lineTo(c * size, h);
      }
      for (let r = 1; r < rows; r++) {
        ctx.moveTo(0, r * size);
        ctx.lineTo(w, r * size);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    if (marked?.size) {
      ctx.strokeStyle = p.grad;
      ctx.lineWidth = Math.max(1, size / 8);
      for (const k of marked) {
        const r = Math.floor(k / cols);
        const c = k % cols;
        if (r < rows) ctx.strokeRect(c * size + 0.5, r * size + 0.5, size - 1, size - 1);
      }
    }
    if (selected) {
      ctx.strokeStyle = p.ink;
      ctx.lineWidth = Math.max(1.5, size / 6);
      ctx.strokeRect(selected[1] * size + 0.75, selected[0] * size + 0.75, size - 1.5, size - 1.5);
    }
  }, [data, offset, rows, cols, scale, size, w, h, selected, marked, gridLines, theme, count]);

  const locate = (e: React.PointerEvent<HTMLCanvasElement>): [number, number] | null => {
    const rect = e.currentTarget.getBoundingClientRect();
    const c = Math.floor(((e.clientX - rect.left) / rect.width) * cols);
    const r = Math.floor(((e.clientY - rect.top) / rect.height) * rows);
    if (count !== undefined && r * cols + c >= count) return null;
    return r >= 0 && r < rows && c >= 0 && c < cols ? [r, c] : null;
  };

  return (
    <canvas
      ref={ref}
      className="heatmap"
      style={{ width: w, height: h }}
      role="img"
      aria-label={label}
      onPointerMove={(e) => {
        const at = locate(e);
        if (at && onHover) onHover(at[0], at[1], data[offset + at[0] * cols + at[1]]);
      }}
      onPointerLeave={onLeave}
      onClick={(e) => {
        const at = locate(e as unknown as React.PointerEvent<HTMLCanvasElement>);
        if (at && onPick) onPick(at[0], at[1], data[offset + at[0] * cols + at[1]]);
      }}
    />
  );
}
