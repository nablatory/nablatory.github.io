/**
 * Display colour maps. These only map computed numbers to colours; the scale
 * used is always returned so plots can print it.
 */

type RGB = [number, number, number];

function hexToRgb(hex: string): RGB {
  const h = hex.trim().replace('#', '');
  const v =
    h.length === 3
      ? h
          .split('')
          .map((c) => c + c)
          .join('')
      : h;
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
}

// sRGB ↔ OKLab for perceptually even interpolation.
function toLinear(c: number): number {
  const x = c / 255;
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}
function fromLinear(x: number): number {
  const v = x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, Math.round(v * 255)));
}
function rgbToOklab([r, g, b]: RGB): RGB {
  const lr = toLinear(r),
    lg = toLinear(g),
    lb = toLinear(b);
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}
function oklabToRgb([L, a, b]: RGB): RGB {
  const l = Math.pow(L + 0.3963377774 * a + 0.2158037573 * b, 3);
  const m = Math.pow(L - 0.1055613458 * a - 0.0638541728 * b, 3);
  const s = Math.pow(L - 0.0894841775 * a - 1.291485548 * b, 3);
  return [
    fromLinear(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
    fromLinear(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    fromLinear(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s),
  ];
}

function ramp(stops: string[], n = 256): Uint8ClampedArray {
  const labs = stops.map((s) => rgbToOklab(hexToRgb(s)));
  const out = new Uint8ClampedArray(n * 3);
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * (labs.length - 1);
    const k = Math.min(labs.length - 2, Math.floor(t));
    const f = t - k;
    const a = labs[k];
    const b = labs[k + 1];
    const rgb = oklabToRgb([
      a[0] + (b[0] - a[0]) * f,
      a[1] + (b[1] - a[1]) * f,
      a[2] + (b[2] - a[2]) * f,
    ]);
    out.set(rgb, i * 3);
  }
  return out;
}

export interface Palette {
  diverging: Uint8ClampedArray;
  sequential: Uint8ClampedArray;
  grey: Uint8ClampedArray;
  plot: string;
  ink: string;
  neg: string;
  pos: string;
  grad: string;
  cmp: string;
  rule: string;
  ink3: string;
}

let cachedKey = '';
let cached: Palette | null = null;

/** Palette built from the current CSS custom properties (follows the theme). */
export function palette(): Palette {
  const cs = getComputedStyle(document.documentElement);
  const v = (n: string) => cs.getPropertyValue(n).trim();
  const key = ['--plot', '--ink', '--neg', '--pos', '--grad', '--cmp'].map(v).join();
  if (cached && key === cachedKey) return cached;
  cachedKey = key;
  cached = {
    diverging: ramp([v('--neg'), v('--plot'), v('--pos')]),
    sequential: ramp([v('--plot'), v('--ink')]),
    grey: ramp(['#000000', '#ffffff']),
    plot: v('--plot'),
    ink: v('--ink'),
    neg: v('--neg'),
    pos: v('--pos'),
    grad: v('--grad'),
    cmp: v('--cmp'),
    rule: v('--rule'),
    ink3: v('--ink-3'),
  };
  return cached;
}

export type ScaleMode = 'diverging' | 'sequential' | 'grey';

export interface Scale {
  mode: ScaleMode;
  min: number;
  max: number;
}

/** Symmetric scale for signed data: [−m, m] with m = max |x|. */
export function symmetricScale(data: ArrayLike<number>): Scale {
  let m = 0;
  for (let i = 0; i < data.length; i++) {
    const a = Math.abs(data[i]);
    if (Number.isFinite(a) && a > m) m = a;
  }
  return { mode: 'diverging', min: -(m || 1), max: m || 1 };
}

export function rangeScale(data: ArrayLike<number>, mode: ScaleMode = 'sequential'): Scale {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo)) return { mode, min: 0, max: 1 };
  if (hi === lo) hi = lo + 1;
  return { mode, min: lo, max: hi };
}

export function colorIndex(v: number, s: Scale): number {
  if (!Number.isFinite(v)) return -1;
  const t = (v - s.min) / (s.max - s.min);
  return Math.max(0, Math.min(255, Math.round(t * 255)));
}

export function lutFor(p: Palette, mode: ScaleMode): Uint8ClampedArray {
  return mode === 'diverging' ? p.diverging : mode === 'sequential' ? p.sequential : p.grey;
}

export function cssColor(v: number, s: Scale, p = palette()): string {
  const i = colorIndex(v, s);
  if (i < 0) return 'transparent';
  const l = lutFor(p, s.mode);
  return `rgb(${l[i * 3]}, ${l[i * 3 + 1]}, ${l[i * 3 + 2]})`;
}
