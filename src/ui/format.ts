/** Number formatting for readouts: 4 significant digits, scientific outside [1e-3, 1e5). */
export function fmt(v: number, digits = 4): string {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'NaN' : v > 0 ? '∞' : '−∞';
  if (v === 0) return '0';
  const a = Math.abs(v);
  let s: string;
  if (a >= 1e5 || a < 1e-3) {
    const [m, e] = v.toExponential(digits - 1).split('e');
    s = `${trimZeros(m)}e${e.replace('+', '')}`;
  } else {
    s = trimZeros(v.toPrecision(digits));
  }
  return s.replace('-', '−');
}

function trimZeros(s: string): string {
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

export function fmtShape(shape: readonly number[]): string {
  return shape.length ? shape.join(' × ') : 'scalar';
}

export function fmtCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)} k`;
  return n.toLocaleString('en-US');
}

export function fmtMs(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
  if (ms < 100) return `${ms.toFixed(1)} ms`;
  if (ms < 10000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function fmtBytes(n: number): string {
  if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(1)} MB`;
  if (n >= 1 << 10) return `${(n / (1 << 10)).toFixed(0)} kB`;
  return `${n} B`;
}

/** Parse user input such as "1e-3", "−0.5", "0.25". */
export function parseNumber(s: string): number | null {
  const t = s.trim().replace('−', '-').replace(',', '.');
  if (!t) return null;
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}

/** "Y. LeCun, L. Bottou, …" + 1998 → "LeCun et al. 1998"; two authors → "Kipf and Welling 2017". */
export function shortCite(r: { authors: string; year: number }): string {
  const names = r.authors
    .split(/,| and /)
    .map((a) => a.trim())
    .filter(Boolean)
    .map((a) => a.split(/\s+/).pop()!);
  const who =
    names.length > 2
      ? `${names[0]} et al.`
      : names.length === 2
        ? `${names[0]} and ${names[1]}`
        : names[0];
  return `${who} ${r.year}`;
}

const METRIC_LABELS: Record<string, string> = {
  agreementAccuracy: 'agreement accuracy',
  bitsPerChar: 'bits per character',
  corruptedMse: 'MSE of the corrupted input',
  dLoss: 'D loss',
  gLoss: 'G loss',
  imaginedReturn: 'imagined return',
  iou: 'IoU',
  kl: 'KL',
  labelledAccuracy: 'labelled accuracy',
  meanT: 'mean t',
  modesCovered: 'modes covered',
  mse: 'MSE',
  negElbo: '−ELBO',
  pixelAccuracy: 'pixel accuracy',
  pixelMse: 'pixel MSE',
  policyQ: 'policy Q',
  recon: 'reconstruction',
  selectedPerSequence: 'selected per sequence',
  sequenceAccuracy: 'sequence accuracy',
  unlabelledAccuracy: 'unlabelled accuracy',
  valueLoss: 'value loss',
};

/** Display name of a metric key: "pixelMse" → "pixel MSE"; unknown camelCase keys are split into words. */
export function metricLabel(key: string): string {
  return (
    METRIC_LABELS[key] ??
    (/^[a-z]+(?:[A-Z][a-z]+)+$/.test(key)
      ? key.replace(/[A-Z]/g, (c) => ` ${c.toLowerCase()}`)
      : key)
  );
}
