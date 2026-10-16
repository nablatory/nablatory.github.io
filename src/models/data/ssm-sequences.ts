import { Rng, Tensor } from '../../engine';
import type { Batch, Config, Dataset } from '../types';

// ---------------------------------------------------------------------------
// System identification for the linear SSM: white-noise input u, target
// y = h ∗ u for a fixed causal filter h. A linear time-invariant model can
// match y exactly only by learning a kernel K equal to h.
// ---------------------------------------------------------------------------

export type FilterKind = 'resonator' | 'delay' | 'average';

export const FILTER_OPTIONS: { value: FilterKind; label: string }[] = [
  { value: 'resonator', label: 'Damped resonator' },
  { value: 'delay', label: 'Pure delay' },
  { value: 'average', label: 'Moving average' },
];

type FilterConfig = Config & { filter: FilterKind; lag: number; length: number };

/** The target impulse response h_0 … h_{L−1}. */
export function targetKernel(c: { filter: FilterKind; lag: number; length: number }): Float64Array {
  const L = c.length;
  const h = new Float64Array(L);
  for (let k = 0; k < L; k++) {
    if (c.filter === 'resonator') h[k] = Math.exp(-k / c.lag) * Math.sin((2 * Math.PI * k) / c.lag);
    else if (c.filter === 'delay') h[k] = k === c.lag ? 1 : 0;
    else h[k] = k < c.lag ? 1 / c.lag : 0;
  }
  return h;
}

export function filterDescription(c: { filter: FilterKind; lag: number }): string {
  if (c.filter === 'resonator') return `h_k = e^{−k/${c.lag}} sin(2πk/${c.lag})`;
  if (c.filter === 'delay') return `h_k = 1 at k = ${c.lag}, else 0 (y_t = u_{t−${c.lag}})`;
  return `h_k = 1/${c.lag} for k < ${c.lag} (mean of the last ${c.lag} inputs)`;
}

function filterBatch(c: FilterConfig, rng: Rng, B: number): Batch {
  const L = c.length;
  const h = targetKernel(c);
  const u = Tensor.zeros([B, L, 1]);
  const y = Tensor.zeros([B, L, 1]);
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < L; t++) u.data[b * L + t] = rng.normal();
    for (let t = 0; t < L; t++) {
      let s = 0;
      for (let k = 0; k <= t; k++) s += h[k] * u.data[b * L + t - k];
      y.data[b * L + t] = s;
    }
  }
  return {
    inputs: { u },
    targets: { y },
    meta: { caption: `White-noise input; target ${filterDescription(c)}.` },
  };
}

export function filterDataset<C extends FilterConfig>(): Dataset<C> {
  return {
    id: 'ssm-filter',
    name: 'Filtered white noise',
    description:
      'Input: Gaussian white noise. Target: the input passed through a fixed causal filter (damped resonator, pure delay or moving average). The task is to recover the filter.',
    source: 'synthetic',
    train: (rng, B, c) => filterBatch(c, rng, B),
    validation: (c) => filterBatch(c, new Rng(7001), 128),
    exampleCount: 6,
    example: (i, c) => filterBatch(c, new Rng(300 + i), 1),
    exampleLabel: (i) => `Noise sequence ${i + 1}`,
  };
}

// ---------------------------------------------------------------------------
// Selective copying (Gu & Dao 2023, §4.1.1).
// m data tokens are placed in a span of blanks; m query tokens follow; the
// model must emit the data tokens in order at the query positions.
// ---------------------------------------------------------------------------

export const COPY_SYMBOLS = ['A', 'B', 'C', 'D'];
export const COPY_VOCAB = [...COPY_SYMBOLS, '·', '?'];
const BLANK = COPY_SYMBOLS.length;
const QUERY = BLANK + 1;

type CopyConfig = Config & { span: number; memorize: number };

export function copyBatch(c: CopyConfig, rng: Rng, B: number): Batch {
  const m = c.memorize;
  const L = c.span + m;
  const ids = new Array<number>(B * L);
  const labels: number[] = [];
  for (let b = 0; b < B; b++) {
    const positions = rng
      .shuffle(Array.from({ length: c.span }, (_, j) => j))
      .slice(0, m)
      .sort((x, y) => x - y);
    for (let t = 0; t < c.span; t++) ids[b * L + t] = BLANK;
    for (const pos of positions) {
      const s = rng.int(COPY_SYMBOLS.length);
      ids[b * L + pos] = s;
      labels.push(s);
    }
    for (let j = 0; j < m; j++) ids[b * L + c.span + j] = QUERY;
  }
  const tokens = ids.slice(0, L).map((k) => COPY_VOCAB[k]);
  return {
    inputs: { ids: new Tensor([B, L], ids) },
    labels,
    meta: {
      tokens,
      vocab: COPY_VOCAB,
      classNames: COPY_VOCAB,
      caption: `Copy ${labels
        .slice(0, m)
        .map((k) => COPY_SYMBOLS[k])
        .join(' ')} at the ${m} “?” positions.`,
    },
  };
}

export function copyDataset<C extends CopyConfig>(): Dataset<C> {
  return {
    id: 'ssm-copying',
    name: 'Selective copying',
    description:
      'A span of blanks “·” containing a few data tokens at random positions, followed by one query “?” per data token. The answer at the j-th query is the j-th data token. Because the positions vary, the model must decide from the content which inputs to keep.',
    source: 'synthetic',
    train: (rng, B, c) => copyBatch(c, rng, B),
    validation: (c) => copyBatch(c, new Rng(8001), 256),
    exampleCount: 8,
    example: (i, c) => copyBatch(c, new Rng(400 + i), 1),
    exampleLabel: (i, c) => {
      const b = copyBatch(c, new Rng(400 + i), 1);
      return b.meta!.tokens!.join('');
    },
  };
}
