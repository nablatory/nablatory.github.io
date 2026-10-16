import { Rng, Tensor } from '../../engine';
import type { Batch, Config, Dataset } from '../types';

export type PlanarKind = 'spirals' | 'moons' | 'circles' | 'xor' | 'blobs';

export const PLANAR_OPTIONS: { value: PlanarKind; label: string }[] = [
  { value: 'spirals', label: 'Two spirals' },
  { value: 'moons', label: 'Two moons' },
  { value: 'circles', label: 'Concentric circles' },
  { value: 'xor', label: 'XOR blobs' },
  { value: 'blobs', label: 'Separable blobs' },
];

/** Deterministic two-class point sets in [−1.5, 1.5]². */
export function planarPoints(
  kind: PlanarKind,
  n: number,
  seed: number,
  noise = 0.08,
): { x: Float32Array; y: number[] } {
  const rng = new Rng(seed);
  const x = new Float32Array(n * 2);
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2;
    let px = 0;
    let py = 0;
    switch (kind) {
      case 'spirals': {
        const t = 0.25 + 3 * Math.sqrt(rng.uniform());
        const r = t / 3.25;
        const a = t * 2.1 + c * Math.PI;
        px = r * Math.cos(a) * 1.3;
        py = r * Math.sin(a) * 1.3;
        break;
      }
      case 'moons': {
        const t = rng.uniform() * Math.PI;
        px = c === 0 ? Math.cos(t) - 0.5 : 1 - Math.cos(t) - 0.5;
        py = c === 0 ? Math.sin(t) - 0.25 : -Math.sin(t) + 0.25;
        px *= 0.9;
        py *= 0.9;
        break;
      }
      case 'circles': {
        const t = rng.uniform() * 2 * Math.PI;
        const r = c === 0 ? 0.45 : 1.1;
        px = r * Math.cos(t);
        py = r * Math.sin(t);
        break;
      }
      case 'xor': {
        const q = rng.int(2);
        const sx = q === 0 ? 1 : -1;
        const sy = c === 0 ? sx : -sx;
        px = sx * 0.7 + 0.25 * rng.normal();
        py = sy * 0.7 + 0.25 * rng.normal();
        break;
      }
      case 'blobs': {
        px = (c === 0 ? -0.6 : 0.6) + 0.3 * rng.normal();
        py = (c === 0 ? -0.4 : 0.5) + 0.3 * rng.normal();
        break;
      }
    }
    x[i * 2] = px + noise * rng.normal();
    x[i * 2 + 1] = py + noise * rng.normal();
    y.push(c);
  }
  return { x, y };
}

export const PLANAR_EXTENT: [number, number, number, number] = [-1.6, 1.6, -1.6, 1.6];

/** Grid of points covering PLANAR_EXTENT, row 0 at the top (y = max). */
export function planarGrid(res: number): Tensor {
  const [x0, x1, y0, y1] = PLANAR_EXTENT;
  const g = new Float32Array(res * res * 2);
  for (let i = 0; i < res; i++)
    for (let j = 0; j < res; j++) {
      g[(i * res + j) * 2] = x0 + ((j + 0.5) / res) * (x1 - x0);
      g[(i * res + j) * 2 + 1] = y1 - ((i + 0.5) / res) * (y1 - y0);
    }
  return new Tensor([res * res, 2], g);
}

export function planarDataset<C extends Config>(
  kindOf: (c: C) => PlanarKind,
  opts: { train?: number; noise?: (c: C) => number } = {},
): Dataset<C> {
  const N = opts.train ?? 400;
  const cache = new Map<string, { x: Float32Array; y: number[] }>();
  const data = (c: C, split: 'train' | 'val') => {
    const kind = kindOf(c);
    const noise = opts.noise?.(c) ?? 0.08;
    const key = `${kind}:${noise}:${split}`;
    if (!cache.has(key))
      cache.set(
        key,
        planarPoints(kind, split === 'train' ? N : 200, split === 'train' ? 11 : 12, noise),
      );
    return cache.get(key)!;
  };
  const batch = (d: { x: Float32Array; y: number[] }, idx: number[], caption: string): Batch => {
    const x = new Float32Array(idx.length * 2);
    idx.forEach((k, i) => {
      x[i * 2] = d.x[k * 2];
      x[i * 2 + 1] = d.x[k * 2 + 1];
    });
    return {
      inputs: { x: new Tensor([idx.length, 2], x) },
      labels: idx.map((k) => d.y[k]),
      meta: { classNames: ['class 0', 'class 1'], caption },
    };
  };
  return {
    id: 'planar',
    name: 'Planar point sets',
    description: `${N} training and 200 validation points in two classes, generated deterministically in the browser.`,
    source: 'synthetic',
    train(rng, batchSize, c) {
      const d = data(c, 'train');
      return batch(
        d,
        Array.from({ length: batchSize }, () => rng.int(N)),
        'training batch',
      );
    },
    validation(c) {
      const d = data(c, 'val');
      return batch(
        d,
        d.y.map((_, i) => i),
        'validation set',
      );
    },
    exampleCount: 9,
    example(index, c) {
      const d = data(c, 'train');
      if (index === 0)
        return batch(
          d,
          d.y.map((_, i) => i),
          `all ${N} training points`,
        );
      return batch(d, [index - 1], `training point ${index - 1}`);
    },
    exampleLabel(index) {
      return index === 0 ? `All ${N} points` : `Point ${index - 1}`;
    },
  };
}
