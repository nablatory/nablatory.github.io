import { Rng, Tensor } from '../../engine';
import type { Batch, Config, Dataset } from '../types';

/**
 * Two-dimensional target distributions for generative models. Every sample is
 * drawn from an explicit Rng, so a given seed always yields the same points.
 *
 * ring    – mixture of 8 Gaussians, σ = 0.05, centres on a circle of radius 1
 * grid    – mixture of 25 Gaussians, σ = 0.04, centres on a 5 × 5 grid in [−1, 1]²
 * spiral  – a one-dimensional spiral curve with Gaussian jitter σ = 0.03
 */
export type Toy2dKind = 'ring' | 'grid' | 'spiral';

export const TOY2D_OPTIONS: { value: Toy2dKind; label: string }[] = [
  { value: 'ring', label: '8 Gaussians on a ring' },
  { value: 'grid', label: '25 Gaussians on a grid' },
  { value: 'spiral', label: 'Spiral' },
];

export const TOY2D_EXTENT: [number, number, number, number] = [-1.6, 1.6, -1.6, 1.6];

/** Mixture centres and component standard deviation (null for the spiral). */
export function toy2dModes(kind: Toy2dKind): { centres: [number, number][]; sigma: number } | null {
  if (kind === 'ring') {
    const centres: [number, number][] = [];
    for (let k = 0; k < 8; k++)
      centres.push([Math.cos((2 * Math.PI * k) / 8), Math.sin((2 * Math.PI * k) / 8)]);
    return { centres, sigma: 0.05 };
  }
  if (kind === 'grid') {
    const centres: [number, number][] = [];
    for (let i = 0; i < 5; i++)
      for (let j = 0; j < 5; j++) centres.push([-1 + 0.5 * i, -1 + 0.5 * j]);
    return { centres, sigma: 0.04 };
  }
  return null;
}

/** n points from the target distribution, as an [n, 2] tensor. */
export function toy2dSample(kind: Toy2dKind, n: number, rng: Rng): Tensor {
  const x = new Float32Array(n * 2);
  const modes = toy2dModes(kind);
  for (let i = 0; i < n; i++) {
    if (modes) {
      const [cx, cy] = modes.centres[rng.int(modes.centres.length)];
      x[i * 2] = cx + modes.sigma * rng.normal();
      x[i * 2 + 1] = cy + modes.sigma * rng.normal();
    } else {
      const t = Math.sqrt(0.05 + 0.95 * rng.uniform()) * 3 * Math.PI;
      const r = (1.35 * t) / (3 * Math.PI);
      x[i * 2] = r * Math.cos(t) + 0.03 * rng.normal();
      x[i * 2 + 1] = r * Math.sin(t) + 0.03 * rng.normal();
    }
  }
  return new Tensor([n, 2], x);
}

/** Grid of points covering TOY2D_EXTENT, row 0 at the top (y = max). */
export function toy2dGrid(res: number): Tensor {
  const [x0, x1, y0, y1] = TOY2D_EXTENT;
  const g = new Float32Array(res * res * 2);
  for (let i = 0; i < res; i++)
    for (let j = 0; j < res; j++) {
      g[(i * res + j) * 2] = x0 + ((j + 0.5) / res) * (x1 - x0);
      g[(i * res + j) * 2 + 1] = y1 - ((i + 0.5) / res) * (y1 - y0);
    }
  return new Tensor([res * res, 2], g);
}

/**
 * Mode coverage of a set of samples: a mode counts as covered when at least
 * `minCount` samples lie within 3σ of its centre; "quality" is the fraction of
 * samples within 3σ of any centre.
 */
export function modeCoverage(
  kind: Toy2dKind,
  pts: Tensor,
  minCount = 5,
): { covered: number; total: number; quality: number } | null {
  const modes = toy2dModes(kind);
  if (!modes) return null;
  const n = pts.shape[0];
  const counts = new Array(modes.centres.length).fill(0);
  const r2 = (3 * modes.sigma) ** 2;
  let near = 0;
  for (let i = 0; i < n; i++) {
    const x = pts.data[i * 2];
    const y = pts.data[i * 2 + 1];
    let best = -1;
    let bd = Infinity;
    modes.centres.forEach(([cx, cy], k) => {
      const d = (x - cx) ** 2 + (y - cy) ** 2;
      if (d < bd) {
        bd = d;
        best = k;
      }
    });
    if (bd <= r2) {
      counts[best]++;
      near++;
    }
  }
  return {
    covered: counts.filter((c) => c >= minCount).length,
    total: modes.centres.length,
    quality: near / Math.max(1, n),
  };
}

/**
 * Dataset of unlabelled points; batches carry `inputs.x` of shape [N, 2].
 * Inspection examples carry `meta.field = true` so that models can attach a
 * field over the plane to them (not to training batches).
 */
export function toy2dDataset<C extends Config>(
  kindOf: (c: C) => Toy2dKind,
  exampleSize = 256,
): Dataset<C> {
  return {
    id: 'toy2d',
    name: 'Two-dimensional target',
    description:
      'Deterministic synthetic point clouds in the plane: a ring or grid of Gaussians, or a spiral.',
    source: 'synthetic',
    train(rng, batchSize, config) {
      return { inputs: { x: toy2dSample(kindOf(config), batchSize, rng) } };
    },
    validation(config) {
      return { inputs: { x: toy2dSample(kindOf(config), 512, new Rng(9001)) } };
    },
    exampleCount: 4,
    example(index, config): Batch {
      return {
        inputs: { x: toy2dSample(kindOf(config), exampleSize, new Rng(1000 + index)) },
        meta: { field: true, caption: `${exampleSize} target samples (draw ${index + 1})` },
      };
    },
    exampleLabel(index) {
      return `Target draw ${index + 1}`;
    },
  };
}
