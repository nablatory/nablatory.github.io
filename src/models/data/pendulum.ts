import { Rng, Tensor } from '../../engine';
import type { Batch, Config, Dataset } from '../types';

/**
 * Damped pendulum θ'' = −ω₀² sin θ − c θ', simulated deterministically with
 * classical RK4 (20 substeps per sample interval). Observations are the bob
 * position (sin θ, −cos θ) of a unit-length pendulum plus Gaussian noise.
 */
export interface PendulumSpec {
  /** Natural angular frequency ω₀ = √(g/L) (rad/s). */
  omega0: number;
  /** Damping coefficient c (1/s). */
  damping: number;
  /** Sample interval Δ (s). */
  dt: number;
  /** Observed samples (encoder input). */
  observed: number;
  /** Further samples used only to judge extrapolation. */
  future: number;
}

export const PENDULUM: PendulumSpec = {
  omega0: Math.PI,
  damping: 0.25,
  dt: 0.15,
  observed: 20,
  future: 40,
};

/** States (θ, θ') at times k·Δ, k = 0 … n − 1. */
export function simulatePendulum(
  theta0: number,
  omega0: number,
  n: number,
  spec: Pick<PendulumSpec, 'omega0' | 'damping' | 'dt'>,
): Float64Array {
  const out = new Float64Array(n * 2);
  const w2 = spec.omega0 * spec.omega0;
  const c = spec.damping;
  const sub = 20;
  const h = spec.dt / sub;
  const f = (th: number, om: number): [number, number] => [om, -w2 * Math.sin(th) - c * om];
  let th = theta0;
  let om = omega0;
  for (let k = 0; k < n; k++) {
    out[k * 2] = th;
    out[k * 2 + 1] = om;
    for (let s = 0; s < sub; s++) {
      const [a1, b1] = f(th, om);
      const [a2, b2] = f(th + (h / 2) * a1, om + (h / 2) * b1);
      const [a3, b3] = f(th + (h / 2) * a2, om + (h / 2) * b2);
      const [a4, b4] = f(th + h * a3, om + h * b3);
      th += (h / 6) * (a1 + 2 * a2 + 2 * a3 + a4);
      om += (h / 6) * (b1 + 2 * b2 + 2 * b3 + b4);
    }
  }
  return out;
}

export interface PendulumSet {
  /** Noisy observations [count, observed + future, 2]. */
  noisy: Float32Array;
  /** Noise-free bob positions [count, observed + future, 2]. */
  clean: Float32Array;
  count: number;
  length: number;
}

/** `count` sequences with θ(0) ~ U(−2, 2) rad and θ'(0) ~ U(−1, 1) rad/s. */
export function pendulumSet(
  count: number,
  seed: number,
  noise: number,
  spec: PendulumSpec = PENDULUM,
): PendulumSet {
  const rng = new Rng(seed);
  const L = spec.observed + spec.future;
  const noisy = new Float32Array(count * L * 2);
  const clean = new Float32Array(count * L * 2);
  for (let i = 0; i < count; i++) {
    const s = simulatePendulum(rng.range(-2, 2), rng.range(-1, 1), L, spec);
    for (let k = 0; k < L; k++) {
      const th = s[k * 2];
      const o = (i * L + k) * 2;
      clean[o] = Math.sin(th);
      clean[o + 1] = -Math.cos(th);
      noisy[o] = clean[o] + noise * rng.normal();
      noisy[o + 1] = clean[o + 1] + noise * rng.normal();
    }
  }
  return { noisy, clean, count, length: L };
}

function slab(src: Float32Array, idx: number[], L: number, from: number, to: number): Tensor {
  const w = to - from;
  const out = new Float32Array(idx.length * w * 2);
  idx.forEach((s, i) => out.set(src.subarray((s * L + from) * 2, (s * L + to) * 2), i * w * 2));
  return new Tensor([idx.length, w, 2], out);
}

/**
 * Batches carry `y` (the observed window, encoder input and reconstruction
 * target), `future` (later noisy samples) and `clean` (noise-free positions
 * over the whole horizon), each [N, time, 2].
 */
export function pendulumDataset<C extends Config>(
  noiseOf: (c: C) => number,
  spec: PendulumSpec = PENDULUM,
): Dataset<C> {
  const TRAIN = 512;
  const VAL = 64;
  const cache = new Map<string, PendulumSet>();
  const get = (c: C, split: 'train' | 'val') => {
    const noise = noiseOf(c);
    const key = `${split}:${noise}`;
    let d = cache.get(key);
    if (!d) {
      d = pendulumSet(split === 'train' ? TRAIN : VAL, split === 'train' ? 31 : 32, noise, spec);
      cache.set(key, d);
    }
    return d;
  };
  const batch = (d: PendulumSet, idx: number[], caption: string): Batch => ({
    inputs: {
      y: slab(d.noisy, idx, d.length, 0, spec.observed),
      future: slab(d.noisy, idx, d.length, spec.observed, d.length),
      clean: slab(d.clean, idx, d.length, 0, d.length),
    },
    meta: { caption },
  });
  return {
    id: 'pendulum',
    name: 'Damped pendulum',
    description: `${TRAIN} training and ${VAL} validation trajectories of a damped pendulum (ω₀ = π rad/s, c = ${spec.damping}/s), sampled every ${spec.dt} s, observed as the noisy bob position (sin θ, −cos θ). Simulated with RK4 in the browser.`,
    source: 'synthetic',
    train(rng, batchSize, c) {
      const d = get(c, 'train');
      return batch(
        d,
        Array.from({ length: batchSize }, () => rng.int(TRAIN)),
        'training batch',
      );
    },
    validation(c) {
      const d = get(c, 'val');
      return batch(
        d,
        Array.from({ length: VAL }, (_, i) => i),
        'validation set',
      );
    },
    exampleCount: 8,
    example(index, c) {
      return batch(get(c, 'val'), [index], `validation trajectory ${index}`);
    },
    exampleLabel(index) {
      return `Validation trajectory ${index}`;
    },
  };
}
