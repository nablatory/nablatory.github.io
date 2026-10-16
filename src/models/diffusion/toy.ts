/**
 * Shared pieces of the two-dimensional diffusion models (DDPM and DDIM):
 * the toy data, the sinusoidal time embedding, the MLP noise predictor
 * ε_θ(x_t, t), the training objective and the two samplers.
 */
import {
  ParamBuilder,
  Rng,
  Tensor,
  add,
  declareLinear,
  dense,
  graphRng,
  mse,
  mul,
  scope,
  silu,
  type Var,
} from '../../engine';
import type {
  Batch,
  Dataset,
  ForwardResult,
  Procedure,
  ProcedureContext,
  ViewSpec,
  ConfigField,
} from '../types';
import { PLANAR_OPTIONS, planarPoints, type PlanarKind } from '../data/planar';
import { constant, input } from '../util';
import {
  SCHEDULE_OPTIONS,
  alphaBarAt,
  ddimSigma,
  qSample,
  qStep,
  schedule,
  type Schedule,
  type ScheduleKind,
} from './schedule';

export type ToyConfig = {
  dataset: PlanarKind;
  noise: number;
  T: number;
  schedule: ScheduleKind;
  width: number;
  depth: number;
  tdim: number;
};

/** Planar point sets are multiplied by this factor so the data spread is comparable to N(0, I). */
export const DATA_SCALE = 1.5;
export const TOY_EXTENT: [number, number, number, number] = [-3, 3, -3, 3];
const TRAIN_N = 2000;
const VAL_N = 400;
const SHOW_N = 400;

// ---------------------------------------------------------------- data

function toyPoints(
  kind: PlanarKind,
  n: number,
  seed: number,
  noise: number,
): { x: Float32Array; y: number[] } {
  const p = planarPoints(kind, n, seed, noise);
  for (let i = 0; i < p.x.length; i++) p.x[i] *= DATA_SCALE;
  return p;
}

const pointCache = new Map<string, { x: Float32Array; y: number[] }>();
export function toyData(
  c: ToyConfig,
  split: 'train' | 'val' | 'show',
): { x: Float32Array; y: number[] } {
  const key = `${c.dataset}:${c.noise}:${split}`;
  let d = pointCache.get(key);
  if (!d) {
    const [n, seed] =
      split === 'train' ? [TRAIN_N, 21] : split === 'val' ? [VAL_N, 22] : [SHOW_N, 23];
    d = toyPoints(c.dataset, n, seed, c.noise);
    pointCache.set(key, d);
  }
  return d;
}

function pointBatch(d: { x: Float32Array; y: number[] }, idx: number[], caption: string): Batch {
  const x = new Float32Array(idx.length * 2);
  idx.forEach((k, i) => {
    x[i * 2] = d.x[k * 2];
    x[i * 2 + 1] = d.x[k * 2 + 1];
  });
  return {
    inputs: { x: new Tensor([idx.length, 2], x) },
    labels: idx.map((k) => d.y[k]),
    meta: { caption },
  };
}

export function toyDataset<C extends ToyConfig>(): Dataset<C> {
  return {
    id: 'toy2d',
    name: 'Planar point clouds',
    description: `${TRAIN_N} training and ${VAL_N} validation points from a two-dimensional distribution (planar point sets scaled by ${DATA_SCALE}), generated deterministically in the browser. Class labels are ignored by the model.`,
    source: 'synthetic',
    train(rng, batchSize, c) {
      const d = toyData(c, 'train');
      return pointBatch(
        d,
        Array.from({ length: batchSize }, () => rng.int(TRAIN_N)),
        'training batch',
      );
    },
    validation(c) {
      const d = toyData(c, 'val');
      return pointBatch(
        d,
        d.y.map((_, i) => i),
        'validation set',
      );
    },
    exampleCount: 2,
    example(index, c) {
      const d = toyData(c, 'train');
      const n = index === 0 ? 256 : 16;
      return pointBatch(
        d,
        Array.from({ length: n }, (_, i) => i),
        `first ${n} training points`,
      );
    },
    exampleLabel(index) {
      return index === 0 ? '256 training points' : '16 training points';
    },
  };
}

// ---------------------------------------------------------------- network

/**
 * Transformer-style sinusoidal embedding of the timestep (Ho et al., 2020,
 * appendix B), applied to the rescaled time s = 1000·t/T so that the
 * embedding of a given noise level does not depend on T:
 * e_k = sin(s·ω_k), e_{k+d/2} = cos(s·ω_k), ω_k = 10000^{−k/(d/2−1)}.
 */
export function timeEmbedding(ts: ArrayLike<number>, T: number, dim: number): Tensor {
  const half = dim / 2;
  const out = new Float32Array(ts.length * dim);
  const k0 = Math.log(10000) / Math.max(half - 1, 1);
  for (let n = 0; n < ts.length; n++) {
    const s = (1000 * ts[n]) / T;
    for (let k = 0; k < half; k++) {
      const a = s * Math.exp(-k0 * k);
      out[n * dim + k] = Math.sin(a);
      out[n * dim + half + k] = Math.cos(a);
    }
  }
  return new Tensor([ts.length, dim], out);
}

export function initToyNet(c: ToyConfig, rng: Rng): ParamBuilder {
  const b = new ParamBuilder(rng);
  declareLinear(b, 'time.fc1', c.tdim, c.width);
  declareLinear(b, 'time.fc2', c.width, c.width);
  declareLinear(b, 'input', 2, c.width);
  for (let l = 1; l <= c.depth; l++) {
    declareLinear(b, `block${l}.fc`, c.width, c.width);
    declareLinear(b, `block${l}.temb`, c.width, c.width);
  }
  declareLinear(b, 'output', c.width, 2);
  return b;
}

/**
 * ε_θ(x_t, t): e = W₂ SiLU(W₁ emb(t)),  h₀ = W_in x_t,
 * h_ℓ = h_{ℓ−1} + SiLU(W_ℓ h_{ℓ−1} + U_ℓ e),  ε̂ = W_out h_L.
 * `temb` is [N, d] or [1, d] (broadcast when every row shares t).
 */
export function epsNet(x: Var, temb: Var, c: ToyConfig): Var {
  const e = scope('time', () => dense(silu(dense(temb, 'time.fc1')), 'time.fc2'));
  let h = scope('input', () => dense(x, 'input'));
  for (let l = 1; l <= c.depth; l++) {
    const prev = h;
    h = scope(`block${l}`, () =>
      add(prev, silu(add(dense(prev, `block${l}.fc`), dense(e, `block${l}.temb`)))),
    );
  }
  return scope('output', () => dense(h, 'output'));
}

/** ε_θ(x, t) for a whole batch sharing one timestep (no gradient; inside a procedure graph). */
export function predictEps(x: Tensor, t: number, c: ToyConfig): Tensor {
  const temb = constant(timeEmbedding([t], c.T, c.tdim), 'sinusoid');
  return epsNet(constant(x, 'x'), temb, c).value;
}

function scheduleOutputs(s: Schedule): Record<string, Var> {
  return scope('schedule', () => ({
    beta: constant(new Tensor([s.T], Float32Array.from(s.beta)), 'β'),
    alphaBar: constant(new Tensor([s.T], Float32Array.from(s.alphaBar)), 'ᾱ'),
    logSnr: constant(
      new Tensor(
        [s.T],
        Float32Array.from(s.snr, (v) => Math.log(v)),
      ),
      'log SNR',
    ),
    timestep: constant(
      new Tensor(
        [s.T],
        Float32Array.from({ length: s.T }, (_, i) => i + 1),
      ),
      't',
    ),
  }));
}

/**
 * Training objective L_simple (Ho et al., 2020, eq. 14): t ~ U{1, …, T},
 * ε ~ N(0, I), x_t = √ᾱ_t x_0 + √(1 − ᾱ_t) ε, loss ‖ε − ε_θ(x_t, t)‖².
 * t and ε are drawn from the graph's seeded generator, so two evaluations
 * with the same seed see the same noise.
 */
export function toyForward(batch: Batch, c: ToyConfig): ForwardResult {
  const s = schedule(c.schedule, c.T);
  const x0v = batch.inputs.x;
  const N = x0v.shape[0];
  const rng = graphRng();
  const ts = new Float32Array(N);
  const a = new Float32Array(N);
  const b = new Float32Array(N);
  const epsData = new Float32Array(N * 2);
  for (let n = 0; n < N; n++) {
    ts[n] = 1 + rng.int(c.T);
    const ab = alphaBarAt(s, ts[n]);
    a[n] = Math.sqrt(ab);
    b[n] = Math.sqrt(1 - ab);
    epsData[n * 2] = rng.normal();
    epsData[n * 2 + 1] = rng.normal();
  }
  const x0 = input('x0', x0v, false);
  const { xt, eps } = scope('diffuse', () => {
    const eps = constant(new Tensor([N, 2], epsData), 'ε');
    const xt = add(
      mul(x0, constant(new Tensor([N, 1], a), '√ᾱ_t')),
      mul(eps, constant(new Tensor([N, 1], b), '√(1−ᾱ_t)')),
    );
    return { xt, eps };
  });
  const temb = scope('embed', () => constant(timeEmbedding(ts, c.T, c.tdim), 'sinusoid'));
  const epsHat = epsNet(xt, temb, c);
  const loss = scope('loss', () => mse(epsHat, eps));
  return {
    outputs: { x0, xt, eps, epsHat, t: constant(new Tensor([N], ts), 't'), ...scheduleOutputs(s) },
    loss,
    metrics: { meanT: ts.reduce((p, q) => p + q, 0) / N },
  };
}

// ---------------------------------------------------------------- samplers

export type EpsFn = (x: Tensor, t: number) => Tensor;
export interface Hooks {
  shouldStop(): boolean;
  progress(f: number): void;
}
const noHooks: Hooks = { shouldStop: () => false, progress: () => {} };

export interface ChainResult {
  x: Tensor;
  /** First `track` rows at every visited timestep: [track, steps + 1, D]. */
  trajectories: Tensor | null;
  /** Network evaluations performed. */
  nfe: number;
  /** Timestep the chain had reached when cancelled (null if it finished). */
  stoppedAt: number | null;
}

function recorder(xT: Tensor, track: number, steps: number) {
  const [N, D] = xT.shape;
  const K = Math.min(track, N);
  const buf = K > 0 ? new Float32Array(K * (steps + 1) * D) : null;
  let k = 0;
  return {
    push(x: Tensor) {
      if (!buf) return;
      for (let i = 0; i < K; i++)
        for (let d = 0; d < D; d++) buf[(i * (steps + 1) + k) * D + d] = x.data[i * D + d];
      k++;
    },
    result(): Tensor | null {
      if (!buf) return null;
      // Repeat the last state if the chain was cancelled early.
      for (let j = k; j <= steps; j++)
        for (let i = 0; i < K; i++)
          for (let d = 0; d < D; d++)
            buf[(i * (steps + 1) + j) * D + d] = buf[(i * (steps + 1) + k - 1) * D + d];
      return new Tensor([K, steps + 1, D], buf);
    },
  };
}

/**
 * Ancestral sampling (Ho et al., 2020, Algorithm 2): for t = T, …, 1
 * x_{t−1} = (x_t − β_t/√(1 − ᾱ_t) · ε_θ(x_t, t)) / √α_t + σ_t z,
 * σ_t² = β_t or β̃_t, and z = 0 at t = 1.
 */
export function ancestralSample(
  eps: EpsFn,
  xT: Tensor,
  s: Schedule,
  variance: 'beta' | 'betaTilde',
  rng: Rng,
  opts: { track?: number; hooks?: Hooks } = {},
): ChainResult {
  const hooks = opts.hooks ?? noHooks;
  const rec = recorder(xT, opts.track ?? 0, s.T);
  let x = new Tensor(xT.shape, Float32Array.from(xT.data));
  rec.push(x);
  let nfe = 0;
  for (let t = s.T; t >= 1; t--) {
    if (hooks.shouldStop()) return { x, trajectories: rec.result(), nfe, stoppedAt: t };
    const e = eps(x, t);
    nfe++;
    const i = t - 1;
    const c1 = 1 / Math.sqrt(s.alpha[i]);
    const c2 = s.beta[i] / Math.sqrt(1 - s.alphaBar[i]);
    const sigma = t > 1 ? Math.sqrt(variance === 'beta' ? s.beta[i] : s.betaTilde[i]) : 0;
    const out = new Float32Array(x.size);
    for (let j = 0; j < out.length; j++)
      out[j] = c1 * (x.data[j] - c2 * e.data[j]) + (sigma > 0 ? sigma * rng.normal() : 0);
    x = new Tensor(x.shape, out);
    rec.push(x);
    hooks.progress((s.T - t + 1) / s.T);
  }
  return { x, trajectories: rec.result(), nfe, stoppedAt: null };
}

/**
 * One DDIM update (Song et al., 2021, eq. 12) from timestep t to prev < t:
 * x̂₀ = (x_t − √(1 − ᾱ_t) ε̂)/√ᾱ_t,
 * x_prev = √ᾱ_prev x̂₀ + √(1 − ᾱ_prev − σ²) ε̂ + σ z.
 * Returns the mean (without σ z) and σ.
 */
export function ddimMean(
  xt: number,
  e: number,
  abT: number,
  abPrev: number,
  eta: number,
): { mean: number; sigma: number } {
  const sigma = ddimSigma(abT, abPrev, eta);
  const x0 = (xt - Math.sqrt(1 - abT) * e) / Math.sqrt(abT);
  const mean = Math.sqrt(abPrev) * x0 + Math.sqrt(Math.max(1 - abPrev - sigma * sigma, 0)) * e;
  return { mean, sigma };
}

/** DDIM sampling over the sub-sequence `taus` (ascending, ending at T). η = 0 draws no noise. */
export function ddimSample(
  eps: EpsFn,
  xT: Tensor,
  s: Schedule,
  taus: number[],
  eta: number,
  rng: Rng,
  opts: { track?: number; hooks?: Hooks } = {},
): ChainResult {
  const hooks = opts.hooks ?? noHooks;
  const S = taus.length;
  const rec = recorder(xT, opts.track ?? 0, S);
  let x = new Tensor(xT.shape, Float32Array.from(xT.data));
  rec.push(x);
  let nfe = 0;
  for (let i = S - 1; i >= 0; i--) {
    const t = taus[i];
    if (hooks.shouldStop()) return { x, trajectories: rec.result(), nfe, stoppedAt: t };
    const prev = i > 0 ? taus[i - 1] : 0;
    const abT = alphaBarAt(s, t);
    const abPrev = alphaBarAt(s, prev);
    const e = eps(x, t);
    nfe++;
    const out = new Float32Array(x.size);
    for (let j = 0; j < out.length; j++) {
      const { mean, sigma } = ddimMean(x.data[j], e.data[j], abT, abPrev, eta);
      out[j] = sigma > 0 ? mean + sigma * rng.normal() : mean;
    }
    x = new Tensor(x.shape, out);
    rec.push(x);
    hooks.progress((S - i) / S);
  }
  return { x, trajectories: rec.result(), nfe, stoppedAt: null };
}

export function gaussian(shape: number[], rng: Rng): Tensor {
  const t = new Tensor(shape, new Float32Array(shape.reduce((a, b) => a * b, 1)));
  for (let i = 0; i < t.size; i++) t.data[i] = rng.normal();
  return t;
}

// ---------------------------------------------------------------- diagnostics

/**
 * Squared maximum mean discrepancy with a Gaussian kernel of bandwidth h
 * (biased V-statistic). Zero when the two point sets coincide in distribution.
 */
export function mmd2(a: ArrayLike<number>, b: ArrayLike<number>, D: number, h = 0.3): number {
  const na = a.length / D;
  const nb = b.length / D;
  const k = (p: ArrayLike<number>, i: number, q: ArrayLike<number>, j: number) => {
    let s = 0;
    for (let d = 0; d < D; d++) {
      const v = p[i * D + d] - q[j * D + d];
      s += v * v;
    }
    return Math.exp(-s / (2 * h * h));
  };
  let aa = 0;
  let bb = 0;
  let ab = 0;
  for (let i = 0; i < na; i++) for (let j = 0; j < na; j++) aa += k(a, i, a, j);
  for (let i = 0; i < nb; i++) for (let j = 0; j < nb; j++) bb += k(b, i, b, j);
  for (let i = 0; i < na; i++) for (let j = 0; j < nb; j++) ab += k(a, i, b, j);
  return aa / (na * na) + bb / (nb * nb) - (2 * ab) / (na * nb);
}

/**
 * Learned score s_θ(x, t) = −ε_θ(x, t)/√(1 − ᾱ_t) on a G × G grid over
 * TOY_EXTENT, laid out for the phase view (row 0 at the bottom).
 */
export function scoreField(c: ToyConfig, t: number, G = 21): Tensor {
  const [x0, x1, y0, y1] = TOY_EXTENT;
  const g = new Float32Array(G * G * 2);
  for (let i = 0; i < G; i++)
    for (let j = 0; j < G; j++) {
      g[(i * G + j) * 2] = x0 + ((j + 0.5) / G) * (x1 - x0);
      g[(i * G + j) * 2 + 1] = y0 + ((i + 0.5) / G) * (y1 - y0);
    }
  const e = predictEps(new Tensor([G * G, 2], g), t, c);
  const k = -1 / Math.sqrt(1 - alphaBarAt(schedule(c.schedule, c.T), t));
  const out = new Float32Array(G * G * 2);
  for (let i = 0; i < out.length; i++) out[i] = k * e.data[i];
  return new Tensor([G, G, 2], out);
}

/** Data points (label 0) followed by samples (label 1), for one scatter plot. */
export function overlay(c: ToyConfig, samples: Tensor): { points: Tensor; labels: Tensor } {
  const d = toyData(c, 'show');
  const n = d.y.length;
  const m = samples.shape[0];
  const pts = new Float32Array((n + m) * 2);
  pts.set(d.x, 0);
  pts.set(samples.data, n * 2);
  const lab = new Float32Array(n + m);
  lab.fill(1, n);
  return { points: new Tensor([n + m, 2], pts), labels: new Tensor([n + m], lab) };
}

export function clampT(t: number, T: number): number {
  return Math.max(1, Math.min(T, Math.round(t)));
}

// ---------------------------------------------------------------- shared definition parts

export const toyConfigSchema: ConfigField[] = [
  {
    key: 'dataset',
    label: 'Data distribution',
    kind: 'select',
    options: PLANAR_OPTIONS,
    affects: 'data',
  },
  {
    key: 'noise',
    label: 'Point noise σ',
    kind: 'float',
    min: 0,
    max: 0.3,
    step: 0.01,
    affects: 'data',
  },
  {
    key: 'T',
    label: 'Diffusion steps T',
    kind: 'int',
    min: 20,
    max: 1000,
    step: 10,
    affects: 'forward',
    help: 'The network sees the rescaled time 1000·t/T, so T can be changed without changing parameter shapes.',
  },
  {
    key: 'schedule',
    label: 'Noise schedule',
    kind: 'select',
    options: SCHEDULE_OPTIONS,
    affects: 'forward',
  },
  {
    key: 'width',
    label: 'Hidden width',
    kind: 'int',
    min: 16,
    max: 128,
    step: 16,
    affects: 'architecture',
  },
  { key: 'depth', label: 'Residual blocks', kind: 'int', min: 1, max: 4, affects: 'architecture' },
  {
    key: 'tdim',
    label: 'Time embedding size',
    kind: 'int',
    min: 4,
    max: 64,
    step: 2,
    affects: 'architecture',
  },
];

export const toyDefaults: ToyConfig = {
  dataset: 'moons',
  noise: 0.05,
  T: 100,
  schedule: 'linear',
  width: 64,
  depth: 3,
  tdim: 16,
};

export function validateToy(c: ToyConfig): string | null {
  if (c.tdim % 2 !== 0)
    return `The sinusoidal embedding pairs a sine and a cosine per frequency, so its size must be even (got ${c.tdim}).`;
  return null;
}

/** Forward-process procedure: q(x_t | x_0) in closed form beside t iterated steps of q(x_s | x_{s−1}). */
export function diffuseProcedure<C extends ToyConfig>(): Procedure<C> {
  return {
    id: 'diffuse',
    label: 'Forward process at step t',
    description:
      'Noises the 400 display points to timestep t twice: in one jump with the closed form q(x_t | x₀), and by iterating q(x_s | x_{s−1}) t times with fresh noise. Both should have mean √ᾱ_t x₀ and variance 1 − ᾱ_t.',
    args: [{ key: 't', label: 'Timestep t', kind: 'int', min: 1, max: 1000, affects: 'forward' }],
    defaults: { t: 20 },
    run(ctx: ProcedureContext<C>) {
      const c = ctx.config;
      const s = schedule(c.schedule, c.T);
      const t = clampT(Number(ctx.args.t), c.T);
      const d = toyData(c, 'show');
      const n = d.y.length;
      const eps = new Float32Array(n * 2);
      for (let i = 0; i < eps.length; i++) eps[i] = ctx.rng.normal();
      const closed = new Float32Array(n * 2);
      qSample(s, d.x, t, eps, closed);
      const iter = Float64Array.from(d.x);
      for (let k = 1; k <= t; k++) {
        if (ctx.shouldStop()) break;
        qStep(s, iter, k, ctx.rng);
        ctx.progress(k / t);
      }
      const ab = alphaBarAt(s, t);
      const resid = (x: ArrayLike<number>) => {
        let m = 0;
        let v = 0;
        for (let i = 0; i < x.length; i++) m += x[i] - Math.sqrt(ab) * d.x[i];
        m /= x.length;
        for (let i = 0; i < x.length; i++) v += (x[i] - Math.sqrt(ab) * d.x[i] - m) ** 2;
        return v / (x.length - 1);
      };
      return {
        tensors: {
          closed: new Tensor([n, 2], closed),
          iterated: new Tensor([n, 2], Float32Array.from(iter)),
          labels: new Tensor([n], Float32Array.from(d.y)),
        },
        scalars: {
          t,
          ᾱ_t: ab,
          '1 − ᾱ_t': 1 - ab,
          'var (closed)': resid(closed),
          'var (iterated)': resid(iter),
        },
      };
    },
  };
}

/** Note for the phase views of the diffusion pages (score field plus sampling chains). */
export const SCORE_NOTE =
  'Arrows: learned score s_θ(x, t) = −ε_θ(x, t)/√(1 − ᾱ_t) at the chosen t on a 21 × 21 grid, length ∝ ‖s_θ‖ (normalised to the largest). Curves: the first 24 sampling chains, every visited state from x_T to x₀.';

export const toyViews: ViewSpec[] = [
  {
    kind: 'series',
    title: 'Noise schedule: β_t and ᾱ_t',
    sources: [
      { ref: 'out:beta', label: 'β_t' },
      { ref: 'out:alphaBar', label: 'ᾱ_t' },
    ],
    x: 'out:timestep',
    xLabel: 't',
  },
  {
    kind: 'series',
    title: 'Log signal-to-noise ratio, log(ᾱ_t / (1 − ᾱ_t))',
    sources: [{ ref: 'out:logSnr', label: 'log SNR' }],
    x: 'out:timestep',
    xLabel: 't',
  },
  {
    kind: 'scatter',
    title: 'q(x_t | x₀) in closed form (colour: data class)',
    source: 'proc:diffuse:closed',
    colorBy: 'proc:diffuse:labels',
    extent: TOY_EXTENT,
  },
  {
    kind: 'scatter',
    title: 'Iterated forward steps q(x_s | x_{s−1}), s = 1…t',
    source: 'proc:diffuse:iterated',
    colorBy: 'proc:diffuse:labels',
    extent: TOY_EXTENT,
  },
];

export const toyMath = {
  diffuse: {
    tex: 'x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon,\\quad \\varepsilon\\sim\\mathcal N(0,I),\\; t\\sim\\mathcal U\\{1,\\dots,T\\}',
    text: 'Each training point gets its own random timestep and noise, drawn from the seeded generator of the step.',
  },
  schedule: {
    tex: '\\alpha_t = 1-\\beta_t,\\quad \\bar\\alpha_t = \\prod_{s=1}^{t}\\alpha_s,\\quad \\mathrm{SNR}(t) = \\frac{\\bar\\alpha_t}{1-\\bar\\alpha_t}',
    text: 'Linear: β_t from 10⁻⁴·1000/T to 0.02·1000/T, the T = 1000 values of Ho et al. rescaled as in Nichol & Dhariwal’s code (clipped at 0.999, which only matters at T = 20). Cosine: ᾱ_t = f(t)/f(0), f(t) = cos²(((t/T + 0.008)/1.008)·π/2), β_t clipped at 0.999.',
  },
  embed: {
    tex: 'e_k(t) = \\sin(s\\,\\omega_k),\\; e_{k+d/2}(t) = \\cos(s\\,\\omega_k),\\quad s = 1000\\,t/T,\\; \\omega_k = 10000^{-k/(d/2-1)}',
  },
  time: { tex: 'e = W_2\\,\\mathrm{SiLU}(W_1\\,\\mathrm{emb}(t) + b_1) + b_2' },
  input: { tex: 'h_0 = W_{\\text{in}} x_t + b_{\\text{in}}' },
  block: {
    tex: 'h_\\ell = h_{\\ell-1} + \\mathrm{SiLU}(W_\\ell h_{\\ell-1} + U_\\ell\\, e + b_\\ell)',
    text: 'Residual MLP block; the time embedding is added inside every block.',
  },
  output: {
    tex: '\\hat\\varepsilon = \\varepsilon_\\theta(x_t, t) = W_{\\text{out}} h_L + b_{\\text{out}}',
  },
  loss: {
    tex: 'L_{\\text{simple}} = \\mathbb E_{t,x_0,\\varepsilon}\\big\\|\\varepsilon - \\varepsilon_\\theta(x_t, t)\\big\\|^2',
    text: 'Averaged over the batch and the two coordinates (a constant factor ½ relative to the summed squared norm).',
  },
};
