/**
 * Noise schedules and the closed-form quantities of the DDPM forward process
 * (Ho et al., 2020). Timesteps are 1-based: arrays are indexed by t − 1.
 *
 *   q(x_t | x_{t−1}) = N(√α_t x_{t−1}, β_t I),   α_t = 1 − β_t
 *   q(x_t | x_0)     = N(√ᾱ_t x_0, (1 − ᾱ_t) I),  ᾱ_t = Π_{s≤t} α_s
 */
import type { Rng } from '../../engine';

export type ScheduleKind = 'linear' | 'cosine';

export const SCHEDULE_OPTIONS: { value: ScheduleKind; label: string }[] = [
  { value: 'linear', label: 'Linear β (Ho et al.)' },
  { value: 'cosine', label: 'Cosine ᾱ (Nichol & Dhariwal)' },
];

export interface Schedule {
  T: number;
  kind: ScheduleKind;
  beta: Float64Array;
  alpha: Float64Array;
  alphaBar: Float64Array;
  /** ᾱ_{t−1}, with ᾱ_0 = 1. */
  alphaBarPrev: Float64Array;
  /** Posterior variance β̃_t = (1 − ᾱ_{t−1}) / (1 − ᾱ_t) · β_t. */
  betaTilde: Float64Array;
  /** Signal-to-noise ratio ᾱ_t / (1 − ᾱ_t). */
  snr: Float64Array;
}

/**
 * Linear: β_t evenly spaced from 10⁻⁴·(1000/T) to 0.02·(1000/T), the T = 1000
 * values of Ho et al. rescaled as in Nichol & Dhariwal's code so that ᾱ_T stays
 * near zero for smaller T (clipped at 0.999, which only matters for T < 21).
 * Cosine: ᾱ_t = f(t)/f(0), f(t) = cos²(((t/T + s)/(1 + s))·π/2), s = 0.008,
 * β_t = min(1 − ᾱ_t/ᾱ_{t−1}, 0.999).
 */
export function makeSchedule(kind: ScheduleKind, T: number): Schedule {
  const beta = new Float64Array(T);
  if (kind === 'linear') {
    const scale = 1000 / T;
    const b0 = 1e-4 * scale;
    const b1 = 0.02 * scale;
    for (let i = 0; i < T; i++)
      beta[i] = Math.min(T === 1 ? b0 : b0 + ((b1 - b0) * i) / (T - 1), 0.999);
  } else {
    const s = 0.008;
    const f = (t: number) => Math.cos(((t / T + s) / (1 + s)) * (Math.PI / 2)) ** 2;
    for (let i = 0; i < T; i++) beta[i] = Math.min(1 - f(i + 1) / f(i), 0.999);
  }
  const alpha = new Float64Array(T);
  const alphaBar = new Float64Array(T);
  const alphaBarPrev = new Float64Array(T);
  const betaTilde = new Float64Array(T);
  const snr = new Float64Array(T);
  let acc = 1;
  for (let i = 0; i < T; i++) {
    alpha[i] = 1 - beta[i];
    alphaBarPrev[i] = acc;
    acc *= alpha[i];
    alphaBar[i] = acc;
    betaTilde[i] = ((1 - alphaBarPrev[i]) / (1 - alphaBar[i])) * beta[i];
    snr[i] = alphaBar[i] / (1 - alphaBar[i]);
  }
  return { T, kind, beta, alpha, alphaBar, alphaBarPrev, betaTilde, snr };
}

const cache = new Map<string, Schedule>();
export function schedule(kind: ScheduleKind, T: number): Schedule {
  const k = `${kind}:${T}`;
  let s = cache.get(k);
  if (!s) {
    s = makeSchedule(kind, T);
    cache.set(k, s);
  }
  return s;
}

/** ᾱ at timestep t ∈ {0, …, T}; ᾱ_0 = 1. */
export function alphaBarAt(s: Schedule, t: number): number {
  return t === 0 ? 1 : s.alphaBar[t - 1];
}

/** x_t = √ᾱ_t x_0 + √(1 − ᾱ_t) ε, elementwise over a flat array. */
export function qSample(
  s: Schedule,
  x0: ArrayLike<number>,
  t: number,
  eps: ArrayLike<number>,
  out: Float32Array | Float64Array,
): void {
  const ab = alphaBarAt(s, t);
  const a = Math.sqrt(ab);
  const b = Math.sqrt(1 - ab);
  for (let i = 0; i < out.length; i++) out[i] = a * x0[i] + b * eps[i];
}

/** One forward step x_t = √α_t x_{t−1} + √β_t ε (in place). */
export function qStep(s: Schedule, x: Float32Array | Float64Array, t: number, rng: Rng): void {
  const a = Math.sqrt(s.alpha[t - 1]);
  const b = Math.sqrt(s.beta[t - 1]);
  for (let i = 0; i < x.length; i++) x[i] = a * x[i] + b * rng.normal();
}

/**
 * DDIM noise scale (Song et al., 2021, eq. 16):
 * σ = η · √((1 − ᾱ_prev)/(1 − ᾱ_t)) · √(1 − ᾱ_t/ᾱ_prev).
 * With η = 1 and consecutive timesteps this equals √β̃_t.
 */
export function ddimSigma(abT: number, abPrev: number, eta: number): number {
  const v = ((1 - abPrev) / (1 - abT)) * (1 - abT / abPrev);
  return eta * Math.sqrt(Math.max(v, 0));
}

/** τ_1 < … < τ_S ⊂ {1, …, T}, evenly spaced and ending at T. */
export function ddimTimesteps(T: number, S: number): number[] {
  const n = Math.max(1, Math.min(S, T));
  const out: number[] = [];
  for (let i = 1; i <= n; i++) {
    const t = Math.round((i * T) / n);
    if (!out.length || t > out[out.length - 1]) out.push(t);
  }
  return out;
}
