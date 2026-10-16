import type { Rng } from '../../engine';

export interface MppiOptions {
  horizon: number;
  actionDim: number;
  /** Gaussian samples per iteration. */
  samples: number;
  elites: number;
  iterations: number;
  /** τ in the elite weights exp(τ (G − max G)). */
  temperature: number;
  minStd: number;
  maxStd: number;
  /** Warm-start mean [H · A] (e.g. the previous plan shifted by one step); zeros otherwise. */
  initMean?: Float64Array;
}

export interface MppiIteration {
  /** Return of every candidate evaluated in this iteration (Gaussian samples, then extra candidates). */
  returns: Float64Array;
  /** Mean return of the elites. */
  eliteMean: number;
  /** Mean action sequence after the refit [H · A]. */
  mean: Float64Array;
  std: Float64Array;
}

export interface MppiResult {
  mean: Float64Array;
  std: Float64Array;
  iterations: MppiIteration[];
  /** Return of the mean sequence before the first iteration and after each one (length J + 1). */
  meanReturn: number[];
}

/**
 * Model predictive path integral planning as used in TD-MPC (Hansen et al.
 * 2022, Algorithm 2): sample action sequences from N(μ, σ²), clip to [−1, 1],
 * score them with `evaluate`, keep the top-k elites and refit
 *   μ = Σ_i w_i A_i / Σ_i w_i,  σ = sqrt(Σ_i w_i (A_i − μ)² / Σ_i w_i),
 *   w_i = exp(τ (G_i − max_j G_j)),
 * with σ clipped to [minStd, maxStd]. `extra` may add candidates (e.g. policy
 * rollouts) to each iteration.
 *
 * `evaluate(actions, n)` receives n sequences [n · H · A] and returns n returns.
 */
export function mppi(
  evaluate: (actions: Float64Array, n: number) => Float64Array,
  opts: MppiOptions,
  rng: Rng,
  extra?: (iteration: number) => Float64Array[],
): MppiResult {
  const { horizon: H, actionDim: A, samples: N, iterations: J, temperature } = opts;
  const D = H * A;
  const mean = opts.initMean ? Float64Array.from(opts.initMean) : new Float64Array(D);
  const std = new Float64Array(D).fill(opts.maxStd);
  const iterations: MppiIteration[] = [];
  const meanReturn: number[] = [];
  for (let j = 0; j < J; j++) {
    const ext = extra?.(j) ?? [];
    const n = N + ext.length + 1;
    const cand = new Float64Array(n * D);
    for (let i = 0; i < N; i++)
      for (let k = 0; k < D; k++)
        cand[i * D + k] = Math.max(-1, Math.min(1, mean[k] + std[k] * rng.normal()));
    ext.forEach((seq, e) => cand.set(seq, (N + e) * D));
    cand.set(mean, (n - 1) * D);
    const all = evaluate(cand, n);
    meanReturn.push(all[n - 1]);
    const returns = all.slice(0, n - 1);
    const order = Array.from(returns.keys()).sort((a, b) => returns[b] - returns[a]);
    const elite = order.slice(0, Math.min(opts.elites, order.length));
    const maxG = returns[elite[0]];
    const w = elite.map((i) => Math.exp(temperature * (returns[i] - maxG)));
    const wSum = w.reduce((a, b) => a + b, 0);
    for (let k = 0; k < D; k++) {
      let m = 0;
      elite.forEach((i, e) => (m += w[e] * cand[i * D + k]));
      m /= wSum;
      let v = 0;
      elite.forEach((i, e) => (v += w[e] * (cand[i * D + k] - m) ** 2));
      mean[k] = m;
      std[k] = Math.max(opts.minStd, Math.min(opts.maxStd, Math.sqrt(v / wSum)));
    }
    const eliteMean = elite.reduce((s, i) => s + returns[i], 0) / elite.length;
    iterations.push({
      returns,
      eliteMean,
      mean: Float64Array.from(mean),
      std: Float64Array.from(std),
    });
  }
  meanReturn.push(evaluate(Float64Array.from(mean), 1)[0]);
  return { mean, std, iterations, meanReturn };
}
