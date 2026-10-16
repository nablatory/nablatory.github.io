import { withDType } from './tensor';
import { Graph, backward, type Var } from './graph';
import type { ParamStore } from './params';
import { Rng } from './rng';

export interface GradCheckEntry {
  param: string;
  index: number;
  analytic: number;
  numeric: number;
  absError: number;
  relError: number;
  /** Agrees with the central difference within the tolerance. */
  pass: boolean;
  /**
   * The loss is not differentiable here: the one-sided differences
   * (L(θ+ε) − L(θ))/ε and (L(θ) − L(θ−ε))/ε disagree, and the backpropagated
   * value equals one of them (a ReLU, leaky ReLU, max or |·| switching branch
   * within ±ε). Backpropagation returns a valid one-sided derivative there,
   * and the central difference, which averages the two sides, is not a
   * reference for it.
   */
  kink: boolean;
  /** One-sided differences, reported for kinks. */
  forward?: number;
  backward?: number;
}

export interface GradCheckResult {
  entries: GradCheckEntry[];
  /** Over entries that are not at a kink. */
  maxRelError: number;
  /** Every entry agrees or is at a kink, and at least one agrees. */
  pass: boolean;
  kinks: number;
  eps: number;
  dtype: 'float64';
}

/**
 * Compare reverse-mode gradients with central finite differences,
 * (L(θ+ε) − L(θ−ε)) / 2ε, evaluated in float64 to keep truncation and
 * rounding error well below the tolerance.
 *
 * `lossFn` builds the scalar loss inside the active graph. It must be
 * deterministic (no dropout sampling that depends on call count).
 */
export function gradCheck(
  lossFn: () => Var,
  params: ParamStore,
  opts: {
    names?: string[];
    perParam?: number;
    eps?: number;
    tol?: number;
    seed?: number;
    training?: boolean;
  } = {},
): GradCheckResult {
  const eps = opts.eps ?? 1e-6;
  const tol = opts.tol ?? 1e-5;
  const perParam = opts.perParam ?? 4;
  const rng = new Rng(opts.seed ?? 7);
  return withDType('float64', () => {
    const store = params.clone('float64');
    const names = opts.names ?? store.names().filter((n) => store.isTrainable(n));
    const g = new Graph({
      params: store,
      record: false,
      grad: true,
      training: opts.training ?? false,
      rng: new Rng(1),
    });
    const loss = g.run(lossFn);
    backward(loss);
    let l0: number | undefined;
    const entries: GradCheckEntry[] = [];
    const evalLoss = () =>
      new Graph({
        params: store,
        record: false,
        grad: false,
        training: opts.training ?? false,
        rng: new Rng(1),
      }).run(lossFn).value.data[0];
    for (const name of names) {
      const v = g.paramVar(name);
      const t = store.get(name);
      const picks = new Set<number>();
      const count = Math.min(perParam, t.size);
      while (picks.size < count) picks.add(rng.int(t.size));
      for (const i of picks) {
        const analytic = v?.grad ? v.grad.data[i] : 0;
        const orig = t.data[i];
        t.data[i] = orig + eps;
        const lp = evalLoss();
        t.data[i] = orig - eps;
        const lm = evalLoss();
        t.data[i] = orig;
        const numeric = (lp - lm) / (2 * eps);
        const absError = Math.abs(analytic - numeric);
        const relError = absError / Math.max(Math.abs(analytic), Math.abs(numeric), 1e-4);
        const pass = relError < tol || absError < 1e-8;
        const entry: GradCheckEntry = {
          param: name,
          index: i,
          analytic,
          numeric,
          absError,
          relError,
          pass,
          kink: false,
        };
        if (!pass) {
          // Smooth loss: the one-sided differences straddle the derivative and
          // differ by about ε·L''. At a kink they jump, and backpropagation's
          // value sits on one side. Require it to be far closer to one side
          // than the sides are to each other, so round-off noise never qualifies.
          l0 ??= evalLoss();
          const fwd = (lp - l0) / eps;
          const bwd = (l0 - lm) / eps;
          const jump = Math.abs(fwd - bwd);
          const near = Math.min(Math.abs(analytic - fwd), Math.abs(analytic - bwd));
          if (jump > 1e-6 && near <= 0.02 * jump) {
            entry.kink = true;
            entry.forward = fwd;
            entry.backward = bwd;
          }
        }
        entries.push(entry);
      }
    }
    const smooth = entries.filter((e) => !e.kink);
    const maxRelError = smooth.reduce((m, e) => Math.max(m, e.relError), 0);
    return {
      entries,
      maxRelError,
      pass: entries.every((e) => e.pass || e.kink) && entries.some((e) => e.pass),
      kinks: entries.length - smooth.length,
      eps,
      dtype: 'float64' as const,
    };
  });
}
