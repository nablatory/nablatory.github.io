/**
 * Fixed-step explicit Runge–Kutta solvers built from engine ops, so reverse
 * mode differentiates through every stage of the solver
 * ("discretise-then-optimise"). The continuous adjoint method of Chen et al.
 * (2018) is not used.
 */
import { add, axpby, scope, type Var } from '../../engine';

export type OdeMethod = 'euler' | 'midpoint' | 'rk4';

export const ODE_METHODS: { value: OdeMethod; label: string }[] = [
  { value: 'euler', label: 'Euler (order 1)' },
  { value: 'midpoint', label: 'Midpoint (order 2)' },
  { value: 'rk4', label: 'Classical RK4 (order 4)' },
];

/** Function evaluations per step. */
export const STAGES: Record<OdeMethod, number> = { euler: 1, midpoint: 2, rk4: 4 };

export const METHOD_TEX: Record<OdeMethod, string> = {
  euler: 'h_{k+1} = h_k + \\Delta t\\, f(h_k, t_k)',
  midpoint:
    'h_{k+1} = h_k + \\Delta t\\, f\\big(h_k + \\tfrac{\\Delta t}{2} f(h_k, t_k),\\ t_k + \\tfrac{\\Delta t}{2}\\big)',
  rk4: 'h_{k+1} = h_k + \\tfrac{\\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)',
};

export type VectorField = (h: Var, t: number) => Var;

/** One step of size dt from (h, t). */
export function odeStep(f: VectorField, h: Var, t: number, dt: number, method: OdeMethod): Var {
  switch (method) {
    case 'euler':
      return axpby(1, h, dt, f(h, t));
    case 'midpoint': {
      const k1 = f(h, t);
      const k2 = f(axpby(1, h, dt / 2, k1), t + dt / 2);
      return axpby(1, h, dt, k2);
    }
    case 'rk4': {
      const k1 = f(h, t);
      const k2 = f(axpby(1, h, dt / 2, k1), t + dt / 2);
      const k3 = f(axpby(1, h, dt / 2, k2), t + dt / 2);
      const k4 = f(axpby(1, h, dt, k3), t + dt);
      const incr = add(axpby(1, k1, 2, k2), axpby(2, k3, 1, k4));
      return axpby(1, h, dt / 6, incr);
    }
  }
}

export interface OdeSolution {
  /** State at t1. */
  h: Var;
  /** States at t0 and after every step (steps + 1 entries). */
  states: Var[];
  /** Number of evaluations of f. */
  nfe: number;
}

/** Integrate dh/dt = f(h, t) from t0 to t1 with `steps` equal steps. */
export function odeSolve(
  f: VectorField,
  h0: Var,
  t0: number,
  t1: number,
  steps: number,
  method: OdeMethod,
  opts: { scopeName?: string; keep?: boolean } = {},
): OdeSolution {
  const dt = (t1 - t0) / steps;
  const states: Var[] = [h0];
  let h = h0;
  for (let k = 0; k < steps; k++) {
    const t = t0 + k * dt;
    const prev = h;
    h = opts.scopeName
      ? scope(opts.scopeName, () => odeStep(f, prev, t, dt, method))
      : odeStep(f, prev, t, dt, method);
    if (opts.keep !== false) states.push(h);
  }
  return { h, states, nfe: steps * STAGES[method] };
}
