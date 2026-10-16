import { describe, expect, it } from 'vitest';
import { Graph, Tensor, backward, leaf, mulScalar, sum, withDType, type Var } from '../src/engine';
import neuralOde from '../src/models/neural-dynamics/neural-ode';
import latentOde from '../src/models/neural-dynamics/latent-ode';
import { STAGES, odeSolve, type OdeMethod } from '../src/models/neural-dynamics/solvers';
import { PENDULUM, simulatePendulum } from '../src/models/data/pendulum';
import { planarPoints } from '../src/models/data/planar';
import { Session } from '../src/runtime/session';

const noop = { shouldStop: () => false, progress: () => {} };

/** Global error at t = 1 of dx/dt = −x, x(0) = 1. */
function decayError(method: OdeMethod, steps: number): number {
  const x0 = leaf(new Tensor([1], new Float64Array([1])));
  const sol = odeSolve((x: Var) => mulScalar(x, -1), x0, 0, 1, steps, method);
  return Math.abs(sol.h.value.data[0] - Math.exp(-1));
}

describe('neural dynamics family', () => {
  it('neural ODE: backpropagation through the solver matches finite differences', () => {
    for (const method of ['euler', 'midpoint', 'rk4'] as const) {
      const s = new Session(neuralOde, { config: { width: 6, steps: 3, method, augment: 1 } });
      s.batchSize = 4;
      const res = s.gradCheck(undefined, 3);
      expect(res.maxRelError).toBeLessThan(1e-5);
    }
  });

  it('latent ODE: the default configuration passes the gradient check at the model’s ε', () => {
    // σ = 0.05 and the extrapolation objective; the model sets ε = 1e-4 (see latent-ode.ts) because
    // float64 round-off dominates at ε = 1e-6. The tolerance stays at 1e-5.
    expect(latentOde.gradCheck?.eps).toBe(1e-4);
    const s = new Session(latentOde);
    const res = s.gradCheck(undefined, 3);
    expect(res.eps).toBe(1e-4);
    expect(res.maxRelError).toBeLessThan(1e-5);
    // The reconstruction-only objective on a tiny network also passes.
    const r = new Session(latentOde, {
      config: { encoder: 8, width: 8, latent: 3, substeps: 2, objective: 'reconstruct' },
    });
    r.batchSize = 2;
    expect(r.gradCheck(undefined, 2).maxRelError).toBeLessThan(1e-5);
  });

  it('solvers converge at orders 1, 2 and 4 on dx/dt = −x', () => {
    const ratio = (m: OdeMethod) => decayError(m, 10) / decayError(m, 20);
    expect(ratio('euler')).toBeGreaterThan(1.8);
    expect(ratio('euler')).toBeLessThan(2.2);
    expect(ratio('midpoint')).toBeGreaterThan(3.6);
    expect(ratio('midpoint')).toBeLessThan(4.4);
    expect(ratio('rk4')).toBeGreaterThan(15);
    expect(ratio('rk4')).toBeLessThan(17);
  });

  it('gradient through RK4 equals the derivative of the discrete map', () => {
    // x_N = R(h)^N x_0 with R(h) = 1 − h + h²/2 − h³/6 + h⁴/24 for dx/dt = −x.
    const steps = 5;
    const h = 1 / steps;
    const grad = withDType('float64', () => {
      const g = new Graph({ record: false, grad: true });
      const x0 = g.run(() => leaf(new Tensor([1], new Float64Array([0.8])), true));
      const out = g.run(() =>
        sum(odeSolve((x: Var) => mulScalar(x, -1), x0, 0, 1, steps, 'rk4').h),
      );
      backward(out);
      return x0.grad!.data[0];
    });
    const R = 1 - h + h ** 2 / 2 - h ** 3 / 6 + h ** 4 / 24;
    expect(grad).toBeCloseTo(R ** steps, 12);
  });

  it('reports NFE = steps × stages', () => {
    for (const method of ['euler', 'midpoint', 'rk4'] as const) {
      const s = new Session(neuralOde, { config: { method, steps: 7 } });
      const tr = s.inspect({ kind: 'none' });
      expect(tr.metrics.NFE).toBe(7 * STAGES[method]);
    }
  });

  it('neural ODE learns two moons; the flow procedure matches the forward pass', () => {
    const s = new Session(neuralOde, { seed: 2 });
    for (let i = 0; i < 120; i++) s.trainStep();
    expect(s.evaluate()!.metrics.accuracy).toBeGreaterThan(0.9);
    const f = s.runProcedure('flow', { t: 0.5 }, noop);
    expect(f.tensors.field.shape).toEqual([21, 21, 2]);
    expect(f.tensors.trajectories.shape).toEqual([40, 9, 2]);
    expect(f.tensors.trajectoryLabels.shape).toEqual([40]);
    // Each trajectory starts at the training point whose class it is coloured by.
    const pts = planarPoints('moons', 400, 11, 0.08);
    for (let k = 0; k < 40; k++) {
      const i = Math.floor((k * 400) / 40);
      expect(f.tensors.trajectoryLabels.data[k]).toBe(pts.y[i]);
      expect(f.tensors.trajectories.data[k * 9 * 2]).toBeCloseTo(pts.x[i * 2], 5);
      expect(f.tensors.trajectories.data[k * 9 * 2 + 1]).toBeCloseTo(pts.x[i * 2 + 1], 5);
    }
    // Example 0 is the 400 training points; h(T) from the forward pass equals the procedure's final states.
    const tr = s.inspect({ kind: 'none' });
    const hT = tr.nodes.find((n) => n.id === tr.outputs.hT)!.value;
    for (let i = 0; i < 800; i += 37) expect(f.tensors.final.data[i]).toBeCloseTo(hT[i], 4);
  });

  it('pendulum simulator: energy is conserved without damping and decays with it', () => {
    const energy = (s: Float64Array, k: number) =>
      0.5 * s[k * 2 + 1] ** 2 + Math.PI ** 2 * (1 - Math.cos(s[k * 2]));
    const free = simulatePendulum(1.5, 0.3, 200, { ...PENDULUM, damping: 0 });
    expect(Math.abs(energy(free, 199) - energy(free, 0)) / energy(free, 0)).toBeLessThan(1e-6);
    const damped = simulatePendulum(1.5, 0.3, 200, PENDULUM);
    expect(energy(damped, 199)).toBeLessThan(0.5 * energy(damped, 0));
  });

  it('latent ODE: training lowers the loss; extrapolation covers the full horizon', () => {
    const s = new Session(latentOde, { seed: 1 });
    const before = s.evaluate()!;
    for (let i = 0; i < 120; i++) s.trainStep();
    const after = s.evaluate()!;
    expect(after.loss).toBeLessThan(before.loss * 0.5);
    expect(after.metrics.mse).toBeLessThan(before.metrics.mse * 0.5);
    const r = s.runProcedure('extrapolate', { example: 3 }, noop);
    const L = PENDULUM.observed + PENDULUM.future;
    expect(r.tensors.predicted.shape).toEqual([L, 2]);
    expect(r.tensors.truth.shape).toEqual([L, 2]);
    expect(r.tensors.observed.shape).toEqual([PENDULUM.observed, 2]);
    expect(r.tensors.trajectories.shape).toEqual([8, L, 2]);
    expect(r.scalars!.NFE).toBe((L - 1) * 4);
    expect(L).toBe(60);
    // The portrait extent is fitted to the latent paths and contains them all.
    const [x0, x1, y0, y1] = Array.from(r.tensors.extent.data);
    const tj = r.tensors.trajectories.data;
    for (let i = 0; i < tj.length; i += 2) {
      expect(tj[i]).toBeGreaterThanOrEqual(x0);
      expect(tj[i]).toBeLessThanOrEqual(x1);
      expect(tj[i + 1]).toBeGreaterThanOrEqual(y0);
      expect(tj[i + 1]).toBeLessThanOrEqual(y1);
    }
    expect(x1 - x0).toBeCloseTo(y1 - y0, 4);
  });
});
