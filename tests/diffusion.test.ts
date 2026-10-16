import { describe, expect, it } from 'vitest';
import { Graph, Rng, Tensor } from '../src/engine';
import ddpm from '../src/models/diffusion/ddpm';
import ddim from '../src/models/diffusion/ddim';
import { Session } from '../src/runtime/session';
import { fetchCheckpoint } from '../src/runtime/checkpoint';
import { fsLoader } from './helpers';
import {
  alphaBarAt,
  ddimSigma,
  ddimTimesteps,
  makeSchedule,
  qSample,
  qStep,
} from '../src/models/diffusion/schedule';
import {
  ancestralSample,
  ddimMean,
  ddimSample,
  gaussian,
  mmd2,
  predictEps,
  timeEmbedding,
  toyData,
  type ToyConfig,
} from '../src/models/diffusion/toy';

const noop = { shouldStop: () => false, progress: () => {} };
const tiny = { width: 16, depth: 1, tdim: 4, T: 20 };

/** ε that is exactly optimal when the data are N(0, sd² I). */
function gaussianEps(s: ReturnType<typeof makeSchedule>, sd: number) {
  return (x: Tensor, t: number) => {
    const ab = s.alphaBar[t - 1];
    const k = Math.sqrt(1 - ab) / (ab * sd * sd + 1 - ab);
    return new Tensor(
      x.shape,
      Float32Array.from(x.data, (v) => k * v),
    );
  };
}

function rms(x: ArrayLike<number>): number {
  let a = 0;
  for (let i = 0; i < x.length; i++) a += x[i] * x[i];
  return Math.sqrt(a / x.length);
}

describe('diffusion family', () => {
  it.each([ddpm, ddim])('$id: backpropagation matches finite differences', (m) => {
    const s = new Session(m, { config: tiny });
    s.batchSize = 4;
    const res = s.gradCheck(undefined, 3);
    expect(res.maxRelError).toBeLessThan(1e-5);
  });

  it('timestep and noise come from the seeded graph generator', () => {
    const s = new Session(ddpm, { config: tiny });
    const batch = s.dataset.train(new Rng(4), 8, s.config);
    const loss = (seed: number) =>
      new Graph({ params: s.params, record: false, grad: false, rng: new Rng(seed) }).run(() =>
        ddpm.forward(batch, s.config, { training: true, rng: new Rng(seed) }),
      ).loss!.value.data[0];
    expect(loss(5)).toBe(loss(5));
    expect(loss(5)).not.toBe(loss(6));
  });

  it('linear and cosine schedules have the published values', () => {
    const lin = makeSchedule('linear', 1000);
    expect(lin.beta[0]).toBeCloseTo(1e-4, 12);
    expect(lin.beta[999]).toBeCloseTo(0.02, 12);
    let prod = 1;
    for (let t = 1; t <= 1000; t++) {
      prod *= 1 - lin.beta[t - 1];
      expect(lin.alphaBar[t - 1]).toBeCloseTo(prod, 12);
    }
    expect(lin.alphaBar[999]).toBeCloseTo(4.0358e-5, 8);
    const cos = makeSchedule('cosine', 1000);
    const f = (t: number) => Math.cos(((t / 1000 + 0.008) / 1.008) * (Math.PI / 2)) ** 2;
    // ᾱ_t = f(t)/f(0) wherever β_t was not clipped at 0.999 (all but the last step).
    for (const t of [1, 10, 250, 500, 750, 990])
      expect(cos.alphaBar[t - 1]).toBeCloseTo(f(t) / f(0), 10);
    expect(cos.beta[999]).toBe(0.999);
    // Posterior variance and SNR.
    for (const t of [2, 50, 999]) {
      expect(lin.betaTilde[t - 1]).toBeCloseTo(
        ((1 - lin.alphaBar[t - 2]) / (1 - lin.alphaBar[t - 1])) * lin.beta[t - 1],
        14,
      );
      expect(lin.snr[t - 1]).toBeCloseTo(lin.alphaBar[t - 1] / (1 - lin.alphaBar[t - 1]), 10);
    }
    // Rescaled linear schedule keeps ᾱ_T near zero for small T.
    expect(makeSchedule('linear', 100).alphaBar[99]).toBeLessThan(1e-4);
  });

  it('q(x_t | x_0) in closed form matches iterating q(x_t | x_{t−1})', () => {
    for (const kind of ['linear', 'cosine'] as const) {
      const s = makeSchedule(kind, 100);
      const t = 30;
      const n = 20000;
      const x0 = 1.3;
      const rng = new Rng(9);
      const iter = new Float64Array(n).fill(x0);
      for (let k = 1; k <= t; k++) qStep(s, iter, k, rng);
      const eps = Float64Array.from({ length: n }, () => rng.normal());
      const closed = new Float64Array(n);
      qSample(s, new Float64Array(n).fill(x0), t, eps, closed);
      const ab = alphaBarAt(s, t);
      for (const xs of [iter, closed]) {
        const mean = xs.reduce((a, b) => a + b, 0) / n;
        const v = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1);
        // Standard errors: √(var/n) for the mean, var·√(2/n) for the variance.
        expect(Math.abs(mean - Math.sqrt(ab) * x0)).toBeLessThan(4 * Math.sqrt((1 - ab) / n));
        expect(Math.abs(v - (1 - ab))).toBeLessThan(4 * (1 - ab) * Math.sqrt(2 / n));
      }
    }
  });

  it('DDIM with η = 1 over the full sequence has the DDPM posterior variance and mean', () => {
    for (const kind of ['linear', 'cosine'] as const) {
      const s = makeSchedule(kind, 200);
      const taus = ddimTimesteps(200, 200);
      expect(taus).toEqual(Array.from({ length: 200 }, (_, i) => i + 1));
      for (let t = 2; t <= 200; t++) {
        const abT = alphaBarAt(s, t);
        const abPrev = alphaBarAt(s, t - 1);
        const sigma = ddimSigma(abT, abPrev, 1);
        expect(sigma * sigma).toBeCloseTo(s.betaTilde[t - 1], 12);
        // Posterior mean μ̃(x_t, x̂₀) of q(x_{t−1} | x_t, x₀) with x̂₀ implied by (x_t, ε).
        const xt = 0.7;
        const e = -0.4;
        const x0 = (xt - Math.sqrt(1 - abT) * e) / Math.sqrt(abT);
        const muTilde =
          ((Math.sqrt(abPrev) * s.beta[t - 1]) / (1 - abT)) * x0 +
          ((Math.sqrt(s.alpha[t - 1]) * (1 - abPrev)) / (1 - abT)) * xt;
        expect(ddimMean(xt, e, abT, abPrev, 1).mean).toBeCloseTo(muTilde, 10);
      }
      // η = 0 has no noise at all.
      expect(ddimSigma(alphaBarAt(s, 50), alphaBarAt(s, 40), 0)).toBe(0);
    }
  });

  it('samplers recover N(0, sd²) when given the exact noise predictor', () => {
    const s = makeSchedule('linear', 200);
    const eps = gaussianEps(s, 0.5);
    const xT = gaussian([4000, 2], new Rng(2));
    expect(rms(ancestralSample(eps, xT, s, 'beta', new Rng(3)).x.data)).toBeCloseTo(0.5, 1);
    expect(rms(ancestralSample(eps, xT, s, 'betaTilde', new Rng(3)).x.data)).toBeCloseTo(0.5, 1);
    expect(rms(ddimSample(eps, xT, s, ddimTimesteps(200, 200), 0, new Rng(3)).x.data)).toBeCloseTo(
      0.5,
      1,
    );
    expect(rms(ddimSample(eps, xT, s, ddimTimesteps(200, 200), 1, new Rng(3)).x.data)).toBeCloseTo(
      0.5,
      1,
    );
  });

  it('DDIM with η = 0 is a deterministic function of x_T', () => {
    const sess = new Session(ddim, { config: tiny, seed: 3 });
    const c = sess.config as unknown as ToyConfig;
    const sched = makeSchedule(c.schedule, c.T);
    const xT = gaussian([32, 2], new Rng(1));
    const run = (eta: number, seed: number) =>
      new Graph({ params: sess.params, record: false, grad: false }).run(
        () =>
          ddimSample(
            (x, t) => predictEps(x, t, c),
            xT,
            sched,
            ddimTimesteps(c.T, 7),
            eta,
            new Rng(seed),
          ).x.data,
      );
    expect(Array.from(run(0, 1))).toEqual(Array.from(run(0, 2)));
    expect(Array.from(run(0.5, 1))).not.toEqual(Array.from(run(0.5, 2)));
  });

  it('sinusoidal time embedding has unit-norm sine/cosine pairs', () => {
    const e = timeEmbedding([1, 37], 100, 8);
    for (let n = 0; n < 2; n++)
      for (let k = 0; k < 4; k++)
        expect(e.data[n * 8 + k] ** 2 + e.data[n * 8 + 4 + k] ** 2).toBeCloseTo(1, 6);
  });

  it('training lowers the validation loss and the procedures return the declared tensors', () => {
    const s = new Session(ddpm, { seed: 2 });
    const before = s.evaluate()!.loss;
    for (let i = 0; i < 150; i++) s.trainStep();
    const after = s.evaluate()!.loss;
    expect(after).toBeLessThan(before * 0.4);
    const tr = s.inspect({ kind: 'none' });
    const ts = tr.nodes.find((n) => n.id === tr.outputs.timestep)!.value;
    expect(ts.length).toBe(100);
    expect(ts[0]).toBe(1);
    expect(ts[99]).toBe(100);
    const smp = s.runProcedure('sample', { count: 32, fieldT: 3 }, noop);
    expect(smp.tensors.samples.shape).toEqual([32, 2]);
    expect(smp.tensors.trajectories.shape).toEqual([24, (s.config.T as number) + 1, 2]);
    expect(smp.tensors.field.shape).toEqual([21, 21, 2]);
    expect(smp.scalars!.NFE).toBe(s.config.T);
    const q = s.runProcedure('diffuse', { t: 40 }, noop);
    expect(q.tensors.closed.shape).toEqual([400, 2]);
    const v = q.scalars!['1 − ᾱ_t'];
    const se = v * Math.sqrt(2 / 799);
    expect(Math.abs(q.scalars!['var (closed)'] - v)).toBeLessThan(4 * se);
    expect(Math.abs(q.scalars!['var (iterated)'] - v)).toBeLessThan(4 * se);
  });

  it('DDIM procedures: S network evaluations, sweep over step counts, cancellation', () => {
    const s = new Session(ddim, { config: { ...tiny, steps: 5 } });
    const r = s.runProcedure('sample', { count: 16, fieldT: 2 }, noop);
    expect(r.scalars!.NFE).toBe(5);
    expect(r.tensors.trajectories.shape).toEqual([16, 6, 2]);
    const sw = s.runProcedure('sweep', {}, noop);
    for (const k of ['steps2', 'steps5', 'steps10', 'stepsT', 'eta0', 'eta1'])
      expect(sw.tensors[k].shape).toEqual([800, 2]);
    // MMD² curve against the real step counts S (T = 20 here).
    expect(Array.from(sw.tensors.S.data)).toEqual([1, 2, 3, 5, 10, 20]);
    expect(sw.tensors.mmd.shape).toEqual([6]);
    expect(sw.tensors.floor.shape).toEqual([6]);
    let calls = 0;
    const stopped = s.runProcedure(
      'sample',
      { count: 16, fieldT: 2 },
      { shouldStop: () => ++calls > 2, progress: () => {} },
    );
    expect(stopped.scalars!.NFE).toBe(2);
    expect(stopped.scalars!['stopped at t']).toBeDefined();
    expect(ddim.validate!({ ...ddim.defaults, T: 10, steps: 20 })).toMatch(/at most 10 steps/);
  });

  it('the shipped checkpoint loads into DDPM and DDIM and samples the two moons', async () => {
    for (const m of [ddpm, ddim]) {
      const s = new Session(m);
      const ck = m.checkpoints![0];
      const { file, data } = await fetchCheckpoint(fsLoader, ck.path);
      s.setConfig(file.config);
      s.loadCheckpoint(ck.id, file.params, data);
      const c = s.config as unknown as ToyConfig;
      const xT = gaussian([400, 2], new Rng(7));
      const x = new Graph({ params: s.params, record: false, grad: false }).run(
        () =>
          ddimSample(
            (v, t) => predictEps(v, t, c),
            xT,
            makeSchedule(c.schedule, c.T),
            ddimTimesteps(c.T, 20),
            0,
            new Rng(8),
          ).x,
      );
      const data2 = toyData(c, 'show').x;
      // Noise is ≈ 0.04 from the data; two independent data sets are ≈ 0.005 apart.
      expect(mmd2(x.data, data2, 2)).toBeLessThan(0.012);
    }
  });
});
