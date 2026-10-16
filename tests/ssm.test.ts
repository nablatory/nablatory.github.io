import { describe, expect, it } from 'vitest';
import {
  Graph,
  ParamStore,
  Rng,
  Tensor,
  add,
  backward,
  exp,
  gradCheck,
  mul,
  p,
  reshape,
  select,
  stack,
  sum,
  withDType,
  type Var,
} from '../src/engine';
import { constant } from '../src/models/util';
import linearSsm, { forwardLinearSsm, type LinearSsmConfig } from '../src/models/ssm/linear-ssm';
import mamba, { forwardMamba, type MambaConfig } from '../src/models/ssm/mamba';
import { selectiveScan } from '../src/models/ssm/ops';
import { Session } from '../src/runtime/session';

function rand(shape: number[], rng: Rng, lo = -1, hi = 1): Tensor {
  const t = Tensor.zeros(shape);
  for (let i = 0; i < t.size; i++) t.data[i] = rng.range(lo, hi);
  return t;
}

/** The selective scan written out with ordinary engine ops, one step at a time. */
function naiveScan(delta: Var, A: Var, B: Var, x: Var): Var {
  const [Bt, L, E] = delta.shape;
  const N = A.shape[1];
  let h: Var | null = null;
  const hs: Var[] = [];
  for (let t = 0; t < L; t++) {
    const dt = reshape(select(delta, 1, t), [Bt, E, 1]);
    const u = mul(
      mul(dt, reshape(select(x, 1, t), [Bt, E, 1])),
      reshape(select(B, 1, t), [Bt, 1, N]),
    );
    h = h ? add(mul(exp(mul(dt, A)), h), u) : u;
    hs.push(h);
  }
  return stack(hs, 1);
}

const smallMamba = { span: 5, memorize: 2, dModel: 4, layers: 2, dState: 3 };

describe('state space models', () => {
  it.each([
    [linearSsm, { length: 10, modes: 3, lag: 4, mode: 'convolution' }],
    [linearSsm, { length: 10, modes: 3, lag: 4, mode: 'recurrent', filter: 'delay' }],
    [mamba, { ...smallMamba, selective: true }],
    [mamba, { ...smallMamba, selective: false }],
  ] as const)('%#: backpropagation matches finite differences', (m, cfg) => {
    const s = new Session(m, { config: cfg });
    s.batchSize = 2;
    const res = s.gradCheck(undefined, 3);
    expect(res.maxRelError).toBeLessThan(1e-5);
  });

  it('linear SSM: recurrent and convolutional forms agree', () => {
    for (const filter of ['resonator', 'delay'] as const) {
      const s = new Session(linearSsm, {
        seed: 5,
        config: { filter, lag: 6, length: 40, modes: 6, dtInit: 0.2 },
      });
      // Move away from the initial values so every parameter matters.
      for (let i = 0; i < 20; i++) s.trainStep();
      const batch = s.dataset.train(new Rng(3), 4, s.config);
      withDType('float64', () => {
        const params = s.params.clone('float64');
        const run = (mode: 'recurrent' | 'convolution') =>
          new Graph({ params, grad: false }).run(() =>
            forwardLinearSsm(batch, s.config as LinearSsmConfig, mode),
          ).outputs.prediction.value;
        const a = run('recurrent');
        const b = run('convolution');
        let maxAbs = 0;
        for (let i = 0; i < a.size; i++) maxAbs = Math.max(maxAbs, Math.abs(a.data[i] - b.data[i]));
        expect(maxAbs).toBeLessThan(1e-10);
      });
      // In float32 the two orders of summation still agree closely.
      s.setConfig({ mode: 'recurrent' });
      const r = s.inspect({ kind: 'none' }, batch);
      s.setConfig({ mode: 'convolution' });
      const c = s.inspect({ kind: 'none' }, batch);
      const yr = r.nodes.find((n) => n.id === r.outputs.prediction)!.value;
      const yc = c.nodes.find((n) => n.id === c.outputs.prediction)!.value;
      for (let i = 0; i < yr.length; i++) expect(Math.abs(yr[i] - yc[i])).toBeLessThan(1e-4);
    }
  });

  it('linear SSM: the impulse response of the recurrence is the kernel K', () => {
    withDType('float64', () => {
      const cfg = { ...linearSsm.defaults, length: 32, modes: 5 } as LinearSsmConfig;
      const params = linearSsm.init(cfg, new Rng(2)).clone('float64');
      const u = Tensor.zeros([1, 32, 1]);
      u.data[0] = 1;
      const out = new Graph({ params, grad: false }).run(() =>
        forwardLinearSsm({ inputs: { u } }, cfg, 'recurrent'),
      ).outputs;
      for (let k = 0; k < 32; k++)
        expect(out.prediction.value.data[k]).toBeCloseTo(out.kernelEffective.value.data[k], 12);
    });
  });

  it('linear SSM: learns the resonator kernel', () => {
    const s = new Session(linearSsm, { seed: 1, config: { length: 48 } });
    const before = s.evaluate()!;
    for (let i = 0; i < 150; i++) s.trainStep();
    const after = s.evaluate()!;
    expect(after.loss).toBeLessThan(before.loss * 0.1);
    expect(after.metrics['relative kernel error']).toBeLessThan(0.15);
  });

  it('selective scan matches a naive loop of engine ops, forward and backward', () => {
    withDType('float64', () => {
      const rng = new Rng(9);
      const [Bt, L, E, N] = [2, 5, 3, 4];
      const values = {
        delta: rand([Bt, L, E], rng, 0.05, 1),
        A: rand([E, N], rng, -2, -0.1),
        B: rand([Bt, L, N], rng),
        x: rand([Bt, L, E], rng),
      };
      const w = rand([Bt, L, E, N], rng);
      const [fast, slow] = [selectiveScan, naiveScan].map((fn) => {
        const g = new Graph({ grad: true });
        return g.run(() => {
          const v = Object.fromEntries(
            Object.entries(values).map(([k, t]) => [k, g.input(k, t, true)]),
          ) as Record<keyof typeof values, Var>;
          const h = fn(v.delta, v.A, v.B, v.x);
          backward(sum(mul(h, g.constant(w))));
          return { h, v };
        });
      });
      for (let i = 0; i < fast.h.value.size; i++)
        expect(fast.h.value.data[i]).toBeCloseTo(slow.h.value.data[i], 12);
      for (const k of ['delta', 'A', 'B', 'x'] as const) {
        const gf = fast.v[k].grad!.data;
        const gs = slow.v[k].grad!.data;
        for (let i = 0; i < gf.length; i++) expect(gf[i]).toBeCloseTo(gs[i], 10);
      }
    });
  });

  it('selective scan VJP matches central differences', () => {
    const rng = new Rng(4);
    const [Bt, L, E, N] = [1, 4, 2, 3];
    const store = new ParamStore();
    store.add('delta', rand([Bt, L, E], rng, 0.1, 1), 'test');
    store.add('A', rand([E, N], rng, -2, -0.1), 'test');
    store.add('B', rand([Bt, L, N], rng), 'test');
    store.add('x', rand([Bt, L, E], rng), 'test');
    const w = rand([Bt, L, E, N], rng);
    const res = gradCheck(
      () => sum(mul(selectiveScan(p('delta'), p('A'), p('B'), p('x')), constant(w))),
      store,
      { perParam: 6 },
    );
    expect(res.maxRelError).toBeLessThan(1e-6);
  });

  it('Mamba: Δ depends on the token only when selection is on', () => {
    for (const selective of [true, false]) {
      const s = new Session(mamba, { seed: 3, config: { selective } });
      const t = s.inspect({ kind: 'none' });
      const d = t.nodes.find((n) => n.id === t.outputs.delta)!;
      const [, L, E] = d.shape;
      let spread = 0;
      for (let e = 0; e < E; e++)
        for (let k = 1; k < L; k++)
          spread = Math.max(spread, Math.abs(d.value[k * E + e] - d.value[e]));
      if (selective) expect(spread).toBeGreaterThan(1e-4);
      else expect(spread).toBe(0);
    }
  });

  it('Mamba is causal: changing a later token leaves earlier outputs unchanged', () => {
    const cfg = { ...mamba.defaults } as MambaConfig;
    const params = mamba.init(cfg, new Rng(1));
    const batch = mamba.datasets[0].example(0, cfg);
    const ids2 = batch.inputs.ids.clone();
    ids2.data[8] = (ids2.data[8] + 1) % 4;
    const run = (ids: Tensor) =>
      new Graph({ params, grad: false }).run(() => forwardMamba({ inputs: { ids } }, cfg)).outputs
        .logits.value;
    const a = run(batch.inputs.ids);
    const b = run(ids2);
    const V = a.shape[2];
    for (let i = 0; i < 8 * V; i++) expect(a.data[i]).toBe(b.data[i]);
    let changed = 0;
    for (let i = 8 * V; i < a.size; i++)
      changed = Math.max(changed, Math.abs(a.data[i] - b.data[i]));
    expect(changed).toBeGreaterThan(0);
  });

  it('Mamba: training lowers the validation loss on selective copying', () => {
    const s = new Session(mamba, { seed: 2, config: { span: 8, memorize: 2, layers: 1 } });
    const before = s.evaluate()!.loss;
    for (let i = 0; i < 100; i++) s.trainStep();
    expect(s.evaluate()!.loss).toBeLessThan(before * 0.75);
  });
});
