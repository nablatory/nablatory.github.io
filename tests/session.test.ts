import { describe, expect, it } from 'vitest';
import vgg from '../src/models/convolutional/vgg';
import mlp from '../src/models/feedforward/mlp';
import gan from '../src/models/gan/gan';
import dreamer from '../src/models/world-model/dreamer';
import tdmpc from '../src/models/world-model/tdmpc';
import { Session } from '../src/runtime/session';
import { fsLoader } from './helpers';
import {
  ParamStore,
  Tensor,
  activeGraph,
  add,
  gradCheck,
  matmul,
  relu,
  sum,
  tanh,
} from '../src/engine';
import { constant } from '../src/models/util';

const param = (name: string) => activeGraph()!.param(name);

const runningStats = (s: Session) =>
  s.params
    .names()
    .filter((n) => n.includes('running'))
    .map((n) => Float64Array.from(s.params.get(n).data));

describe('Session', () => {
  it('a training-mode inspection leaves batch-norm running statistics unchanged', async () => {
    const s = new Session(vgg, { config: { batchNorm: true, dropout: 0 } });
    await s.loadData(fsLoader);
    const before = runningStats(s);
    expect(before.length).toBeGreaterThan(0);
    s.inspect({ kind: 'loss' }, s.inspectionBatch(), true);
    runningStats(s).forEach((d, i) => expect(Array.from(d)).toEqual(Array.from(before[i])));
  });

  it('one detailed step updates running statistics exactly once', async () => {
    const a = new Session(vgg, { config: { batchNorm: true, dropout: 0 } });
    const b = new Session(vgg, { config: { batchNorm: true, dropout: 0 } });
    await a.loadData(fsLoader);
    await b.loadData(fsLoader);
    a.trainStep();
    b.trainStep({ detail: true });
    runningStats(a).forEach((d, i) =>
      expect(Array.from(runningStats(b)[i])).toEqual(Array.from(d)),
    );
  });

  it('gradient check in training mode covers batch norm with batch statistics', async () => {
    const s = new Session(vgg, { config: { batchNorm: true, dropout: 0.5 } });
    await s.loadData(fsLoader);
    // Move biases off zero so no unit sits exactly on a ReLU kink.
    for (const n of s.params.names()) if (n.endsWith('.bias')) s.params.get(n).data.fill(0.01);
    const res = s.gradCheck(undefined, 2, undefined, true);
    expect(res.maxRelError).toBeLessThan(1e-5);
  });

  it('training-mode and evaluation-mode checks agree for a model without stochastic layers', () => {
    const s = new Session(mlp);
    expect(s.gradCheck(undefined, 2, undefined, true).maxRelError).toBeLessThan(1e-5);
  });
});

describe('gradient check at non-differentiable points', () => {
  // L(w, b) = Σ_i relu(w·x_i + b) with x = 0 for half the rows: at b = 0 those
  // rows sit exactly on the ReLU kink, so ∂L/∂b does not exist there.
  const X = new Tensor([4, 2], Float32Array.from([0, 0, 0, 0, 1, 2, -1, 0.5]));
  const loss = (smooth: boolean) => () => {
    const z = add(matmul(constant(X, 'x'), param('w')), param('b'));
    return sum(smooth ? tanh(z) : relu(z));
  };
  const store = () => {
    const s = new ParamStore();
    s.add('w', new Tensor([2, 1], Float32Array.from([0.3, -0.7])), 'test');
    s.add('b', new Tensor([1], Float32Array.from([0])), 'test');
    return s;
  };

  it('classifies an entry on a ReLU kink, with one-sided differences that bracket the jump', () => {
    const r = gradCheck(loss(false), store(), { names: ['b'], perParam: 1 });
    const e = r.entries[0];
    expect(e.pass).toBe(false);
    expect(e.kink).toBe(true);
    // relu'(0) = 0 in the engine: the tape value equals the left-sided difference.
    expect(Math.abs(e.analytic - e.backward!)).toBeLessThan(1e-6);
    expect(Math.abs(e.forward! - e.backward!)).toBeGreaterThan(1.5);
    expect(r.kinks).toBe(1);
    expect(r.pass).toBe(false); // no entry agrees, so the check cannot pass on kinks alone
  });

  it('never classifies a smooth loss as a kink', () => {
    const r = gradCheck(loss(true), store(), { perParam: 2 });
    expect(r.kinks).toBe(0);
    expect(r.pass).toBe(true);
  });
});

describe('Session.gradCheck', () => {
  it('checks every training phase on the parameters that phase updates', async () => {
    for (const m of [gan, dreamer, tdmpc]) {
      const s = new Session(m);
      await s.loadData(fsLoader);
      const r = s.gradCheck(undefined, 2);
      expect(r.pass, m.id).toBe(true);
      expect(r.maxRelError, m.id).toBeLessThan(1e-5);
      for (const phase of m.training.phases!)
        expect(
          r.entries.some((e) => phase.params(e.param)),
          `${m.id}: ${phase.id}`,
        ).toBe(true);
    }
  }, 60_000);
});
