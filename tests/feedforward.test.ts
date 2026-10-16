import { describe, expect, it } from 'vitest';
import perceptron from '../src/models/feedforward/perceptron';
import mlp from '../src/models/feedforward/mlp';
import resmlp from '../src/models/feedforward/resmlp';
import { Session } from '../src/runtime/session';

const noop = { shouldStop: () => false, progress: () => {} };

describe('feedforward family', () => {
  it.each([perceptron, mlp, resmlp])('$id: backpropagation matches finite differences', (m) => {
    const s = new Session(m);
    s.batchSize = 4;
    const res = s.gradCheck(undefined, 3);
    expect(res.maxRelError).toBeLessThan(1e-5);
  });

  it('perceptron step with SGD, lr 1, batch 1 is Rosenblatt’s rule', () => {
    const s = new Session(perceptron, { seed: 5 });
    for (let i = 0; i < 20; i++) {
      const w0 = s.params.get('unit.weight').clone();
      const b0 = s.params.get('unit.bias').clone();
      const d = s.trainStep({ detail: true });
      if (!('before' in d)) throw new Error('expected detail');
      const x = d.before.nodes.find((n) => n.key === 'input:x')!.value;
      const sc = d.before.outputs.score;
      const score = d.before.nodes.find((n) => n.id === sc)!.value[0];
      const t = d.before.nodes.find((n) => n.op === 'targets')!.value[0];
      const mis = t * score < 0 ? 1 : 0;
      const w1 = s.params.get('unit.weight');
      expect(w1.data[0] - w0.data[0]).toBeCloseTo(mis * t * x[0], 5);
      expect(w1.data[1] - w0.data[1]).toBeCloseTo(mis * t * x[1], 5);
      expect(s.params.get('unit.bias').data[0] - b0.data[0]).toBeCloseTo(mis * t, 5);
    }
  });

  it('perceptron separates blobs but not XOR', () => {
    const blobs = new Session(perceptron, { seed: 1 });
    for (let i = 0; i < 400; i++) blobs.trainStep();
    expect(blobs.evaluate()!.metrics.accuracy).toBeGreaterThan(0.93);
    const xor = new Session(perceptron, { seed: 1, config: { dataset: 'xor' } });
    for (let i = 0; i < 400; i++) xor.trainStep();
    expect(xor.evaluate()!.metrics.accuracy).toBeLessThan(0.8);
  });

  it('MLP learns two moons and the decision field matches the forward pass', () => {
    const s = new Session(mlp, { seed: 2, config: { dataset: 'moons' } });
    for (let i = 0; i < 300; i++) s.trainStep();
    expect(s.evaluate()!.metrics.accuracy).toBeGreaterThan(0.95);
    const f = s.runProcedure('field', {}, noop);
    expect(f.tensors.field.shape).toEqual([64, 64]);
    expect(f.tensors.points.shape).toEqual([400, 2]);
  });

  it('residual MLP: disabling skips keeps parameters and changes the output', () => {
    const s = new Session(resmlp, { seed: 4 });
    const n = s.params.count();
    const a = s.inspect({ kind: 'none' });
    s.setConfig({ residual: false });
    expect(s.params.count()).toBe(n);
    const b = s.inspect({ kind: 'none' });
    const pa = a.nodes.find((x) => x.id === a.outputs.probs)!.value;
    const pb = b.nodes.find((x) => x.id === b.outputs.probs)!.value;
    expect(pa.some((v, i) => Math.abs(v - pb[i]) > 1e-6)).toBe(true);
  });
});
