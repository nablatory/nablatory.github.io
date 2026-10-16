import { describe, expect, it } from 'vitest';
import lenet5, { C3_TABLE } from '../src/models/convolutional/lenet5';
import { Session, effectiveCount } from '../src/runtime/session';
import { fsLoader } from './helpers';
import { fetchCheckpoint } from '../src/runtime/checkpoint';

describe('LeNet-5', () => {
  it('has the published parameter counts', () => {
    const s = new Session(lenet5);
    const count = (prefix: string) => effectiveCount(s.params, (n) => n.startsWith(prefix));
    expect(count('c1.')).toBe(156);
    expect(count('s2.')).toBe(12);
    expect(count('c3.')).toBe(1516);
    expect(count('s4.')).toBe(32);
    expect(count('c5.')).toBe(48120);
    expect(count('f6.')).toBe(10164);
    expect(C3_TABLE.flat().length).toBe(60);
  });

  it('propagates shapes and rejects geometries with no spatial extent', () => {
    expect(lenet5.validate!({ ...lenet5.defaults, c1Stride: 2 })).toMatch(/no spatial extent/);
    const s = new Session(lenet5, { config: { c1Padding: 2 } });
    expect(s.params.get('f6.weight').shape).toEqual([84, 120 * 2 * 2]);
  });

  it('backpropagation matches finite differences (float64)', async () => {
    const s = new Session(lenet5);
    await s.loadData(fsLoader);
    s.batchSize = 2;
    const res = s.gradCheck(undefined, 2);
    expect(res.maxRelError).toBeLessThan(1e-5);
    expect(res.pass).toBe(true);
  });

  it('masked C3 weights receive zero gradient', async () => {
    const s = new Session(lenet5);
    await s.loadData(fsLoader);
    const trace = s.inspect();
    const w = trace.nodes.find((n) => n.key === 'param:c3.weight')!;
    // map 0 reads S2 maps {0,1,2}: input channel 3 is disconnected.
    for (let k = 0; k < 25; k++) expect(Math.abs(w.grad![0 * 150 + 3 * 25 + k])).toBe(0);
    expect(Math.abs(w.grad![0 * 150 + 0 * 25 + 12])).toBeGreaterThan(0);
  });

  it('training reduces the loss on MNIST', async () => {
    const s = new Session(lenet5, { seed: 3 });
    await s.loadData(fsLoader);
    const before = s.evaluate()!.loss;
    for (let i = 0; i < 60; i++) s.trainStep();
    const after = s.evaluate()!.loss;
    expect(after).toBeLessThan(before * 0.6);
  });

  it('one detailed step reports the update actually applied', async () => {
    const s = new Session(lenet5);
    await s.loadData(fsLoader);
    s.setOptimizer({ kind: 'sgd', lr: 0.05 });
    const w0 = s.params.get('f6.bias').clone();
    const d = s.trainStep({ detail: true });
    expect('before' in d).toBe(true);
    if (!('before' in d)) return;
    const g = d.before.nodes.find((n) => n.key === 'param:f6.bias')!.grad!;
    const w1 = s.params.get('f6.bias');
    for (let i = 0; i < 84; i++) {
      expect(w1.data[i] - w0.data[i]).toBeCloseTo(-0.05 * g[i], 6);
      expect(d.updates['f6.bias'].data[i]).toBeCloseTo(-0.05 * g[i], 6);
    }
    expect(d.after.loss).toBeDefined();
  });
});

describe('LeNet-5 checkpoint', () => {
  it('loads and classifies the shipped test images about as well as its record states', async () => {
    const s = new Session(lenet5);
    await s.loadData(fsLoader);
    const ck = lenet5.checkpoints![0];
    const { file, data } = await fetchCheckpoint(fsLoader, ck.path);
    s.loadCheckpoint(ck.id, file.params, data);
    const e = s.evaluate()!;
    // The record is on all 10,000 MNIST test images; the site ships the first 1,000.
    const rec = file.provenance!.metrics as Record<string, number>;
    expect(rec.testAccuracy).toBeGreaterThan(0.98);
    expect(e.metrics.accuracy).toBeGreaterThan(rec.testAccuracy - 0.015);
    expect(e.loss).toBeLessThan(rec.testLoss * 1.6);
  });
});
