import { describe, expect, it } from 'vitest';
import {
  Graph,
  Rng,
  Tensor,
  gruCell,
  leaf,
  lstmCell,
  p,
  rnnCell,
  withDType,
  linear,
  ParamBuilder,
  type Var,
} from '../src/engine';
import rnn, { rnnCellDef } from '../src/models/recurrent/rnn';
import lstm, { lstmCellDef } from '../src/models/recurrent/lstm';
import gru, { gruCellDef } from '../src/models/recurrent/gru';
import { Session } from '../src/runtime/session';
import { buildModules } from '../src/viz/architecture';

const noop = { shouldStop: () => false, progress: () => {} };
const small = { length: 6, hidden: 5 };

function randomCellParams(blocks: number, nIn: number, H: number, seed: number) {
  const b = new ParamBuilder(new Rng(seed));
  b.add('cell.weight_ih', [blocks * H, nIn], { kind: 'uniform', bound: 0.8 });
  b.add('cell.weight_hh', [blocks * H, H], { kind: 'uniform', bound: 0.8 });
  b.add('cell.bias_ih', [blocks * H], { kind: 'uniform', bound: 0.5 });
  b.add('cell.bias_hh', [blocks * H], { kind: 'uniform', bound: 0.5 });
  return b.store;
}

function randomVar(shape: number[], rng: Rng): Var {
  const t = Tensor.zeros(shape);
  for (let i = 0; i < t.size; i++) t.data[i] = rng.range(-1, 1);
  return leaf(t);
}

function maxDiff(a: Var, b: Var): number {
  let m = 0;
  for (let i = 0; i < a.value.size; i++)
    m = Math.max(m, Math.abs(a.value.data[i] - b.value.data[i]));
  return m;
}

describe('recurrent family', () => {
  it.each([
    [rnn, { ...small, task: 'recall' }],
    [rnn, { ...small, task: 'sine', activation: 'relu' }],
    [lstm, { ...small, task: 'recall' }],
    [lstm, { ...small, task: 'sine' }],
    [gru, { ...small, task: 'recall' }],
    [gru, { ...small, task: 'sine' }],
  ] as const)('%#: backpropagation through time matches finite differences', (m, cfg) => {
    const s = new Session(m, { config: cfg });
    s.batchSize = 3;
    const res = s.gradCheck(undefined, 3);
    expect(res.maxRelError).toBeLessThan(1e-5);
  });

  it('scoped cells reproduce the engine’s rnnCell, lstmCell and gruCell exactly', () => {
    withDType('float64', () => {
      const H = 4;
      const nIn = 3;
      const rng = new Rng(11);
      for (const [def, blocks] of [
        [rnnCellDef, 1],
        [lstmCellDef, 4],
        [gruCellDef, 3],
      ] as const) {
        const params = randomCellParams(blocks, nIn, H, blocks).clone('float64');
        const g = new Graph({ params, grad: false });
        g.run(() => {
          const x = randomVar([2, nIn], rng);
          const h = randomVar([2, H], rng);
          const c = randomVar([2, H], rng);
          const [wih, whh, bih, bhh] = ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'].map((n) =>
            p(`cell.${n}`),
          );
          const cfg = {
            task: 'recall',
            length: 3,
            hidden: H,
            activation: 'tanh',
            forgetBias: 0,
            updateBias: 0,
          } as never;
          const xp = linear(x, wih, bih);
          const prev = { h, c };
          const gates = def.gates(xp, prev, cfg);
          const next = def.update(gates, prev, cfg);
          if (def === rnnCellDef) {
            expect(maxDiff(next.h, rnnCell(x, h, wih, whh, bih, bhh))).toBeLessThan(1e-14);
          } else if (def === lstmCellDef) {
            const ref = lstmCell(x, h, c, wih, whh, bih, bhh);
            expect(maxDiff(next.h, ref.h)).toBeLessThan(1e-14);
            expect(maxDiff(next.c, ref.c)).toBeLessThan(1e-14);
            expect(maxDiff(gates.f, ref.f)).toBeLessThan(1e-14);
          } else {
            const ref = gruCell(x, h, wih, whh, bih, bhh);
            expect(maxDiff(next.h, ref.h)).toBeLessThan(1e-14);
            expect(maxDiff(gates.z, ref.z)).toBeLessThan(1e-14);
          }
        });
      }
    });
  });

  it('GRU with z forced to 1 copies its state', () => {
    const s = new Session(gru, { seed: 2, config: { length: 8, hidden: 6 } });
    const H = 6;
    const t = 4;
    const zKey = `gates/t${t}/sigmoid#1`;
    s.setOverride(
      zKey,
      Array.from({ length: H }, (_, index) => ({ index, value: 1 })),
    );
    const trace = s.inspect();
    const zNode = trace.nodes.find((n) => n.key === zKey)!;
    expect(Array.from(zNode.value)).toEqual(Array(H).fill(1));
    const h = trace.nodes.find((n) => n.id === trace.outputs.h)!.value;
    for (let j = 0; j < H; j++) expect(h[t * H + j]).toBe(h[(t - 1) * H + j]);
    // Without the override the state changes at that step.
    s.setOverride(zKey, []);
    const h2 = s.inspect().nodes.find((n) => n.id === trace.outputs.h)!.value;
    expect(
      Array.from({ length: H }, (_, j) => Math.abs(h2[t * H + j] - h2[(t - 1) * H + j])).some(
        (d) => d > 1e-4,
      ),
    ).toBe(true);
  });

  it('LSTM with f = 1 and i = 0 keeps its cell state', () => {
    const s = new Session(lstm, { seed: 3, config: { length: 8, hidden: 4 } });
    const t = 5;
    const H = 4;
    // Gate order inside gates/t: sigmoid (i), sigmoid#1 (f).
    s.setOverride(
      `gates/t${t}/sigmoid`,
      Array.from({ length: H }, (_, index) => ({ index, value: 0 })),
    );
    s.setOverride(
      `gates/t${t}/sigmoid#1`,
      Array.from({ length: H }, (_, index) => ({ index, value: 1 })),
    );
    const trace = s.inspect();
    const c = trace.nodes.find((n) => n.id === trace.outputs.c)!.value;
    for (let j = 0; j < H; j++) expect(c[t * H + j]).toBeCloseTo(c[(t - 1) * H + j], 6);
  });

  it('the diagram has gates → state with a recurrence edge back', () => {
    for (const m of [rnn, lstm, gru]) {
      const s = new Session(m, { config: { length: 5, hidden: 4 } });
      const g = buildModules(s.inspect());
      expect(g.byName.get('state')!.inputs).toContain('gates');
      expect(g.byName.get('gates')!.feedback).toContain('state');
      expect(g.byName.get('readout')!.inputs).toContain('state');
    }
  });

  it('gradient-through-time procedure reports real ∂L/∂h_t', () => {
    const s = new Session(rnn, { seed: 4, config: { length: 30 } });
    const r = s.runProcedure('gradients', { batch: 16 }, noop);
    const n = r.tensors.hNorm.data;
    expect(n.length).toBe(30);
    // Raw norms (the view draws them on a log axis).
    expect(Object.keys(r.tensors).sort()).toEqual(['hNorm']);
    expect(Array.from(n).every((v) => v > 0 && Number.isFinite(v))).toBe(true);
    const view = rnn.views.find(
      (v) => v.kind === 'series' && v.sources[0].ref === 'proc:gradients:hNorm',
    );
    expect(view && view.kind === 'series' && view.logY).toBe(true);
    // The LSTM also reports the cell-state path.
    const lr = new Session(lstm, { seed: 4, config: { length: 30 } }).runProcedure(
      'gradients',
      { batch: 8 },
      noop,
    );
    expect(Object.keys(lr.tensors).sort()).toEqual(['cNorm', 'hNorm']);
    // At initialisation the tanh RNN loses orders of magnitude over 29 steps.
    expect(n[29] / n[0]).toBeGreaterThan(1e3);
    // GRU with the update-gate offset keeps far more of the signal.
    const sg = new Session(gru, { seed: 4, config: { length: 30 } });
    const ng = sg.runProcedure('gradients', { batch: 16 }, noop).tensors.hNorm.data;
    expect(ng[0] / ng[29]).toBeGreaterThan((100 * n[0]) / n[29]);

    // At the last step ∂L/∂h_{T−1} = W_outᵀ (p − onehot(y)) / B, checked by hand.
    const batch = s.dataset.train(
      new Rng(s.seed).fork('proc:gradients:{"batch":16}').fork('batch'),
      16,
      s.config,
    );
    const W = s.params.get('readout.weight');
    const trace = s.inspect({ kind: 'loss' }, batch);
    const probs = trace.nodes.find((x) => x.id === trace.outputs.probs)!.value;
    let expected = 0;
    for (let b = 0; b < 16; b++) {
      let sq = 0;
      for (let j = 0; j < 32; j++) {
        let gj = 0;
        for (let k = 0; k < 4; k++)
          gj += W.data[k * 32 + j] * (probs[b * 4 + k] - (k === batch.labels![b] ? 1 : 0));
        sq += gj * gj;
      }
      expected += Math.sqrt(sq) / 16;
    }
    expect(n[29]).toBeCloseTo(expected, 4);
  });

  it('training lowers the validation loss on both tasks', () => {
    const cases = [
      [lstm, { task: 'recall', length: 8, hidden: 16 }, 0.3],
      [gru, { task: 'recall', length: 8, hidden: 16 }, 0.3],
      [rnn, { task: 'sine', length: 12, hidden: 16 }, 0.3],
    ] as const;
    for (const [m, cfg, ratio] of cases) {
      const s = new Session(m, { seed: 1, config: cfg });
      const before = s.evaluate()!.loss;
      for (let i = 0; i < 80; i++) s.trainStep();
      expect(s.evaluate()!.loss).toBeLessThan(before * ratio);
    }
  });
});
