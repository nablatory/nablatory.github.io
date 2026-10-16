import { describe, expect, it, beforeAll } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import gan from '../src/models/gan/gan';
import dcgan, { dcganEvaluate } from '../src/models/gan/dcgan';
import { isD, isG } from '../src/models/gan/common';
import { Graph, Rng, Tensor, backward, gradCheck, withDType, type ParamStore } from '../src/engine';
import {
  TOY2D_EXTENT,
  modeCoverage,
  toy2dGrid,
  toy2dModes,
  toy2dSample,
} from '../src/models/data/toy2d';
import { digitBatch, loadMnist, mnist } from '../src/models/data/mnist';
import { Session } from '../src/runtime/session';
import type { CheckpointFile } from '../src/runtime/checkpoint';
import type { Batch, ModelDef, TrainPhase } from '../src/models/types';
import { PUBLIC, fsLoader } from './helpers';

const noop = { shouldStop: () => false, progress: () => {} };

beforeAll(async () => {
  await loadMnist(fsLoader);
});

function snapshot(params: ParamStore, filter: (n: string) => boolean): Map<string, Float32Array> {
  const m = new Map<string, Float32Array>();
  for (const n of params.names())
    if (params.isTrainable(n) && filter(n)) m.set(n, Float32Array.from(params.get(n).data));
  return m;
}

function changed(params: ParamStore, before: Map<string, Float32Array>): string[] {
  return [...before]
    .filter(([n, v]) => params.get(n).data.some((x, i) => x !== v[i]))
    .map(([n]) => n);
}

/** The model with only one of its training phases. */
function onlyPhase(model: ModelDef, k: number): ModelDef {
  return {
    ...model,
    training: { ...model.training, validateEvery: 0, phases: [model.training.phases![k]] },
  };
}

/** Float64 central differences of a phase loss in training mode (batch statistics in batch norm). */
function trainingModeGradCheck(
  s: Session,
  phase: TrainPhase,
  batch: Batch,
  names: string[],
  perParam = 2,
) {
  return withDType('float64', () => {
    const store = s.params.clone('float64');
    const make = (grad: boolean) =>
      new Graph({ params: store, record: false, grad, training: true, rng: new Rng(1) });
    const f = () => phase.loss(batch, s.config, { training: true, rng: new Rng(2) }).loss;
    const g = make(true);
    backward(g.run(f));
    const rng = new Rng(3);
    let worst = 0;
    for (const name of names) {
      const t = store.get(name);
      for (let k = 0; k < perParam; k++) {
        const i = rng.int(t.size);
        const a = g.paramVar(name)!.grad!.data[i];
        const o = t.data[i];
        const eps = 1e-6;
        t.data[i] = o + eps;
        const lp = make(false).run(f).value.data[0];
        t.data[i] = o - eps;
        const lm = make(false).run(f).value.data[0];
        t.data[i] = o;
        const num = (lp - lm) / (2 * eps);
        worst = Math.max(worst, Math.abs(a - num) / Math.max(Math.abs(a), Math.abs(num), 1e-4));
      }
    }
    return worst;
  });
}

describe('two-dimensional targets', () => {
  it('are deterministic given the generator and centred on the stated modes', () => {
    expect(toy2dSample('ring', 50, new Rng(3)).data).toEqual(
      toy2dSample('ring', 50, new Rng(3)).data,
    );
    const pts = toy2dSample('ring', 4000, new Rng(1));
    const cov = modeCoverage('ring', pts)!;
    expect(cov.covered).toBe(8);
    expect(cov.quality).toBeGreaterThan(0.98);
    expect(toy2dModes('grid')!.centres).toHaveLength(25);
    const g = toy2dGrid(4);
    expect(g.data[0]).toBeCloseTo(TOY2D_EXTENT[0] + 0.4, 5);
    expect(g.data[1]).toBeCloseTo(TOY2D_EXTENT[3] - 0.4, 5);
  });
});

describe('GAN (two-dimensional)', () => {
  it.each([{ width: 8 }, { width: 8, zDim: 3, gLoss: 'minimax' }])(
    'both phases match finite differences (%o)',
    (config) => {
      const s = new Session(gan, { config });
      s.batchSize = 4;
      // Session.gradCheck checks each phase on its own parameters.
      expect(s.gradCheck(undefined, 3).maxRelError).toBeLessThan(1e-5);
      const batch = s.dataset.train(new Rng(4), 4, s.config);
      const gPhase = gan.training.phases![1];
      const res = gradCheck(
        () => gPhase.loss(batch, s.config, { training: false, rng: new Rng(3) }).loss,
        s.params,
        {
          names: s.params.names().filter(isG),
          perParam: 3,
        },
      );
      expect(res.maxRelError).toBeLessThan(1e-5);
    },
  );

  it('every trainable parameter belongs to exactly one phase', () => {
    for (const m of [gan, dcgan]) {
      const s = new Session(m);
      for (const n of s.params.names().filter((n) => s.params.isTrainable(n))) {
        expect(m.training.phases!.filter((ph) => ph.params(n))).toHaveLength(1);
      }
    }
  });

  it('the discriminator phase leaves G unchanged and the generator phase leaves D unchanged', () => {
    for (const k of [0, 1]) {
      const s = new Session(onlyPhase(gan, k), { seed: 2 });
      const g0 = snapshot(s.params, isG);
      const d0 = snapshot(s.params, isD);
      for (let i = 0; i < 3; i++) s.trainStep();
      if (k === 0) {
        expect(changed(s.params, g0)).toEqual([]);
        expect(changed(s.params, d0).length).toBe(d0.size);
      } else {
        expect(changed(s.params, d0)).toEqual([]);
        expect(changed(s.params, g0).length).toBe(g0.size);
      }
    }
  });

  it('when D rejects generated points, the minimax loss gives G much smaller gradients', () => {
    const s = new Session(onlyPhase(gan, 0), { seed: 3 });
    for (let i = 0; i < 150; i++) s.trainStep();
    const batch = s.dataset.train(new Rng(8), 128, s.config);
    const norms: Record<string, number> = {};
    let dFake = 0;
    for (const gLoss of ['nonsaturating', 'minimax']) {
      const g = new Graph({ params: s.params, record: false, grad: true, rng: new Rng(1) });
      const c = { ...s.config, gLoss };
      const r = g.run(() => gan.forward(batch, c, { training: false, rng: new Rng(1) }));
      backward(r.outputs.gLoss);
      dFake = r.metrics!['D(G(z))'];
      norms[gLoss] = Math.hypot(
        ...s.params
          .names()
          .filter(isG)
          .flatMap((n) => Array.from(g.paramVar(n)!.grad!.data)),
      );
    }
    expect(dFake).toBeLessThan(0.1);
    expect(norms.nonsaturating).toBeGreaterThan(3 * norms.minimax);
  });

  it('training moves generated points towards the ring', () => {
    const s = new Session({ ...gan, training: { ...gan.training, validateEvery: 0 } }, { seed: 1 });
    const centres = toy2dModes('ring')!.centres;
    // Mean distance from the 1,000 generated points (labels = 1) to the nearest mode centre.
    const spread = () => {
      const r = s.runProcedure('samples', {}, noop);
      let sum = 0;
      let n = 0;
      for (let i = 0; i < r.tensors.labels.size; i++) {
        if (r.tensors.labels.data[i] !== 1) continue;
        const x = r.tensors.points.data[2 * i];
        const y = r.tensors.points.data[2 * i + 1];
        sum += Math.min(...centres.map(([cx, cy]) => Math.hypot(x - cx, y - cy)));
        n++;
      }
      return sum / n;
    };
    const before = spread();
    for (let i = 0; i < 500; i++) s.trainStep();
    expect(spread()).toBeLessThan(0.5 * before);
  });

  it('the decision field equals D evaluated by the forward pass at the grid points', () => {
    const s = new Session(gan, { seed: 5 });
    for (let i = 0; i < 20; i++) s.trainStep();
    const ex = s.dataset.example(0, s.config);
    const trace = s.inspect({ kind: 'none' });
    const field = trace.nodes.find((n) => n.id === trace.outputs.field)!;
    expect(field.shape).toEqual([64, 64]);
    const grid = toy2dGrid(64);
    const pick = [0, 100, 2080, 4095];
    const x = new Tensor(
      [pick.length, 2],
      pick.flatMap((i) => [grid.data[2 * i], grid.data[2 * i + 1]]),
    );
    const g = new Graph({ params: s.params, record: false, grad: false, rng: new Rng(0) });
    const r = g.run(() =>
      gan.forward({ inputs: { x } }, s.config, { training: false, rng: new Rng(0) }),
    );
    pick.forEach((i, k) =>
      expect(field.value[i]).toBeCloseTo(1 / (1 + Math.exp(-r.outputs.dReal.value.data[k])), 5),
    );
    expect(ex.meta?.field).toBe(true);
    const proc = s.runProcedure('samples', {}, noop);
    expect(proc.tensors.points.shape).toEqual([2000, 2]);
    expect(proc.tensors.field.shape).toEqual([64, 64]);
  });
});

describe('DCGAN', () => {
  it('generator and discriminator have the documented shapes', () => {
    const s = new Session(dcgan);
    const g = new Graph({ params: s.params, record: false, grad: false, rng: new Rng(1) });
    const r = g.run(() =>
      dcgan.forward(s.dataset.example(0, s.config), s.config, { training: false, rng: new Rng(1) }),
    );
    expect(r.outputs.fake.shape).toEqual([16, 1, 28, 28]);
    expect(Math.max(...r.outputs.fake.value.data.map(Math.abs))).toBeLessThan(1);
    expect(r.outputs.dReal.shape).toEqual([1, 1]);
    expect(s.params.get('G.deconv1.weight').shape).toEqual([32, 16, 4, 4]);
    expect(s.params.get('D.out.weight').shape).toEqual([1, 32 * 49]);
    expect(s.params.isTrainable('G.project.bn.runningMean')).toBe(false);
  });

  it('both phases match finite differences, with batch norm in evaluation and training mode', () => {
    // At initialisation (weights N(0, 0.02), zero biases, running statistics 0 / 1)
    // generated images are ~1e-4, so D's pre-activations on them lie within
    // ε = 1e-6 of the leaky-ReLU kink and a bias perturbation crosses it. A few
    // training steps move the parameters to a generic point first.
    const s = new Session(dcgan, { seed: 3, config: { zDim: 4, gf: 4, df: 4 } });
    s.batchSize = 8;
    for (let i = 0; i < 10; i++) s.trainStep();
    s.batchSize = 3;
    expect(s.gradCheck(undefined, 2).maxRelError).toBeLessThan(1e-5);
    const batch = s.dataset.train(new Rng(4), 3, s.config);
    const [dPhase, gPhase] = dcgan.training.phases!;
    const gNames = s.params.names().filter((n) => isG(n) && s.params.isTrainable(n));
    const dNames = s.params.names().filter((n) => isD(n) && s.params.isTrainable(n));
    const evalG = gradCheck(
      () => gPhase.loss(batch, s.config, { training: false, rng: new Rng(3) }).loss,
      s.params,
      { names: gNames, perParam: 2 },
    );
    expect(evalG.maxRelError).toBeLessThan(1e-5);
    expect(trainingModeGradCheck(s, dPhase, batch, dNames)).toBeLessThan(1e-5);
    expect(trainingModeGradCheck(s, gPhase, batch, gNames)).toBeLessThan(1e-5);
  });

  it('phases update only their own parameters; batch-norm running statistics move in training', () => {
    for (const k of [0, 1]) {
      const s = new Session(onlyPhase(dcgan, k), { config: { gf: 8, df: 8 } });
      s.batchSize = 8;
      const g0 = snapshot(s.params, isG);
      const d0 = snapshot(s.params, isD);
      const rm = Float32Array.from(s.params.get('G.project.bn.runningMean').data);
      s.trainStep();
      expect(changed(s.params, k === 0 ? g0 : d0)).toEqual([]);
      expect(changed(s.params, k === 0 ? d0 : g0).length).toBe((k === 0 ? d0 : g0).size);
      expect(s.params.get('G.project.bn.runningMean').data).not.toEqual(rm);
    }
  });

  it('fixed-noise samples are reproducible and interpolation rows start and end at G(z₀), G(z₁)', () => {
    const s = new Session(dcgan, { config: { gf: 8, df: 8 } });
    const a = s.runProcedure('samples', { noiseSeed: 7 }, noop).tensors.samples;
    expect(a.shape).toEqual([64, 1, 28, 28]);
    expect(s.runProcedure('samples', { noiseSeed: 7 }, noop).tensors.samples.data).toEqual(a.data);
    expect(s.runProcedure('samples', { noiseSeed: 8 }, noop).tensors.samples.data).not.toEqual(
      a.data,
    );
    const grid = s.runProcedure('interpolate', { rows: 2 }, noop).tensors.grid;
    expect(grid.shape).toEqual([20, 1, 28, 28]);
    const row0 = Array.from(grid.data.subarray(0, 784));
    expect(row0).not.toEqual(Array.from(grid.data.subarray(9 * 784, 10 * 784)));
  });

  it('training raises D(x) above D(G(z)) at first', () => {
    const s = new Session(dcgan, { seed: 2, config: { gf: 8, df: 8 } });
    s.batchSize = 16;
    let last: Record<string, number> = {};
    for (let i = 0; i < 25; i++)
      last = (s.trainStep() as { metrics: Record<string, number> }).metrics;
    expect(last['D(x)']).toBeGreaterThan(last['D(G(z))']);
  });

  const ckpt = join(PUBLIC, 'weights', 'dcgan-mnist.json');
  it.runIf(existsSync(ckpt))(
    'the shipped checkpoint loads and reproduces its recorded measurements',
    () => {
      const file = JSON.parse(readFileSync(ckpt, 'utf8')) as CheckpointFile;
      const bin = readFileSync(join(PUBLIC, 'weights', 'dcgan-mnist.bin'));
      expect(bin.byteLength).toBeLessThan(2 * 1024 * 1024);
      const s = new Session(dcgan);
      s.setConfig(file.config);
      s.loadCheckpoint(
        'mnist',
        file.params,
        new Float32Array(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)),
      );
      const d = mnist();
      const real = digitBatch(
        d.testImages,
        d.testLabels,
        Array.from({ length: 1000 }, (_, i) => i),
        'symmetric',
      );
      const ev = dcganEvaluate(s.params, s.config, real, 0, 1000);
      const m = file.provenance.metrics;
      expect(ev.metrics.dLoss).toBeCloseTo(m.testDLoss, 3);
      expect(ev.metrics.gLoss).toBeCloseTo(m.testGLoss, 3);
      expect(ev.metrics.meanDReal).toBeCloseTo(m.testMeanDReal, 3);
      expect(ev.metrics.meanDFake).toBeCloseTo(m.testMeanDFake, 3);
    },
  );
});
