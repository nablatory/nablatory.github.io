import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Graph, ParamStore, Rng, Tensor, gradCheck, mul, p, sum } from '../src/engine';
import type { Config, ModelDef } from '../src/models/types';
import { Session, effectiveCount } from '../src/runtime/session';
import vgg, { vggReceptiveFields } from '../src/models/convolutional/vgg';
import resnet, { resnetNetwork, type ResNetConfig } from '../src/models/convolutional/resnet';
import unet, { unetGeometry, unetNetwork, type UNetConfig } from '../src/models/convolutional/unet';
import { models as convolutional } from '../src/models/convolutional';
import { subsample2d } from '../src/models/convolutional/ops';
import { receptiveFields } from '../src/models/convolutional/common';
import { renderScene } from '../src/models/data/shapes';
import { input } from '../src/models/util';
import { PUBLIC, fsLoader } from './helpers';

const noop = { shouldStop: () => false, progress: () => {} };

async function session(model: ModelDef, config: Partial<Config> = {}, seed = 1): Promise<Session> {
  const s = new Session(model, { config, seed });
  await s.loadData(fsLoader);
  return s;
}

/**
 * Biases and BN shifts are initialised to exactly 0, and MNIST's background
 * is exactly 0, so many pre-activations sit exactly on the ReLU kink, where
 * central differences return ½. Move them off it before gradient checks.
 */
function jitterBiases(s: Session): Session {
  const rng = new Rng(17);
  for (const n of s.params.names()) {
    if (!n.endsWith('.bias') || !s.params.isTrainable(n)) continue;
    const t = s.params.get(n);
    for (let i = 0; i < t.size; i++) t.data[i] += rng.range(-0.2, 0.2);
  }
  return s;
}

function outputShape(s: Session, name: string): number[] {
  const tr = s.inspect({ kind: 'none' });
  return tr.nodes.find((n) => n.id === tr.outputs[name])!.shape;
}

describe('convolutional family registry', () => {
  it('lists LeNet-5 first, then VGG, ResNet and U-Net', () => {
    expect(convolutional.map((m) => m.id)).toEqual(['lenet-5', 'vgg', 'resnet', 'unet']);
    for (const m of convolutional.slice(1)) {
      expect(m.fidelity).toBe('reduced');
      expect(m.scaleNote).toBeTruthy();
      expect(m.validate!(m.defaults)).toBeNull();
      for (const r of m.references) expect(r.url).toMatch(/^https:\/\//);
    }
  });
});

describe('subsample2d', () => {
  it('picks every s-th pixel and its VJP matches finite differences', () => {
    const x = new Tensor(
      [2, 3, 5, 5],
      Array.from({ length: 150 }, (_, i) => i),
    );
    const g = new Graph({ grad: false });
    const y = g.run(() => subsample2d(input('x', x, false), 2));
    expect(y.shape).toEqual([2, 3, 3, 3]);
    expect(y.value.data[0]).toBe(0);
    expect(y.value.data[1]).toBe(2);
    expect(y.value.data[3]).toBe(10);
    const rng = new Rng(4);
    const store = new ParamStore();
    store.add(
      'x',
      new Tensor(
        [2, 3, 5, 5],
        Array.from({ length: 150 }, () => rng.normal()),
      ),
      'test',
    );
    const w = new Tensor(
      [2, 3, 3, 3],
      Array.from({ length: 54 }, () => rng.normal()),
    );
    const res = gradCheck(() => sum(mul(subsample2d(p('x'), 2), input('w', w, false))), store, {
      perParam: 40,
    });
    expect(res.maxRelError).toBeLessThan(1e-7);
  });
});

describe('VGG', () => {
  const tiny = { stages: 2, convsPerStage: 2, baseChannels: 2, fcWidth: 16, dropout: 0.5 };

  it('backpropagation matches finite differences (float64, evaluation and training mode)', async () => {
    for (const batchNorm of [false, true]) {
      const s = jitterBiases(await session(vgg, { ...tiny, batchNorm }));
      s.batchSize = 2;
      expect(s.gradCheck(undefined, 2).maxRelError).toBeLessThan(1e-5);
      // Training mode: batch statistics in every batch norm, the same dropout mask in L(θ±ε).
      expect(s.gradCheck(undefined, 2, undefined, true).maxRelError).toBeLessThan(1e-5);
    }
  });

  it('halves the spatial size and doubles the channels at every stage', async () => {
    const s = await session(vgg, { stages: 4, baseChannels: 4 });
    expect(outputShape(s, 'stage1')).toEqual([1, 4, 28, 28]);
    expect(outputShape(s, 'stage2')).toEqual([1, 8, 14, 14]);
    expect(outputShape(s, 'stage3')).toEqual([1, 16, 7, 7]);
    expect(outputShape(s, 'stage4')).toEqual([1, 32, 3, 3]);
    expect(s.params.get('fc1.weight').shape).toEqual([vgg.defaults.fcWidth, 32 * 1 * 1]);
    expect(vgg.validate!({ ...vgg.defaults, stages: 5 })).toMatch(/below 1×1/);
    // Batch norm removes the convolution bias and adds γ, β (+ running statistics).
    const bnS = await session(vgg, { batchNorm: true });
    expect(bnS.params.has('stage1.conv1.bias')).toBe(false);
    expect(bnS.params.get('stage1.conv1.bn.runningVar').shape).toEqual([
      vgg.defaults.baseChannels as number,
    ]);
  });

  it('receptive fields: analytic recurrence and measured gradient support', async () => {
    // Two 3×3 convs (5) then 2×2 pool (6); ×2 jump: 6 + 2·2·2 = 14, pool → 16; 16 + 2·2·4 = 32, pool → 36.
    expect(vggReceptiveFields({ ...vgg.defaults, stages: 3, convsPerStage: 2 } as never)).toEqual([
      6, 16, 36,
    ]);
    expect(
      receptiveFields([
        { kernel: 3, stride: 1 },
        { kernel: 3, stride: 1 },
        { kernel: 3, stride: 1 },
      ]).map((r) => r.size),
    ).toEqual([3, 5, 7]);
    const s = await session(vgg);
    const r = s.runProcedure('receptive-field', { example: 0 }, noop);
    expect(r.tensors.gradient.shape).toEqual([3, 1, 28, 28]);
    for (let k = 1; k <= 3; k++) {
      const measured = r.scalars![`stage ${k} non-zero ∂y/∂x extent ●`];
      expect(measured).toBeGreaterThan(0);
      expect(measured).toBeLessThanOrEqual(Math.min(28, r.scalars![`stage ${k} analytic RF ○`]));
    }
  });

  it('training reduces the validation loss', async () => {
    const s = await session(vgg, { baseChannels: 4, dropout: 0 }, 2);
    const before = s.evaluate()!.loss;
    for (let i = 0; i < 60; i++) s.trainStep();
    expect(s.evaluate()!.loss).toBeLessThan(before * 0.75);
  });
});

describe('ResNet', () => {
  const tiny = { stages: 2, blocksPerStage: 1, width: 2 };

  it('backpropagation matches finite differences for each block variant (float64, both BN modes)', async () => {
    const variants: Partial<ResNetConfig>[] = [
      { shortcut: 'A', preActivation: false },
      { shortcut: 'B', preActivation: false },
      { shortcut: 'B', preActivation: true },
      { shortcut: 'A', preActivation: true, shortcuts: false },
    ];
    for (const v of variants) {
      const s = jitterBiases(await session(resnet, { ...tiny, ...v }));
      s.batchSize = 2;
      expect(s.gradCheck(undefined, 2).maxRelError).toBeLessThan(1e-5);
      expect(s.gradCheck(undefined, 2, undefined, true).maxRelError).toBeLessThan(1e-5);
    }
  });

  it('stage shapes: 28 → 14 → 7 with widths w, 2w, 4w', async () => {
    const s = await session(resnet, { width: 4, blocksPerStage: 2 });
    expect(outputShape(s, 'block2')).toEqual([1, 4, 28, 28]);
    expect(outputShape(s, 'block3')).toEqual([1, 8, 14, 14]);
    expect(outputShape(s, 'block6')).toEqual([1, 16, 7, 7]);
    expect(outputShape(s, 'shortcut3')).toEqual([1, 8, 14, 14]);
    const b = await session(resnet, { width: 4, shortcut: 'B' });
    expect(b.params.get('shortcut2.conv.weight').shape).toEqual([8, 4, 1, 1]);
  });

  it('turning shortcuts off keeps every parameter and removes only the addition', async () => {
    for (const shortcut of ['A', 'B'] as const) {
      const on = await session(resnet, { shortcut, shortcuts: true }, 4);
      const off = await session(resnet, { shortcut, shortcuts: false }, 4);
      expect(effectiveCount(off.params)).toBe(effectiveCount(on.params));
      expect(off.params.names()).toEqual(on.params.names());
      for (const n of on.params.names())
        expect(off.params.get(n).data).toEqual(on.params.get(n).data);
      const c = on.config as ResNetConfig;
      const batch = on.inspectionBatch();
      const run = (cfg: ResNetConfig) =>
        new Graph({ params: on.params, grad: false, record: false }).run(() =>
          resnetNetwork(input('image', batch.inputs.image, false), cfg),
        );
      const tOn = run(c);
      const tOff = run({ ...c, shortcuts: false });
      // Block 1 sees the same input in both modes, so F₁ is identical; only the sum differs.
      expect(tOff.residuals[0].value.data).toEqual(tOn.residuals[0].value.data);
      for (const [t, withSkip] of [
        [tOn, true],
        [tOff, false],
      ] as const) {
        t.blockOutputs.forEach((y, k) => {
          const f = t.residuals[k].value.data;
          const sc = t.shortcuts[k]?.value.data;
          expect(!!sc).toBe(withSkip);
          for (let i = 0; i < y.value.size; i += 7)
            expect(y.value.data[i]).toBeCloseTo(Math.max(0, f[i] + (sc ? sc[i] : 0)), 5);
        });
      }
      expect(tOn.shortcuts[0]).toBe(tOn.blockInputs[0]);
    }
  });

  it('measures ‖∂L/∂x_l‖ at every block input with and without shortcuts', async () => {
    const s = await session(resnet);
    const before = s.params.get('stem.bn.runningMean').clone();
    const r = s.runProcedure('gradient-norms', { batch: 8 }, noop);
    expect(r.tensors.withShortcuts.shape).toEqual([3]);
    expect(r.tensors.plain.shape).toEqual([3]);
    for (const t of [r.tensors.withShortcuts, r.tensors.plain])
      for (const v of t.data) expect(v).toBeGreaterThan(0);
    expect(r.tensors.withShortcuts.data).not.toEqual(r.tensors.plain.data);
    // The procedure works on a copy: running statistics are untouched.
    expect(s.params.get('stem.bn.runningMean').data).toEqual(before.data);
  });

  it('training reduces the loss and updates the running statistics', async () => {
    // Early in training the evaluation-mode loss (running statistics) is noisy,
    // so compare the training loss of the first and last ten steps.
    const s = await session(resnet, { width: 4 }, 2);
    for (let i = 0; i < 100; i++) s.trainStep();
    const avg = (a: number, b: number) =>
      s.history.slice(a, b).reduce((t, r) => t + r.losses.main, 0) / (b - a);
    expect(avg(90, 100)).toBeLessThan(0.85 * avg(0, 10));
    expect(Number.isFinite(s.evaluate()!.loss)).toBe(true);
    expect(s.params.get('stem.bn.runningMean').data.some((v) => v !== 0)).toBe(true);
  });
});

describe('U-Net', () => {
  const tiny: Partial<UNetConfig> = { depth: 1, baseChannels: 2, tile: 12, diceWeight: 0.5 };

  it('backpropagation matches finite differences (float64, cross-entropy + Dice)', async () => {
    for (const cfg of [
      { ...tiny, classes: '3' as const },
      { ...tiny, classes: '2' as const, padding: 'valid' as const, tile: 20 },
    ]) {
      const s = jitterBiases(await session(unet, cfg));
      s.batchSize = 2;
      expect(s.gradCheck(undefined, 2).maxRelError).toBeLessThan(1e-5);
    }
  });

  it('output size equals input size and skips concatenate C + C channels', async () => {
    const c = { ...unet.defaults, depth: 3, baseChannels: 4, classes: '3' } as UNetConfig;
    const x = new Tensor([1, 1, 32, 32]);
    const t = new Graph({ params: unet.init(c, new Rng(1)), grad: false }).run(() =>
      unetNetwork(input('image', x, false), c),
    );
    expect(t.logits.shape).toEqual([1, 3, 32, 32]);
    expect(t.skips.map((v) => v.shape)).toEqual([
      [1, 4, 32, 32],
      [1, 8, 16, 16],
      [1, 16, 8, 8],
    ]);
    expect(t.bottleneck.shape).toEqual([1, 32, 4, 4]);
    // Deepest level first: [crop(e_l); upconv] has 2·C_l channels.
    expect(t.concats.map((v) => v.shape)).toEqual([
      [1, 32, 8, 8],
      [1, 16, 16, 16],
      [1, 8, 32, 32],
    ]);
  });

  it('unpadded variant crops the skips and matches the geometry', async () => {
    const c = { ...unet.defaults, padding: 'valid', tile: 60 } as UNetConfig;
    expect(unetGeometry(c)).toEqual({
      input: 60,
      skips: [56, 24],
      bottleneck: 8,
      upsampled: [16, 24],
      output: 20,
    });
    expect(unet.validate!({ ...c, tile: 32 })).toMatch(/unpadded/);
    expect(unet.validate!({ ...unet.defaults, depth: 3, tile: 36 } as UNetConfig)).toMatch(
      /multiple of 8/,
    );
    const s = await session(unet, { padding: 'valid', tile: 60 });
    expect(outputShape(s, 'logits')).toEqual([1, 2, 20, 20]);
    expect(outputShape(s, 'truth')).toEqual([1, 3, 20, 20]);
    // Same parameters in both modes.
    expect(effectiveCount(s.params)).toBe(effectiveCount(new Session(unet).params));
  });

  it('scenes are deterministic and labelled', () => {
    const spec = { size: 32, crop: 32, classes: 3 as const, noise: 0.1 };
    const a = renderScene(new Rng(9), spec);
    const b = renderScene(new Rng(9), spec);
    expect(a.image).toEqual(b.image);
    expect(a.labels).toEqual(b.labels);
    expect(Math.max(...a.labels)).toBeLessThanOrEqual(2);
    expect(a.labels.some((v) => v > 0)).toBe(true);
    const cropped = renderScene(new Rng(9), { ...spec, crop: 20 });
    expect(cropped.labels[0]).toBe(a.labels[6 * 32 + 6]);
  });

  it('training reduces the loss', async () => {
    const s = await session(unet, { baseChannels: 4 }, 2);
    s.batchSize = 4;
    const before = s.evaluate()!;
    for (let i = 0; i < 60; i++) s.trainStep();
    const after = s.evaluate()!;
    expect(after.loss).toBeLessThan(before.loss * 0.7);
    expect(after.metrics.pixelAccuracy).toBeGreaterThan(before.metrics.pixelAccuracy);
  });
});

describe('checkpoints', () => {
  for (const model of [vgg, resnet, unet]) {
    for (const ck of model.checkpoints ?? []) {
      it(`${model.id}/${ck.id} loads and reproduces its recorded validation metric`, async () => {
        const json = join(PUBLIC, `${ck.path}.json`);
        expect(existsSync(json)).toBe(true);
        const file = JSON.parse(readFileSync(json, 'utf8'));
        const buf = readFileSync(join(PUBLIC, `${ck.path}.bin`));
        expect(buf.byteLength).toBeLessThan(2 * 1024 * 1024);
        const s = await session(model, file.config);
        s.loadCheckpoint(
          ck.id,
          file.params,
          new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)),
        );
        const e = s.evaluate()!;
        const rec = file.provenance.metrics;
        expect(e.loss).toBeCloseTo(rec.validationLoss, 4);
        // validationAccuracy → accuracy, validationIou → iou, …
        const keys = Object.keys(rec).filter(
          (k) => k.startsWith('validation') && k !== 'validationLoss',
        );
        expect(keys.length).toBeGreaterThan(0);
        for (const k of keys)
          expect(e.metrics[k[10].toLowerCase() + k.slice(11)]).toBeCloseTo(rec[k], 4);
      });
    }
  }
});
