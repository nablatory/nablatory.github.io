import { describe, expect, it, beforeAll } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import autoencoder from '../src/models/autoencoder/autoencoder';
import denoising, { corrupt } from '../src/models/autoencoder/denoising';
import vae, { probit, vaeNegElbo } from '../src/models/autoencoder/vae';
import { pca2 } from '../src/models/autoencoder/common';
import { bernoulliNll, squaredError } from '../src/models/autoencoder/ops';
import {
  Graph,
  ParamStore,
  Rng,
  Tensor,
  bceWithLogits,
  gradCheck,
  mse,
  p,
  withDType,
} from '../src/engine';
import { digitBatch, loadMnist, mnist } from '../src/models/data/mnist';
import { Session } from '../src/runtime/session';
import type { CheckpointFile } from '../src/runtime/checkpoint';
import type { Batch, ModelDef } from '../src/models/types';
import { input } from '../src/models/util';
import { PUBLIC, fsLoader } from './helpers';

const noop = { shouldStop: () => false, progress: () => {} };

beforeAll(async () => {
  await loadMnist(fsLoader);
});

/** Forward pass outside the session, returning output values and metrics. */
function run(model: ModelDef, s: Session, batch: Batch, seed = 1) {
  const g = new Graph({ params: s.params, record: false, grad: false, rng: new Rng(seed) });
  const r = g.run(() => model.forward(batch, s.config, { training: false, rng: new Rng(seed) }));
  const out: Record<string, Tensor> = {};
  for (const [k, v] of Object.entries(r.outputs)) out[k] = v.value;
  return { out, loss: r.loss!.value.data[0], metrics: r.metrics! };
}

function replicate(image: Tensor, n: number): Batch {
  const x = new Float32Array(n * 784);
  for (let i = 0; i < n; i++) x.set(image.data.subarray(0, 784), i * 784);
  return { inputs: { image: new Tensor([n, 1, 28, 28], x) } };
}

describe('reconstruction ops', () => {
  it('match the engine losses and their finite-difference gradients', () => {
    const rng = new Rng(4);
    const n = 300;
    const z0 = Tensor.zeros([3, n / 3]);
    const t = Tensor.zeros([3, n / 3]);
    for (let i = 0; i < n; i++) {
      z0.data[i] = 3 * rng.normal();
      t.data[i] = rng.uniform();
    }
    withDType('float64', () => {
      const z = z0.cast();
      const g = new Graph({ grad: false });
      g.run(() => {
        const leaf = input('z', z, false);
        expect(bernoulliNll(leaf, t, 1 / n).value.data[0]).toBeCloseTo(
          bceWithLogits(leaf, t).value.data[0],
          12,
        );
        expect(squaredError(leaf, t, 1 / n).value.data[0]).toBeCloseTo(
          mse(leaf, t).value.data[0],
          12,
        );
      });
    });
    const store = new ParamStore();
    store.add('z', z0, 'test');
    for (const f of [() => bernoulliNll(p('z'), t, 0.37), () => squaredError(p('z'), t, 0.37)]) {
      const res = gradCheck(f, store, { perParam: 20 });
      expect(res.maxRelError).toBeLessThan(1e-6);
    }
  });
});

describe('autoencoder', () => {
  it.each([
    { hidden: 16, latent: 2 },
    { hidden: 16, latent: 3, loss: 'mse', activation: 'tanh' },
    { arch: 'conv', latent: 3 },
  ])('backpropagation matches finite differences (%o)', (config) => {
    const s = new Session(autoencoder, { config });
    s.batchSize = 2;
    const res = s.gradCheck(undefined, 3);
    expect(res.maxRelError).toBeLessThan(1e-5);
  });

  it('decoder output has the input shape and lies in (0, 1); conv variant halves to 7 × 7', () => {
    const s = new Session(autoencoder, { config: { arch: 'conv', latent: 5 } });
    const r = run(autoencoder, s, s.dataset.example(0, s.config));
    expect(r.out.reconstruction.shape).toEqual([1, 1, 28, 28]);
    expect(r.out.z.shape).toEqual([1, 5]);
    expect(Math.min(...r.out.reconstruction.data)).toBeGreaterThan(0);
    expect(Math.max(...r.out.reconstruction.data)).toBeLessThan(1);
    expect(s.params.get('dec2.weight').shape).toEqual([32, 16, 4, 4]);
  });

  it('training reduces the validation reconstruction loss', () => {
    const s = new Session(autoencoder, { seed: 2, config: { hidden: 64, latent: 2 } });
    s.batchSize = 32;
    const before = s.evaluate()!;
    for (let i = 0; i < 80; i++) s.trainStep();
    const after = s.evaluate()!;
    expect(after.loss).toBeLessThan(before.loss * 0.5);
    expect(after.metrics.pixelMse).toBeLessThan(before.metrics.pixelMse);
  });

  it('latent procedure encodes the validation set; d > 2 is projected by PCA', () => {
    const s = new Session(autoencoder, { config: { hidden: 32, latent: 2 } });
    const r = s.runProcedure('latent', {}, noop);
    expect(r.tensors.z.shape).toEqual([500, 2]);
    expect(Array.from(r.tensors.labels.data.subarray(0, 10))).toEqual(
      Array.from(mnist().testLabels.subarray(0, 10)),
    );
    // The codes themselves when d = 2.
    const b = digitBatch(mnist().testImages, mnist().testLabels, [0], 'unit');
    const z0 = run(autoencoder, s, b).out.z;
    expect(r.tensors.z.data[0]).toBeCloseTo(z0.data[0], 5);
    const s8 = new Session(autoencoder, { config: { hidden: 32, latent: 8 } });
    const r8 = s8.runProcedure('latent', {}, noop);
    expect(r8.tensors.z.shape).toEqual([500, 2]);
    expect(r8.scalars!.varianceShown).toBeGreaterThan(0);
    expect(r8.scalars!.varianceShown).toBeLessThanOrEqual(1 + 1e-9);
  });

  it('PCA recovers the dominant axes of an anisotropic cloud', () => {
    const rng = new Rng(9);
    const n = 2000;
    const z = Tensor.zeros([n, 3]);
    for (let i = 0; i < n; i++) {
      const a = 5 * rng.normal();
      const b = 2 * rng.normal();
      const c = 0.1 * rng.normal();
      // axes: (1,1,0)/√2 with σ = 5, (0,0,1) with σ = 2, (1,−1,0)/√2 with σ = 0.1
      z.data[i * 3] = (a + c) / Math.SQRT2;
      z.data[i * 3 + 1] = (a - c) / Math.SQRT2;
      z.data[i * 3 + 2] = b;
    }
    const { points, explained } = pca2(z);
    let v0 = 0;
    let v1 = 0;
    for (let i = 0; i < n; i++) {
      v0 += points.data[i * 2] ** 2 / n;
      v1 += points.data[i * 2 + 1] ** 2 / n;
    }
    expect(Math.sqrt(v0)).toBeCloseTo(5, 0);
    expect(Math.sqrt(v1)).toBeCloseTo(2, 0);
    expect(explained).toBeGreaterThan(0.99);
  });
});

describe('denoising autoencoder', () => {
  it.each([
    { hidden: 16 },
    { hidden: 16, tied: false, noise: 'gaussian', activation: 'relu' },
    { hidden: 16, noise: 'saltpepper', loss: 'mse' },
  ])(
    'backpropagation matches finite differences with the corruption fixed by the seed (%o)',
    (config) => {
      const s = new Session(denoising, { config });
      s.batchSize = 2;
      expect(s.gradCheck(undefined, 3).maxRelError).toBeLessThan(1e-5);
    },
  );

  it('masking zeroes exactly ⌊ν·784⌋ pixels; Gaussian noise has variance σ²', () => {
    const ones = new Tensor([3, 1, 28, 28], new Float32Array(3 * 784).fill(1));
    const g = new Graph({ grad: false, rng: new Rng(5) });
    const masked = g.run(() =>
      corrupt(input('x', ones, false), {
        ...denoising.defaults,
        noise: 'masking',
        fraction: 0.3,
      } as never),
    ).value;
    for (let n = 0; n < 3; n++) {
      let zeros = 0;
      for (let i = 0; i < 784; i++) if (masked.data[n * 784 + i] === 0) zeros++;
      expect(zeros).toBe(Math.floor(0.3 * 784));
    }
    const noisy = g.run(() =>
      corrupt(input('x', ones, false), {
        ...denoising.defaults,
        noise: 'gaussian',
        sigma: 0.4,
      } as never),
    ).value;
    let m = 0;
    let v = 0;
    for (let i = 0; i < noisy.size; i++) m += (noisy.data[i] - 1) / noisy.size;
    for (let i = 0; i < noisy.size; i++) v += (noisy.data[i] - 1 - m) ** 2 / noisy.size;
    expect(Math.abs(m)).toBeLessThan(0.03);
    expect(v).toBeCloseTo(0.16, 1);
  });

  it('the corruption is redrawn from the graph generator: same seed, same x̃', () => {
    const s = new Session(denoising);
    const b = s.dataset.example(0, s.config);
    expect(run(denoising, s, b, 3).out.corrupted.data).toEqual(
      run(denoising, s, b, 3).out.corrupted.data,
    );
    expect(run(denoising, s, b, 3).out.corrupted.data).not.toEqual(
      run(denoising, s, b, 4).out.corrupted.data,
    );
  });

  it('after training, reconstructions are closer to the clean digit than the corrupted input is', () => {
    const s = new Session(denoising, {
      seed: 3,
      config: { hidden: 64, noise: 'gaussian', sigma: 0.5 },
    });
    s.batchSize = 32;
    const before = s.evaluate()!;
    for (let i = 0; i < 100; i++) s.trainStep();
    const after = s.evaluate()!;
    expect(after.loss).toBeLessThan(before.loss * 0.5);
    expect(after.metrics.pixelMse).toBeLessThan(after.metrics.corruptedMse);
    const f = s.runProcedure('filters', {}, noop);
    expect(f.tensors.filters.shape).toEqual([64, 1, 28, 28]);
  });
});

describe('variational autoencoder', () => {
  it.each([
    { hidden: 16, latent: 2 },
    { hidden: 16, latent: 3, activation: 'tanh', beta: 2.5 },
  ])('backpropagation matches finite differences with ε fixed by the seed (%o)', (config) => {
    const s = new Session(vae, { config });
    s.batchSize = 2;
    expect(s.gradCheck(undefined, 3).maxRelError).toBeLessThan(1e-5);
  });

  it('ε is reproducible for a given graph seed (the premise of the gradient check)', () => {
    const s = new Session(vae);
    const b = s.dataset.example(1, s.config);
    const a = run(vae, s, b, 11);
    expect(run(vae, s, b, 11).out.eps.data).toEqual(a.out.eps.data);
    expect(run(vae, s, b, 11).loss).toBe(a.loss);
    expect(run(vae, s, b, 12).out.eps.data).not.toEqual(a.out.eps.data);
  });

  it('the KL term equals the closed form and a Monte Carlo estimate', () => {
    const s = new Session(vae, { seed: 4, config: { hidden: 64, latent: 3 } });
    for (let i = 0; i < 20; i++) s.trainStep();
    const b = digitBatch(mnist().testImages, mnist().testLabels, [0, 1, 2, 3, 4, 5], 'unit');
    const r = run(vae, s, b);
    const { mu, logvar } = r.out;
    const N = 6;
    const d = 3;
    let closed = 0;
    for (let i = 0; i < N * d; i++)
      closed += 0.5 * (mu.data[i] ** 2 + Math.exp(logvar.data[i]) - logvar.data[i] - 1);
    closed /= N;
    expect(r.metrics.kl).toBeCloseTo(closed, 4);
    // E_q[log q(z) − log p(z)] by sampling, for the first data point.
    const rng = new Rng(8);
    const M = 200000;
    let mc = 0;
    let exact0 = 0;
    for (let j = 0; j < d; j++) {
      const m = mu.data[j];
      const lv = logvar.data[j];
      exact0 += 0.5 * (m * m + Math.exp(lv) - lv - 1);
      for (let k = 0; k < M; k++) {
        const e = rng.normal();
        const z = m + Math.exp(0.5 * lv) * e;
        mc += (-0.5 * lv - 0.5 * e * e + 0.5 * z * z) / M;
      }
    }
    expect(Math.abs(mc - exact0)).toBeLessThan(0.02 * Math.max(1, exact0));
  });

  it('the reparameterised sample has mean μ and variance σ²', () => {
    const s = new Session(vae, { seed: 6, config: { hidden: 64, latent: 2 } });
    for (let i = 0; i < 20; i++) s.trainStep();
    const n = 2500;
    const b = replicate(s.dataset.example(2, s.config).inputs.image, n);
    const r = run(vae, s, b, 21);
    for (let j = 0; j < 2; j++) {
      const mu = r.out.mu.data[j];
      const sd = r.out.sigma.data[j];
      let m = 0;
      for (let i = 0; i < n; i++) m += r.out.z.data[i * 2 + j] / n;
      let v = 0;
      for (let i = 0; i < n; i++) v += (r.out.z.data[i * 2 + j] - m) ** 2 / (n - 1);
      expect(Math.abs(m - mu)).toBeLessThan((4 * sd) / Math.sqrt(n));
      expect(v / (sd * sd)).toBeGreaterThan(0.9);
      expect(v / (sd * sd)).toBeLessThan(1.1);
      // Exactly z = μ + σ ε for each draw.
      expect(r.out.z.data[j]).toBeCloseTo(mu + sd * r.out.eps.data[j], 5);
    }
  });

  it('training lowers the negative ELBO; β scales only the KL term', () => {
    const s = new Session(vae, { seed: 7, config: { hidden: 64 } });
    s.batchSize = 32;
    const before = s.evaluate()!;
    for (let i = 0; i < 80; i++) s.trainStep();
    const after = s.evaluate()!;
    expect(after.metrics.negElbo).toBeLessThan(before.metrics.negElbo - 100);
    const b = s.dataset.example(0, s.config);
    const r1 = run(vae, s, b, 5);
    s.setConfig({ beta: 3 });
    const r3 = run(vae, s, b, 5);
    expect(r3.metrics.recon).toBe(r1.metrics.recon);
    expect(r3.loss * 784).toBeCloseTo(r1.metrics.recon + 3 * r1.metrics.kl, 3);
    expect(r1.loss * 784).toBeCloseTo(r1.metrics.negElbo, 3);
  });

  it('prior samples, the 2-D manifold and the traversal have the documented shapes', () => {
    const s = new Session(vae, { config: { hidden: 32, latent: 2 } });
    expect(s.runProcedure('prior', {}, noop).tensors.samples.shape).toEqual([32, 1, 28, 28]);
    expect(s.runProcedure('manifold', {}, noop).tensors.grid.shape).toEqual([100, 1, 28, 28]);
    expect(s.runProcedure('latent', {}, noop).tensors.z.shape).toEqual([500, 2]);
    const s3 = new Session(vae, { config: { hidden: 32, latent: 3 } });
    expect(s3.runProcedure('manifold', {}, noop).tensors.grid.shape).toEqual([30, 1, 28, 28]);
    expect(probit(0.5)).toBeCloseTo(0, 12);
    expect(probit(0.975)).toBeCloseTo(1.959963985, 8);
    expect(probit(0.01)).toBeCloseTo(-2.326347874, 8);
  });

  const ckpt = join(PUBLIC, 'weights', 'vae-mnist.json');
  it.runIf(existsSync(ckpt))(
    'the shipped checkpoint loads and reproduces its recorded test −ELBO',
    () => {
      const file = JSON.parse(readFileSync(ckpt, 'utf8')) as CheckpointFile;
      const bin = readFileSync(join(PUBLIC, 'weights', 'vae-mnist.bin'));
      const data = new Float32Array(
        bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength),
      );
      expect(bin.byteLength).toBeLessThan(2 * 1024 * 1024);
      const s = new Session(vae);
      s.setConfig(file.config);
      s.loadCheckpoint('mnist', file.params, data);
      const d = mnist();
      const batch = digitBatch(
        d.testImages,
        d.testLabels,
        Array.from({ length: 1000 }, (_, i) => i),
        'unit',
      );
      const r = vaeNegElbo(s.params, s.config, batch, 0, 10);
      expect(r.negElbo).toBeCloseTo(file.provenance.metrics.testNegElbo, 2);
      expect(r.kl).toBeCloseTo(file.provenance.metrics.testKl, 2);
    },
  );
});
