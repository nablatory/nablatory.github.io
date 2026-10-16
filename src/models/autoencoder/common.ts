import {
  Tensor,
  conv,
  convTranspose2d,
  declareConv,
  declareLinear,
  dense,
  flatten,
  p,
  reshape,
  scope,
  type ParamBuilder,
  type Var,
} from '../../engine';
import { bernoulliNll, squaredError } from './ops';
import type { Batch, Config, CustomInput, Procedure, ViewSpec } from '../types';
import { DIGITS, MNIST_PREPROCESSING, mnistDataset } from '../data/mnist';
import { input } from '../util';

/** Images are 28 × 28 MNIST digits with intensities in [0, 1]. */
export const SIDE = 28;
export const PIXELS = SIDE * SIDE;

export type Arch = 'mlp' | 'conv';
export type Act = (x: Var) => Var;

export const mnistUnit = <C extends Config>() => mnistDataset<C>(() => 'unit');

/** Channels of the convolutional encoder/decoder (28 → 14 → 7 and back). */
export const CONV_CH: [number, number] = [16, 32];
const CONV_FEAT = CONV_CH[1] * 7 * 7;

/** Declares the encoder trunk (everything before the code layer). Returns its feature size. */
export function declareTrunk(b: ParamBuilder, arch: Arch, hidden: number): number {
  if (arch === 'mlp') {
    declareLinear(b, 'enc1', PIXELS, hidden);
    return hidden;
  }
  declareConv(b, 'enc1', 1, CONV_CH[0], 4);
  declareConv(b, 'enc2', CONV_CH[0], CONV_CH[1], 4);
  return CONV_FEAT;
}

/** Encoder trunk: [N, 1, 28, 28] → [N, F]. */
export function trunk(x: Var, arch: Arch, act: Act): Var {
  if (arch === 'mlp') return scope('enc1', () => act(dense(flatten(x), 'enc1')));
  const h1 = scope('enc1', () => act(conv(x, 'enc1', { stride: 2, padding: 1 })));
  return scope('enc2', () => flatten(act(conv(h1, 'enc2', { stride: 2, padding: 1 }))));
}

/** PyTorch's ConvTranspose2d default: U(±1/√(C_out k²)). */
function declareConvT(b: ParamBuilder, name: string, cIn: number, cOut: number, k: number): void {
  const fanIn = cOut * k * k;
  b.add(`${name}.weight`, [cIn, cOut, k, k], { kind: 'lecunUniform', fanIn });
  b.add(`${name}.bias`, [cOut], { kind: 'lecunUniform', fanIn });
}

export function declareDecoder(b: ParamBuilder, arch: Arch, latent: number, hidden: number): void {
  if (arch === 'mlp') {
    declareLinear(b, 'dec1', latent, hidden);
    declareLinear(b, 'dec2', hidden, PIXELS);
    return;
  }
  declareLinear(b, 'dec1', latent, CONV_FEAT);
  declareConvT(b, 'dec2', CONV_CH[1], CONV_CH[0], 4);
  declareConvT(b, 'dec3', CONV_CH[0], 1, 4);
}

/** Decoder: [N, latent] → pixel logits [N, 1, 28, 28]. */
export function decoder(z: Var, arch: Arch, act: Act): Var {
  const N = z.shape[0];
  if (arch === 'mlp') {
    const h = scope('dec1', () => act(dense(z, 'dec1')));
    return scope('dec2', () => reshape(dense(h, 'dec2'), [N, 1, SIDE, SIDE]));
  }
  const h = scope('dec1', () => reshape(act(dense(z, 'dec1')), [N, CONV_CH[1], 7, 7]));
  const h2 = scope('dec2', () =>
    act(convTranspose2d(h, p('dec2.weight'), p('dec2.bias'), { stride: 2, padding: 1 })),
  );
  return scope('dec3', () =>
    convTranspose2d(h2, p('dec3.weight'), p('dec3.bias'), { stride: 2, padding: 1 }),
  );
}

/**
 * Reconstruction loss averaged over pixels and examples: Bernoulli
 * cross-entropy on the logits, or squared error on σ(logits).
 */
export function reconLoss(logits: Var, recon: Var, target: Tensor, kind: 'bce' | 'mse'): Var {
  const scale = 1 / target.size;
  return kind === 'bce' ? bernoulliNll(logits, target, scale) : squaredError(recon, target, scale);
}

/** Mean squared error per pixel (comparable across loss choices). */
export function pixelMse(recon: Tensor, target: Tensor): number {
  let s = 0;
  for (let i = 0; i < target.size; i++) {
    const d = recon.data[i] - target.data[i];
    s += d * d;
  }
  return s / target.size;
}

/**
 * Projection of [N, d] codes onto their first two principal components
 * (power iteration with deflation on the d × d sample covariance).
 * Returns the codes unchanged when d ≤ 2 (padding with zeros when d = 1).
 */
export function pca2(z: Tensor): { points: Tensor; explained: number } {
  const [N, d] = z.shape;
  if (d === 2) return { points: new Tensor([N, 2], Float32Array.from(z.data)), explained: 1 };
  if (d === 1) {
    const out = new Float32Array(N * 2);
    for (let i = 0; i < N; i++) out[i * 2] = z.data[i];
    return { points: new Tensor([N, 2], out), explained: 1 };
  }
  const mean = new Float64Array(d);
  for (let i = 0; i < N; i++) for (let j = 0; j < d; j++) mean[j] += z.data[i * d + j] / N;
  const cov = new Float64Array(d * d);
  for (let i = 0; i < N; i++)
    for (let a = 0; a < d; a++) {
      const za = z.data[i * d + a] - mean[a];
      for (let b = 0; b < d; b++) cov[a * d + b] += (za * (z.data[i * d + b] - mean[b])) / (N - 1);
    }
  let trace = 0;
  for (let a = 0; a < d; a++) trace += cov[a * d + a];
  const vecs: Float64Array[] = [];
  let captured = 0;
  for (let k = 0; k < 2; k++) {
    let v = new Float64Array(d).map((_, j) => 1 + 0.1 * j * (k + 1));
    let lambda = 0;
    for (let it = 0; it < 300; it++) {
      const w = new Float64Array(d);
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) w[a] += cov[a * d + b] * v[b];
      for (const u of vecs) {
        let dot = 0;
        for (let a = 0; a < d; a++) dot += w[a] * u[a];
        for (let a = 0; a < d; a++) w[a] -= dot * u[a];
      }
      let nrm = 0;
      for (let a = 0; a < d; a++) nrm += w[a] * w[a];
      nrm = Math.sqrt(nrm);
      if (nrm === 0) break;
      lambda = nrm;
      v = w.map((x) => x / nrm);
    }
    vecs.push(v);
    captured += lambda;
  }
  const out = new Float32Array(N * 2);
  for (let i = 0; i < N; i++)
    for (let k = 0; k < 2; k++) {
      let s = 0;
      for (let a = 0; a < d; a++) s += (z.data[i * d + a] - mean[a]) * vecs[k][a];
      out[i * 2 + k] = s;
    }
  return { points: new Tensor([N, 2], out), explained: trace > 0 ? captured / trace : 0 };
}

/** Encodes the validation digits and returns 2-D points coloured by digit label. */
export function latentProcedure<C extends Config>(
  encode: (x: Var, c: C) => Var,
  what = 'code z',
): Procedure<C> {
  return {
    id: 'latent',
    label: 'Latent space',
    description: `Encodes the 500 validation digits and plots each ${what}, coloured by its label. Codes with more than two dimensions are projected onto their first two principal components.`,
    run({ config, dataset }) {
      const b = dataset.validation!(config);
      const z = encode(input('image', b.inputs.image, false), config).value;
      const { points, explained } = pca2(z);
      return {
        tensors: {
          z: points,
          labels: new Tensor([b.labels!.length], Float32Array.from(b.labels!)),
        },
        scalars: { dimensions: z.shape[1], varianceShown: explained },
        meta: { classNames: DIGITS },
      };
    },
  };
}

export const latentView = (title = 'Latent codes of 500 validation digits'): ViewSpec => ({
  kind: 'scatter',
  title,
  source: 'proc:latent:z',
  colorBy: 'proc:latent:labels',
});

export const unitCustomInput: CustomInput = {
  kind: 'image',
  channels: 1,
  height: SIDE,
  width: SIDE,
  invertUploads: true,
  preprocessing: MNIST_PREPROCESSING.unit,
};

export function unitCustomBatch(t: Tensor): Batch {
  const x = new Float32Array(PIXELS);
  for (let i = 0; i < PIXELS; i++) x[i] = Math.max(0, Math.min(1, t.data[i]));
  return {
    inputs: { image: new Tensor([1, 1, SIDE, SIDE], x) },
    meta: { classNames: DIGITS, caption: 'Your drawing' },
  };
}

export const ACT_OPTIONS = [
  { value: 'relu', label: 'ReLU' },
  { value: 'tanh', label: 'tanh' },
  { value: 'sigmoid', label: 'sigmoid' },
];
