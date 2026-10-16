import {
  Tensor,
  activeGraph,
  batchNorm,
  crossEntropy,
  p,
  reshape,
  transpose,
  type ParamBuilder,
  type Var,
} from '../../engine';
import type { Batch } from '../types';
import { DIGITS, encodeDigit, type MnistEncoding } from '../data/mnist';

/** Momentum of the running-statistics update (PyTorch default). */
export const BN_MOMENTUM = 0.1;

/**
 * Batch-norm parameters: trainable scale γ (`.weight`) and shift β (`.bias`),
 * plus running mean and variance stored as fixed tensors so they are saved
 * with checkpoints. Training-mode forward passes update the running
 * statistics in place.
 */
export function declareBatchNorm(b: ParamBuilder, name: string, channels: number): void {
  b.add(`${name}.weight`, [channels], { kind: 'ones' });
  b.add(`${name}.bias`, [channels], { kind: 'zeros' });
  b.fixed(
    `${name}.runningMean`,
    Tensor.zeros([channels]),
    'Running mean (updated by training-mode forward passes)',
  );
  b.fixed(
    `${name}.runningVar`,
    Tensor.full([channels], 1),
    'Running variance, unbiased (updated by training-mode forward passes)',
  );
}

/** Batch normalisation with the parameters declared by `declareBatchNorm`. */
export function bn(x: Var, name: string): Var {
  const store = activeGraph()?.params;
  if (!store) throw new Error(`batch norm ${name} outside a graph with parameters`);
  return batchNorm(x, p(`${name}.weight`), p(`${name}.bias`), {
    mean: store.get(`${name}.runningMean`),
    var: store.get(`${name}.runningVar`),
    momentum: BN_MOMENTUM,
  });
}

export interface RfLayer {
  kernel: number;
  stride: number;
}

/**
 * Receptive-field size after each layer: r_l = r_{l−1} + (k_l − 1)·j_{l−1},
 * j_l = j_{l−1}·s_l, with r_0 = j_0 = 1 (Araujo et al., 2019).
 */
export function receptiveFields(layers: RfLayer[]): { size: number; jump: number }[] {
  let r = 1;
  let j = 1;
  return layers.map((l) => {
    r += (l.kernel - 1) * j;
    j *= l.stride;
    return { size: r, jump: j };
  });
}

/**
 * Pixel-wise cross-entropy: logits [N, K, H, W], integer labels with N·H·W
 * entries (row-major n, y, x). Mean over all pixels.
 */
export function pixelCrossEntropy(logits: Var, labels: ArrayLike<number>): Var {
  const K = logits.shape[1];
  const nhwk = transpose(logits, [0, 2, 3, 1]);
  return crossEntropy(reshape(nhwk, [-1, K]), labels);
}

/** Root-mean-square of a tensor's entries. */
export function rms(t: Tensor): number {
  let s = 0;
  for (let i = 0; i < t.size; i++) s += t.data[i] * t.data[i];
  return Math.sqrt(s / Math.max(1, t.size));
}

/** Euclidean norm of a tensor's entries. */
export function l2(t: Tensor | null | undefined): number {
  if (!t) return 0;
  let s = 0;
  for (let i = 0; i < t.size; i++) s += t.data[i] * t.data[i];
  return Math.sqrt(s);
}

/** Side length of the bounding box of the non-zero entries of an H×W map (0 if all zero). */
export function supportExtent(d: ArrayLike<number>, H: number, W: number, tol = 0): number {
  let y0 = H;
  let y1 = -1;
  let x0 = W;
  let x1 = -1;
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++)
      if (Math.abs(d[y * W + x]) > tol) {
        y0 = Math.min(y0, y);
        y1 = Math.max(y1, y);
        x0 = Math.min(x0, x);
        x1 = Math.max(x1, x);
      }
  return y1 < 0 ? 0 : Math.max(y1 - y0 + 1, x1 - x0 + 1);
}

/** A user drawing ([1, 1, 28, 28], ink = 1) encoded like the MNIST training data. */
export function digitCustomBatch(input: Tensor, enc: MnistEncoding, size: number): Batch {
  const u8 = new Float32Array(784);
  for (let i = 0; i < 784; i++) u8[i] = Math.max(0, Math.min(1, input.data[i])) * 255;
  const x = new Float32Array(size * size);
  encodeDigit(u8, 0, enc, x, 0);
  return {
    inputs: { image: new Tensor([1, 1, size, size], x) },
    meta: { classNames: DIGITS, caption: 'Your drawing' },
  };
}
