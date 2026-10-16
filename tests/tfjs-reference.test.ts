/**
 * Independent cross-check against TensorFlow.js (CPU backend, float32):
 * whole computations, not single ops, so that composition, broadcasting and
 * layout conversions are exercised. Tolerances reflect float32 arithmetic.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-core/dist/register_all_gradients';
import {
  Graph,
  Rng,
  Tensor,
  backward,
  conv2d,
  crossEntropy,
  flatten,
  linear,
  maxPool2d,
  relu,
  softmax,
  matmul,
  transpose,
  tanh,
  sigmoid,
  mean,
  mulScalar,
  layerNorm,
  type Var,
} from '../src/engine';

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

const rng = new Rng(2024);
const rand = (shape: number[], s = 1) =>
  new Tensor(
    shape,
    Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () => s * rng.normal()),
  );

function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    num = Math.max(num, Math.abs(a[i] - b[i]));
    den = Math.max(den, Math.abs(b[i]));
  }
  return num / Math.max(den, 1e-8);
}

/** Run an engine computation and return the loss and gradients of the named inputs. */
function engine(inputs: Record<string, Tensor>, f: (v: Record<string, Var>) => Var) {
  const g = new Graph({ grad: true, record: true });
  let loss!: Var;
  const vars: Record<string, Var> = {};
  g.run(() => {
    for (const [k, t] of Object.entries(inputs)) vars[k] = g.input(k, t, true);
    loss = f(vars);
    backward(loss);
  });
  return {
    loss: loss.value.data[0],
    grads: Object.fromEntries(Object.entries(vars).map(([k, v]) => [k, v.grad!.data])),
  };
}

/** NCHW ↔ NHWC and OIHW → HWIO, as plain arrays. */
const nchwToNhwc = (t: Tensor) =>
  tf.transpose(tf.tensor(Array.from(t.data), t.shape), [0, 2, 3, 1]);
const oihwToHwio = (t: Tensor) =>
  tf.transpose(tf.tensor(Array.from(t.data), t.shape), [2, 3, 1, 0]);

describe('cross-check against TensorFlow.js', () => {
  it('conv → ReLU → max-pool → linear → softmax cross-entropy', () => {
    const N = 3,
      C = 2,
      H = 8,
      O = 4,
      K = 3,
      classes = 5;
    const x = rand([N, C, H, H]);
    const w = rand([O, C, K, K], 0.4);
    const b = rand([O], 0.1);
    const fw = rand([classes, O * 4 * 4], 0.2);
    const fb = rand([classes], 0.1);
    const labels = [1, 4, 0];

    const ours = engine({ x, w, b, fw, fb }, (v) => {
      const h = maxPool2d(relu(conv2d(v.x, v.w, v.b, { padding: 1 })), 2, 2);
      return crossEntropy(linear(flatten(h), v.fw, v.fb), labels);
    });

    const f = (
      tx: tf.Tensor4D,
      tw: tf.Tensor4D,
      tb: tf.Tensor1D,
      tfw: tf.Tensor2D,
      tfb: tf.Tensor1D,
    ) => {
      const h = tf.maxPool(
        tf.relu(tf.add(tf.conv2d(tx, tw, 1, 'same'), tb)) as tf.Tensor4D,
        2,
        2,
        'valid',
      ) as tf.Tensor4D;
      // Back to NCHW before flattening so feature order matches the engine.
      const flat = tf.reshape(tf.transpose(h, [0, 3, 1, 2]), [N, O * 16]);
      const logits = tf.add(tf.matMul(flat, tfw, false, true), tfb);
      return tf.losses.softmaxCrossEntropy(tf.oneHot(labels, classes), logits) as tf.Scalar;
    };
    const args = [
      nchwToNhwc(x),
      oihwToHwio(w),
      tf.tensor1d(Array.from(b.data)),
      tf.tensor2d(Array.from(fw.data), fw.shape as [number, number]),
      tf.tensor1d(Array.from(fb.data)),
    ];
    const { value, grads } = tf.valueAndGrads(f as (...a: tf.Tensor[]) => tf.Scalar)(args);

    expect(Math.abs(value.dataSync()[0] - ours.loss)).toBeLessThan(1e-5);
    expect(relErr(ours.grads.x, tf.transpose(grads[0], [0, 3, 1, 2]).dataSync())).toBeLessThan(
      1e-4,
    );
    expect(relErr(ours.grads.w, tf.transpose(grads[1], [3, 2, 0, 1]).dataSync())).toBeLessThan(
      1e-4,
    );
    expect(relErr(ours.grads.b, grads[2].dataSync())).toBeLessThan(1e-4);
    expect(relErr(ours.grads.fw, grads[3].dataSync())).toBeLessThan(1e-4);
    expect(relErr(ours.grads.fb, grads[4].dataSync())).toBeLessThan(1e-4);
  });

  it('single-head attention with layer normalisation, tanh and sigmoid', () => {
    const T = 5,
      D = 6;
    const x = rand([T, D]);
    const wq = rand([D, D], 0.4);
    const wk = rand([D, D], 0.4);
    const wv = rand([D, D], 0.4);

    const ours = engine({ x, wq, wk, wv }, (v) => {
      const h = layerNorm(v.x);
      const q = matmul(h, v.wq);
      const k = matmul(h, v.wk);
      const a = softmax(mulScalar(matmul(q, transpose(k, [1, 0])), 1 / Math.sqrt(D)), -1);
      const y = matmul(a, tanh(matmul(h, v.wv)));
      return mean(sigmoid(y));
    });

    const f = (tx: tf.Tensor2D, tq: tf.Tensor2D, tk: tf.Tensor2D, tv: tf.Tensor2D) => {
      const m = tf.moments(tx, -1, true);
      const h = tf.div(tf.sub(tx, m.mean), tf.sqrt(tf.add(m.variance, 1e-5))) as tf.Tensor2D;
      const a = tf.softmax(
        tf.mul(tf.matMul(tf.matMul(h, tq), tf.matMul(h, tk), false, true), 1 / Math.sqrt(D)),
      );
      return tf.mean(tf.sigmoid(tf.matMul(a, tf.tanh(tf.matMul(h, tv))))) as tf.Scalar;
    };
    const t2 = (t: Tensor) => tf.tensor2d(Array.from(t.data), t.shape as [number, number]);
    const { value, grads } = tf.valueAndGrads(f as (...a: tf.Tensor[]) => tf.Scalar)([
      t2(x),
      t2(wq),
      t2(wk),
      t2(wv),
    ]);

    expect(Math.abs(value.dataSync()[0] - ours.loss)).toBeLessThan(1e-6);
    for (const [i, k] of ['x', 'wq', 'wk', 'wv'].entries())
      expect(relErr(ours.grads[k], grads[i].dataSync())).toBeLessThan(1e-4);
  });
});
