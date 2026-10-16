/**
 * Classifier statistics of generated digits, using the shipped LeNet-5
 * checkpoint (public/weights/lenet-5-mnist). This is a diagnostic only: it is
 * neither FID nor the Inception score, and LeNet-5 was trained on real digits,
 * so its confidence on generated images is a weak proxy for quality.
 *
 *   classFractions  fraction of samples assigned to each digit (argmax)
 *   meanMaxProb     mean of max_y p(y | x)
 *   lenetScore      exp(E_x KL(p(y | x) ‖ p̄(y))), the Inception-score formula
 *                   evaluated with LeNet-5 in place of Inception (1 … 10)
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Graph, Rng, Tensor } from '../../src/engine';
import lenet5 from '../../src/models/convolutional/lenet5';
import { encodeDigit } from '../../src/models/data/mnist';
import type { CheckpointFile } from '../../src/runtime/checkpoint';
import { Session } from '../../src/runtime/session';

const WEIGHTS = join(import.meta.dirname, '..', '..', 'public', 'weights');

export function loadLenet(): Session {
  const file = JSON.parse(
    readFileSync(join(WEIGHTS, 'lenet-5-mnist.json'), 'utf8'),
  ) as CheckpointFile;
  const bin = readFileSync(join(WEIGHTS, 'lenet-5-mnist.bin'));
  const s = new Session(lenet5);
  s.setConfig(file.config);
  s.loadCheckpoint(
    'mnist',
    file.params,
    new Float32Array(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)),
  );
  return s;
}

/** Images [N, 1, 28, 28] with values in [lo, hi] (generated: [−1, 1]). */
export function lenetStats(lenet: Session, images: Tensor, lo = -1, hi = 1) {
  const N = images.shape[0];
  const x = new Float32Array(N * 32 * 32);
  const u8 = new Float32Array(784);
  for (let n = 0; n < N; n++) {
    for (let i = 0; i < 784; i++)
      u8[i] = Math.max(0, Math.min(1, (images.data[n * 784 + i] - lo) / (hi - lo))) * 255;
    encodeDigit(u8, 0, 'lecun', x, n * 1024);
  }
  const g = new Graph({ params: lenet.params, record: false, grad: false, rng: new Rng(0) });
  const probs = g.run(() =>
    lenet5.forward({ inputs: { image: new Tensor([N, 1, 32, 32], x) } }, lenet.config, {
      training: false,
      rng: new Rng(0),
    }),
  ).outputs.probs.value;
  const marginal = new Array(10).fill(0);
  const counts = new Array(10).fill(0);
  let maxProb = 0;
  for (let n = 0; n < N; n++) {
    let best = 0;
    for (let k = 0; k < 10; k++) {
      marginal[k] += probs.data[n * 10 + k] / N;
      if (probs.data[n * 10 + k] > probs.data[n * 10 + best]) best = k;
    }
    counts[best]++;
    maxProb += probs.data[n * 10 + best] / N;
  }
  let kl = 0;
  for (let n = 0; n < N; n++)
    for (let k = 0; k < 10; k++) {
      const q = probs.data[n * 10 + k];
      if (q > 0) kl += (q * Math.log(q / marginal[k])) / N;
    }
  return {
    classFractions: counts.map((c) => c / N),
    meanMaxProb: maxProb,
    lenetScore: Math.exp(kl),
  };
}
