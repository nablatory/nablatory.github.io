/**
 * Trains the Vision Transformer (default configuration: 7×7 patches, 3 layers,
 * D = 32, 4 heads) on the full MNIST training set and writes
 * public/weights/vit-mnist.{json,bin}.
 *
 * Usage: npx tsx scripts/train/vit-mnist.ts [steps=2500] [batch=32]
 */
import vit from '../../src/models/transformer/vit';
import { Session } from '../../src/runtime/session';
import { setMnist, digitBatch } from '../../src/models/data/mnist';
import { Graph, Rng } from '../../src/engine';
import { accuracy } from '../../src/models/util';
import { loadMnist } from '../data/fetch-mnist';
import { saveCheckpoint, log } from './common';

const steps = Number(process.argv[2] ?? 2500);
const batch = Number(process.argv[3] ?? 32);
const data = await loadMnist();
setMnist(data);

const session = new Session(vit, { seed: 1 });
session.batchSize = batch;
const peak = 3e-3;
const floor = 1e-4;
const warmup = 100;
const lrAt = (s: number) =>
  s <= warmup
    ? (peak * s) / warmup
    : floor + 0.5 * (peak - floor) * (1 + Math.cos((Math.PI * (s - warmup)) / (steps - warmup)));

const t0 = Date.now();
for (let s = 1; s <= steps; s++) {
  session.setOptimizer({ lr: lrAt(s) });
  const r = session.trainStep() as {
    losses: Record<string, number>;
    metrics: Record<string, number>;
  };
  log(
    s,
    steps,
    `loss ${r.losses.main.toFixed(4)} acc ${r.metrics.accuracy.toFixed(3)} (${((Date.now() - t0) / 1000).toFixed(0)} s)`,
  );
}

// Session.evaluate(): the first 500 test images (the browser's validation batch).
const validation = session.evaluate()!;
// Full 10,000-image test set.
let correct = 0;
let lossSum = 0;
for (let start = 0; start < 10000; start += 500) {
  const idx = Array.from({ length: 500 }, (_, i) => start + i);
  const b = digitBatch(data.testImages, data.testLabels, idx, 'unit');
  const g = new Graph({ params: session.params, record: false, grad: false, rng: new Rng(0) });
  const res = g.run(() => vit.forward(b, session.config, { training: false, rng: new Rng(0) }));
  correct += accuracy(res.outputs.probs.value, b.labels!) * 500;
  lossSum += res.loss!.value.data[0] * 500;
}
const metrics = {
  validationLoss: validation.loss,
  validationAccuracy: validation.metrics.accuracy,
  testAccuracy: correct / 10000,
  testLoss: lossSum / 10000,
  trainSeconds: Math.round((Date.now() - t0) / 1000),
};
console.log(metrics);

saveCheckpoint(session, 'vit-mnist', {
  dataset: 'MNIST training set (60,000 images), intensities / 255',
  script: 'scripts/train/vit-mnist.ts',
  metrics,
  notes: `Adam, batch ${batch}, ${steps} steps (${((steps * batch) / 60000).toFixed(2)} epochs); learning rate warmed up linearly to ${peak} over ${warmup} steps, then cosine-decayed to ${floor}; gradient norm clipped at 1. validationLoss/validationAccuracy: Session.evaluate() on the first 500 test images; testLoss/testAccuracy: all 10,000 test images.`,
});
