/**
 * Trains LeNet-5 (default configuration) on the full MNIST training set and
 * writes public/weights/lenet-5-mnist.{json,bin}.
 *
 * Usage: npx tsx scripts/train/lenet5.ts [epochs=3]
 */
import lenet5 from '../../src/models/convolutional/lenet5';
import { Session } from '../../src/runtime/session';
import { setMnist, digitBatch } from '../../src/models/data/mnist';
import { Graph, Rng } from '../../src/engine';
import { accuracy } from '../../src/models/util';
import { loadMnist } from '../data/fetch-mnist';
import { saveCheckpoint, log } from './common';

const epochs = Number(process.argv[2] ?? 3);
const data = await loadMnist();
setMnist(data);

const session = new Session(lenet5, { seed: 1 });
session.batchSize = 64;
session.setOptimizer({ kind: 'adam', lr: 1e-3 });
const stepsPerEpoch = Math.floor(60000 / 64);
const total = epochs * stepsPerEpoch;
const t0 = Date.now();
for (let s = 1; s <= total; s++) {
  if (s === 2 * stepsPerEpoch + 1) session.setOptimizer({ lr: 3e-4 });
  const r = session.trainStep() as {
    losses: Record<string, number>;
    metrics: Record<string, number>;
  };
  log(
    s,
    total,
    `loss ${r.losses.main.toFixed(4)} acc ${r.metrics.accuracy.toFixed(3)} (${((Date.now() - t0) / 1000).toFixed(0)} s)`,
  );
}

// Full 10,000-image test set.
let correct = 0;
let lossSum = 0;
for (let start = 0; start < 10000; start += 500) {
  const idx = Array.from({ length: 500 }, (_, i) => start + i);
  const batch = digitBatch(data.testImages, data.testLabels, idx, 'lecun');
  const g = new Graph({ params: session.params, record: false, grad: false, rng: new Rng(0) });
  const res = g.run(() =>
    lenet5.forward(batch, session.config, { training: false, rng: new Rng(0) }),
  );
  correct += accuracy(res.outputs.probs.value, batch.labels!) * 500;
  lossSum += res.loss!.value.data[0] * 500;
}
const metrics = { testAccuracy: correct / 10000, testLoss: lossSum / 10000 };
console.log(metrics);

saveCheckpoint(session, 'lenet-5-mnist', {
  dataset: 'MNIST training set (60,000 images), LeCun encoding',
  script: 'scripts/train/lenet5.ts',
  metrics,
  notes: `Adam, batch 64, ${epochs} epochs; lr 1e-3 then 3e-4 for the final epoch(s).`,
});
