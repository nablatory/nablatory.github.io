/**
 * Shared offline training loop for the MNIST classifiers of the convolutional
 * family (VGG, ResNet). Uses the same Session class as the browser, the full
 * 60,000-image MNIST training set, and evaluates on the full 10,000-image
 * test set. Writes public/weights/<name>.{json,bin}.
 */
import type { ModelDef } from '../../src/models/types';
import { Session } from '../../src/runtime/session';
import { setMnist, digitBatch } from '../../src/models/data/mnist';
import { Graph, Rng } from '../../src/engine';
import { accuracy } from '../../src/models/util';
import { loadMnist } from '../data/fetch-mnist';
import { saveCheckpoint, log } from './common';

export async function trainMnistClassifier(
  model: ModelDef,
  name: string,
  opts: { steps: number; batch: number; lr: number; seed?: number },
) {
  const data = await loadMnist();
  setMnist(data);
  const session = new Session(model, { seed: opts.seed ?? 1 });
  session.batchSize = opts.batch;
  session.setOptimizer({ kind: 'adam', lr: opts.lr });
  const decayAt = Math.floor((2 * opts.steps) / 3);
  const t0 = Date.now();
  for (let s = 1; s <= opts.steps; s++) {
    if (s === decayAt + 1) session.setOptimizer({ lr: opts.lr * 0.3 });
    const r = session.trainStep() as {
      losses: Record<string, number>;
      metrics: Record<string, number>;
    };
    log(
      s,
      opts.steps,
      `loss ${r.losses.main.toFixed(4)} acc ${r.metrics.accuracy.toFixed(3)} (${((Date.now() - t0) / 1000).toFixed(0)} s)`,
    );
  }

  // Full 10,000-image test set, evaluation mode (batch norm uses running statistics).
  let correct = 0;
  let lossSum = 0;
  for (let start = 0; start < 10000; start += 500) {
    const idx = Array.from({ length: 500 }, (_, i) => start + i);
    const batch = digitBatch(data.testImages, data.testLabels, idx, 'unit');
    const g = new Graph({ params: session.params, record: false, grad: false, rng: new Rng(0) });
    const res = g.run(() =>
      model.forward(batch, session.config, { training: false, rng: new Rng(0) }),
    );
    correct += accuracy(res.outputs.probs.value, batch.labels!) * 500;
    lossSum += res.loss!.value.data[0] * 500;
  }
  // The browser's validation batch (first 500 test images), recorded so the checkpoint test can reproduce it.
  const v = session.evaluate()!;
  const metrics = {
    testAccuracy: correct / 10000,
    testLoss: lossSum / 10000,
    validationLoss: v.loss,
    validationAccuracy: v.metrics.accuracy,
  };
  console.log(metrics, `${((Date.now() - t0) / 1000).toFixed(0)} s`);

  saveCheckpoint(session, name, {
    dataset: 'MNIST training set (60,000 images), intensities / 255',
    script: `scripts/train/${model.id}.ts`,
    metrics,
    notes: `Adam, batch ${opts.batch}, ${opts.steps} steps (${((opts.steps * opts.batch) / 60000).toFixed(2)} epochs); lr ${opts.lr} then ${opts.lr * 0.3} for the final third. Test metrics on the 10,000 MNIST test images; validation metrics on the first 500.`,
  });
}
