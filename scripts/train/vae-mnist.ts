/**
 * Trains the VAE (default configuration: 784–256–2 Gaussian encoder, 2–256–784
 * Bernoulli decoder, ReLU, β = 1) on the full MNIST training set and writes
 * public/weights/vae-mnist.{json,bin}.
 *
 * The recorded metric is the negative ELBO in nats per image on the 1,000 test
 * images shipped in public/data/mnist (the first 1,000 of the MNIST test set),
 * averaged over 10 draws of ε from Rng(0); tests/autoencoder.test.ts recomputes it.
 *
 * Usage: npx tsx scripts/train/vae-mnist.ts [epochs=4]
 */
import vae, { vaeNegElbo } from '../../src/models/autoencoder/vae';
import { Session } from '../../src/runtime/session';
import { setMnist, digitBatch } from '../../src/models/data/mnist';
import { loadMnist } from '../data/fetch-mnist';
import { saveCheckpoint, log } from './common';

const epochs = Number(process.argv[2] ?? 4);
const data = await loadMnist();
setMnist(data);

const session = new Session(vae, { seed: 1 });
session.batchSize = 100;
session.setOptimizer({ kind: 'adam', lr: 3e-3 });
const stepsPerEpoch = Math.floor(60000 / session.batchSize);
const total = epochs * stepsPerEpoch;
const c0 = process.cpuUsage();
const t0 = Date.now();
for (let s = 1; s <= total; s++) {
  if (s === (epochs - 1) * stepsPerEpoch + 1) session.setOptimizer({ lr: 1e-3 });
  const r = session.trainStep() as { metrics: Record<string, number> };
  log(
    s,
    total,
    `−ELBO ${r.metrics.negElbo.toFixed(2)} (recon ${r.metrics.recon.toFixed(2)}, KL ${r.metrics.kl.toFixed(2)}) ${((Date.now() - t0) / 1000).toFixed(0)} s`,
  );
}
const cpu = process.cpuUsage(c0);
const cpuSeconds = (cpu.user + cpu.system) / 1e6;

const shipped = digitBatch(
  data.testImages,
  data.testLabels,
  Array.from({ length: 1000 }, (_, i) => i),
  'unit',
);
const sub = vaeNegElbo(session.params, session.config, shipped, 0, 10);

// Full 10,000-image test set, one ε draw per image.
let full = 0;
for (let start = 0; start < 10000; start += 1000) {
  const b = digitBatch(
    data.testImages,
    data.testLabels,
    Array.from({ length: 1000 }, (_, i) => start + i),
    'unit',
  );
  full += vaeNegElbo(session.params, session.config, b, 1 + start, 1).negElbo / 10;
}

const metrics = {
  testNegElbo: sub.negElbo,
  testRecon: sub.recon,
  testKl: sub.kl,
  fullTestNegElbo: full,
  trainCpuSeconds: Math.round(cpuSeconds),
};
console.log(metrics);

saveCheckpoint(session, 'vae-mnist', {
  dataset: 'MNIST training set (60,000 images), intensities / 255 used as Bernoulli targets',
  script: 'scripts/train/vae-mnist.ts',
  metrics,
  notes:
    `Adam, batch 100, ${epochs} epochs (${total} steps); lr 3e-3, then 1e-3 for the final epoch. ` +
    'testNegElbo / testRecon / testKl: nats per image on the 1,000 shipped test images (first 1,000 of the MNIST test set), ' +
    'averaged over 10 draws of ε from Rng(0) (vaeNegElbo in src/models/autoencoder/vae.ts). ' +
    'fullTestNegElbo: same quantity on all 10,000 test images with a single ε draw. ' +
    'Non-binarised pixels, so these values are not comparable with results on binarised MNIST.',
});
