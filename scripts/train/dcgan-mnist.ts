/**
 * Trains DCGAN (default configuration: z ∈ ℝ³², g = d = 16, non-saturating
 * generator loss) on the full MNIST training set, pixels mapped to [−1, 1], and
 * writes public/weights/dcgan-mnist.{json,bin}.
 *
 * Recorded measurements (no FID or Inception score is computed):
 *   test*         evaluation-mode losses and mean D outputs on the 1,000 shipped
 *                 test images and 1,000 generated images from Rng(0) noise
 *                 (dcganEvaluate in src/models/gan/dcgan.ts; recomputed by tests/gan.test.ts)
 *   lenet*        LeNet-5 classifier statistics of those 1,000 generated images,
 *                 and of the 1,000 real test images for reference (scripts/train/dcgan-lenet.ts)
 *   final*        mean training losses over the last 100 steps
 *
 * Usage: npx tsx scripts/train/dcgan-mnist.ts [steps=2500] [batch=32]
 */
import dcgan, { dcganEvaluate } from '../../src/models/gan/dcgan';
import { Session, type TrainRecord } from '../../src/runtime/session';
import { setMnist, digitBatch } from '../../src/models/data/mnist';
import { loadMnist } from '../data/fetch-mnist';
import { saveCheckpoint, log } from './common';
import { lenetStats, loadLenet } from './dcgan-lenet';

const total = Number(process.argv[2] ?? 2500);
const batch = Number(process.argv[3] ?? 32);
const data = await loadMnist();
setMnist(data);

const session = new Session(dcgan, { seed: 1 });
session.batchSize = batch;
session.setOptimizer({ kind: 'adam', lr: 2e-4, beta1: 0.5, beta2: 0.999 });
const c0 = process.cpuUsage();
const t0 = Date.now();
const tail: TrainRecord[] = [];
for (let s = 1; s <= total; s++) {
  const r = session.trainStep() as TrainRecord;
  tail.push(r);
  if (tail.length > 100) tail.shift();
  log(
    s,
    total,
    `L_D ${r.losses.discriminator.toFixed(3)} L_G ${r.losses.generator.toFixed(3)} D(x) ${r.metrics['D(x)'].toFixed(3)} D(G(z)) ${r.metrics['D(G(z))'].toFixed(3)} (${((Date.now() - t0) / 1000).toFixed(0)} s)`,
  );
}
const cpu = process.cpuUsage(c0);

const real = digitBatch(
  data.testImages,
  data.testLabels,
  Array.from({ length: 1000 }, (_, i) => i),
  'symmetric',
);
const ev = dcganEvaluate(session.params, session.config, real, 0, 1000);
const lenet = loadLenet();
const gen = lenetStats(lenet, ev.samples);
const ref = lenetStats(lenet, real.inputs.image);
const mean = (f: (r: TrainRecord) => number) => tail.reduce((s, r) => s + f(r), 0) / tail.length;

const metrics: Record<string, number> = {
  testDLoss: ev.metrics.dLoss,
  testGLoss: ev.metrics.gLoss,
  testMeanDReal: ev.metrics.meanDReal,
  testMeanDFake: ev.metrics.meanDFake,
  finalTrainDLoss: mean((r) => r.losses.discriminator),
  finalTrainGLoss: mean((r) => r.losses.generator),
  lenetScoreGenerated: gen.lenetScore,
  lenetMeanMaxProbGenerated: gen.meanMaxProb,
  lenetScoreRealTest: ref.lenetScore,
  lenetMeanMaxProbRealTest: ref.meanMaxProb,
  trainCpuSeconds: Math.round((cpu.user + cpu.system) / 1e6),
};
gen.classFractions.forEach((f, k) => (metrics[`lenetFractionGenerated${k}`] = f));
console.log(metrics);

saveCheckpoint(session, 'dcgan-mnist', {
  dataset: 'MNIST training set (60,000 images), pixels mapped to [−1, 1]',
  script: 'scripts/train/dcgan-mnist.ts',
  metrics,
  notes:
    `Adam (lr 2e-4, β1 0.5, β2 0.999), batch ${batch}, ${total} alternating D/G steps (${((total * batch) / 60000).toFixed(2)} epochs). ` +
    'test*: evaluation-mode (running batch-norm statistics) losses and mean D outputs on the 1,000 shipped test images and 1,000 generated images from noise drawn with Rng(0). ' +
    'lenet*: statistics of the shipped LeNet-5 classifier on the same 1,000 generated images (and on the 1,000 real test images for reference); ' +
    'lenetScore is the Inception-score formula with LeNet-5 in place of Inception. No FID or Inception score was computed.',
});
