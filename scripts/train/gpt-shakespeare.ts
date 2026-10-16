/**
 * Trains the character-level GPT (default configuration) on the first 90 % of
 * public/data/text/shakespeare.txt and writes
 * public/weights/gpt-shakespeare.{json,bin}.
 *
 * Usage: npx tsx scripts/train/gpt-shakespeare.ts [steps=1600] [batch=16]
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import gpt from '../../src/models/transformer/gpt';
import { Session } from '../../src/runtime/session';
import { setShakespeare, windowBatch } from '../../src/models/data/text';
import { Graph, Rng } from '../../src/engine';
import { saveCheckpoint, log } from './common';

const steps = Number(process.argv[2] ?? 1600);
const batch = Number(process.argv[3] ?? 16);
const text = readFileSync(
  join(import.meta.dirname, '..', '..', 'public', 'data', 'text', 'shakespeare.txt'),
  'utf8',
);
const corpus = setShakespeare(text);

const session = new Session(gpt, { seed: 1 });
session.batchSize = batch;
const peak = 3e-3;
const floor = 2e-4;
const warmup = 100;
const lrAt = (s: number) =>
  s <= warmup
    ? (peak * s) / warmup
    : floor + 0.5 * (peak - floor) * (1 + Math.cos((Math.PI * (s - warmup)) / (steps - warmup)));

const t0 = Date.now();
let ema = 0;
for (let s = 1; s <= steps; s++) {
  session.setOptimizer({ lr: lrAt(s) });
  const r = session.trainStep() as { losses: Record<string, number> };
  ema = s === 1 ? r.losses.main : 0.98 * ema + 0.02 * r.losses.main;
  log(
    s,
    steps,
    `loss ${ema.toFixed(4)} lr ${lrAt(s).toExponential(2)} (${((Date.now() - t0) / 1000).toFixed(0)} s)`,
  );
  if (s % 200 === 0) console.log(`  validation loss ${session.evaluate()!.loss.toFixed(4)}`);
}

// Validation loss as Session.evaluate() computes it (32 fixed held-out windows), and
// over every non-overlapping window of the held-out 10 %.
const validationLoss = session.evaluate()!.loss;
const T = session.config.context as number;
let sum = 0;
let count = 0;
for (let start = corpus.split; start + T + 1 <= corpus.ids.length; start += T * 32) {
  const starts: number[] = [];
  for (let k = 0; k < 32 && start + k * T + T + 1 <= corpus.ids.length; k++)
    starts.push(start + k * T);
  const b = windowBatch(corpus, starts, T);
  const g = new Graph({ params: session.params, record: false, grad: false, rng: new Rng(0) });
  const res = g.run(() => gpt.forward(b, session.config, { training: false, rng: new Rng(0) }));
  sum += res.loss!.value.data[0] * starts.length;
  count += starts.length;
}
const heldOutLoss = sum / count;
const metrics = {
  validationLoss,
  heldOutLoss,
  heldOutBitsPerChar: heldOutLoss / Math.LN2,
  trainSeconds: Math.round((Date.now() - t0) / 1000),
};
console.log(metrics);

const sample = session.runProcedure('sample', {}, { shouldStop: () => false, progress: () => {} });
console.log('--- sample (prompt “First Citizen:”, τ = 0.8, top-k 10) ---\n' + sample.text);

saveCheckpoint(session, 'gpt-shakespeare', {
  dataset: `tiny-shakespeare subset shipped in public/data/text (training split: first ${corpus.split} characters)`,
  script: 'scripts/train/gpt-shakespeare.ts',
  metrics,
  notes: `Adam, batch ${batch} windows of ${T} characters, ${steps} steps; learning rate warmed up linearly to ${peak} over ${warmup} steps, then cosine-decayed to ${floor}; gradient norm clipped at 1. validationLoss is Session.evaluate() on 32 fixed held-out windows; heldOutLoss averages every non-overlapping window of the last 10 %.`,
});
