/**
 * Trains the RSSM world model (default configuration) on the point-mass
 * trajectory dataset and writes public/weights/rssm-point-mass.{json,bin}.
 *
 * Usage: npx tsx scripts/train/rssm.ts [steps=6000]
 */
import rssm from '../../src/models/world-model/rssm';
import { Session } from '../../src/runtime/session';
import { saveCheckpoint, log } from './common';

const noop = { shouldStop: () => false, progress: () => {} };
const total = Number(process.argv[2] ?? 6000);
const session = new Session(rssm, { seed: 1 });
session.setOptimizer({ kind: 'adam', lr: 3e-3 }, 16);
const t0 = Date.now();
for (let s = 1; s <= total; s++) {
  if (s === Math.floor(total * 0.6)) session.setOptimizer({ lr: 1e-3 });
  if (s === Math.floor(total * 0.85)) session.setOptimizer({ lr: 3e-4 });
  const r = session.trainStep() as { losses: Record<string, number> };
  log(s, total, `loss ${r.losses.main.toFixed(4)} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}

const val = session.evaluate()!;
// Open-loop pixel error (mean squared error per pixel) averaged over the 8 held-out episodes, 5 context frames.
let err5 = 0;
let err15 = 0;
for (let e = 1; e <= 8; e++) {
  const r = session.runProcedure(
    'openloop',
    { episode: e, context: 5, horizon: 15, sample: false },
    noop,
  );
  err5 += r.tensors.pixelError.data[5 + 4] / 8;
  err15 += r.tensors.pixelError.data[5 + 14] / 8;
}
const metrics = {
  validationLoss: val.loss,
  ...val.metrics,
  openLoopPixelMse5: err5,
  openLoopPixelMse15: err15,
};
console.log(metrics);
saveCheckpoint(session, 'rssm-point-mass', {
  dataset:
    'Point-mass reaching: 96 episodes × 40 steps from the scripted policy (src/models/world-model/env.ts), 12-step subsequences',
  script: 'scripts/train/rssm.ts',
  metrics,
  notes: `Adam, batch 16, ${total} steps; lr 3e-3, then 1e-3 from 60 % and 3e-4 from 85 % of the steps; gradient-norm clip 100. Open-loop errors are per-pixel MSE at the 5th and 15th predicted step after 5 context frames, averaged over the 8 held-out episodes.`,
});
