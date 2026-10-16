/**
 * Trains TD-MPC (default configuration) on the point-mass trajectory dataset
 * and writes public/weights/td-mpc-point-mass.{json,bin}.
 *
 * Usage: npx tsx scripts/train/td-mpc.ts [steps=4000]
 */
import tdmpc from '../../src/models/world-model/tdmpc';
import { Session } from '../../src/runtime/session';
import { saveCheckpoint, log } from './common';

const noop = { shouldStop: () => false, progress: () => {} };
const total = Number(process.argv[2] ?? 4000);
const session = new Session(tdmpc, { seed: 1 });
session.setOptimizer({ kind: 'adam', lr: 3e-3 }, 64);
const t0 = Date.now();
for (let s = 1; s <= total; s++) {
  if (s === Math.floor(total * 0.6)) session.setOptimizer({ lr: 1e-3 });
  if (s === Math.floor(total * 0.85)) session.setOptimizer({ lr: 3e-4 });
  const r = session.trainStep() as { losses: Record<string, number> };
  log(
    s,
    total,
    `model ${r.losses.model.toFixed(4)} policy ${r.losses.policy.toFixed(3)} (${((Date.now() - t0) / 1000).toFixed(0)} s)`,
  );
}

const val = session.evaluate()!;
let mpc = 0;
let scripted = 0;
for (let e = 1; e <= 8; e++) {
  const r = session.runProcedure('control', { episode: e }, noop);
  mpc += r.scalars!.mpcReturn / 8;
  scripted += r.scalars!.scriptedReturn / 8;
}
const metrics = {
  validationLoss: val.loss,
  ...val.metrics,
  mpcReturn: mpc,
  scriptedPolicyReturn: scripted,
};
console.log(metrics);
saveCheckpoint(session, 'td-mpc-point-mass', {
  dataset:
    'Point-mass reaching: 96 episodes × 40 steps from the scripted policy (src/models/world-model/env.ts), 6-step windows',
  script: 'scripts/train/td-mpc.ts',
  metrics,
  notes: `Adam (one per phase), batch 64, ${total} steps; lr 3e-3, then 1e-3 from 60 % and 3e-4 from 85 %; gradient-norm clip 10. Returns are Σ_{t=1..40} r_t of closed-loop MPPI (64 + 4 candidates, 8 elites, 6 iterations) from the start states of the 8 held-out episodes.`,
});
