/**
 * Trains Dreamer (world model, actor and critic; default configuration) on the
 * point-mass trajectory dataset and writes public/weights/dreamer-point-mass.{json,bin}.
 *
 * Usage: npx tsx scripts/train/dreamer.ts [steps=5000]
 */
import dreamer from '../../src/models/world-model/dreamer';
import { Session } from '../../src/runtime/session';
import { saveCheckpoint, log } from './common';

const noop = { shouldStop: () => false, progress: () => {} };
const total = Number(process.argv[2] ?? 5000);
const session = new Session(dreamer, { seed: 1 });
session.setOptimizer({ kind: 'adam', lr: 3e-3 }, 16);
const t0 = Date.now();
for (let s = 1; s <= total; s++) {
  if (s === Math.floor(total * 0.6)) session.setOptimizer({ lr: 1e-3 });
  if (s === Math.floor(total * 0.85)) session.setOptimizer({ lr: 3e-4 });
  const r = session.trainStep() as { losses: Record<string, number> };
  const l = r.losses;
  log(
    s,
    total,
    `world ${l.world.toFixed(3)} actor ${l.actor.toFixed(3)} critic ${l.critic.toFixed(3)} (${((Date.now() - t0) / 1000).toFixed(0)} s)`,
  );
}

const val = session.evaluate()!;
let actor = 0;
let scripted = 0;
for (let e = 1; e <= 8; e++) {
  const r = session.runProcedure('control', { episode: e }, noop);
  actor += r.scalars!.actorReturn / 8;
  scripted += r.scalars!.scriptedReturn / 8;
}
const metrics = {
  validationLoss: val.loss,
  ...val.metrics,
  actorReturn: actor,
  scriptedPolicyReturn: scripted,
};
console.log(metrics);
saveCheckpoint(session, 'dreamer-point-mass', {
  dataset:
    'Point-mass reaching: 96 episodes × 40 steps from the scripted policy (src/models/world-model/env.ts), 12-step subsequences',
  script: 'scripts/train/dreamer.ts',
  metrics,
  notes: `Adam (one per phase), batch 16, ${total} steps of world model, actor and critic; lr 3e-3, then 1e-3 from 60 % and 3e-4 from 85 %; gradient-norm clip 100. Returns are Σ_{t=1..40} r_t in the simulator from the start states of the 8 held-out episodes (actor: mode actions, closed loop).`,
});
