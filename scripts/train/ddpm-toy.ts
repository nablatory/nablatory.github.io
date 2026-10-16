/**
 * Trains the two-dimensional DDPM noise predictor (default configuration,
 * two moons, T = 100, linear schedule) and writes
 * public/weights/ddpm-toy-moons.{json,bin}. The DDPM and DDIM pages both load
 * this file; DDIM changes only the sampler.
 *
 * Usage: npx tsx scripts/train/ddpm-toy.ts [steps=8000]
 */
import ddpm from '../../src/models/diffusion/ddpm';
import { Session } from '../../src/runtime/session';
import { Graph, Rng } from '../../src/engine';
import { schedule, ddimTimesteps } from '../../src/models/diffusion/schedule';
import {
  ancestralSample,
  ddimSample,
  gaussian,
  mmd2,
  predictEps,
  toyData,
  type ToyConfig,
} from '../../src/models/diffusion/toy';
import { saveCheckpoint, log } from './common';

const total = Number(process.argv[2] ?? 8000);
const session = new Session(ddpm, { seed: 1 });
session.setOptimizer({ kind: 'adam', lr: 2e-3 }, 256);
const t0 = Date.now();
for (let s = 1; s <= total; s++) {
  if (s === Math.floor(total * 0.6)) session.setOptimizer({ lr: 1e-3 });
  if (s === Math.floor(total * 0.85)) session.setOptimizer({ lr: 3e-4 });
  const r = session.trainStep() as { losses: Record<string, number> };
  log(s, total, `loss ${r.losses.main.toFixed(4)} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}

const c = session.config as unknown as ToyConfig;
const sched = schedule(c.schedule, c.T);
const data = toyData(c, 'show').x;
const eps = (x: Parameters<typeof predictEps>[0], t: number) => predictEps(x, t, c);
const g = new Graph({ params: session.params, record: false, grad: false });
const metrics = g.run(() => {
  const xT = gaussian([400, 2], new Rng(100));
  const anc = ancestralSample(eps, xT, sched, 'beta', new Rng(101)).x;
  const det = ddimSample(eps, xT, sched, ddimTimesteps(c.T, 20), 0, new Rng(102)).x;
  return {
    validationLoss: session.evaluate()!.loss,
    mmd2AncestralBeta: mmd2(anc.data, data, 2),
    mmd2Ddim20Eta0: mmd2(det.data, data, 2),
    mmd2Noise: mmd2(xT.data, data, 2),
    mmd2HeldOutData: mmd2(toyData(c, 'val').x, data, 2),
  };
});
console.log(metrics);

// Store only the configuration shared by the DDPM and DDIM pages.
const shared: Record<string, string | number | boolean> = {};
for (const k of ['dataset', 'noise', 'T', 'schedule', 'width', 'depth', 'tdim'] as const)
  shared[k] = c[k];
session.config = shared;
saveCheckpoint(session, 'ddpm-toy-moons', {
  dataset: `Two moons (planar points × 1.5, noise ${c.noise}), 2,000 training points`,
  script: 'scripts/train/ddpm-toy.ts',
  metrics,
  notes: `Adam, batch 256, ${total} steps; lr 2e-3, then 1e-3 from 60 % and 3e-4 from 85 % of the steps. MMD² uses a Gaussian kernel of bandwidth 0.3 against the 400 display points.`,
});
