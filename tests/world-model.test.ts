import { describe, expect, it } from 'vitest';
import { Graph, Rng, Tensor, gradCheck, leaf, withDType, type Var } from '../src/engine';
import rssm from '../src/models/world-model/rssm';
import dreamer, {
  actorLoss,
  criticLoss,
  isActor,
  isCritic,
  lambdaReturns,
} from '../src/models/world-model/dreamer';
import tdmpc, {
  computeTargets,
  isPolicy,
  modelLoss,
  policyLoss,
} from '../src/models/world-model/tdmpc';
import { klDiag, worldModelLoss } from '../src/models/world-model/rssm-core';
import { mppi } from '../src/models/world-model/mppi';
import { ENV, envReward, envStep, episodes, render } from '../src/models/world-model/env';
import type { Batch, Config, ModelDef } from '../src/models/types';
import { Session } from '../src/runtime/session';
import { fetchCheckpoint } from '../src/runtime/checkpoint';
import { fsLoader } from './helpers';

const noop = { shouldStop: () => false, progress: () => {} };

/** Small world model for gradient checks. */
const TINY = { deter: 4, stoch: 2, hidden: 8, embed: 4, seqLen: 3, freeNats: 0 };

/**
 * Float64 central differences with ε = 1e-5. These losses sum over 256 pixels,
 * so with ε = 1e-6 the round-off in L(θ ± ε) alone (≈ 1e-16 · |L| / ε) is
 * ~1e-9, comparable to the smallest gradients; ε = 1e-5 keeps both round-off
 * and the O(ε²) truncation error far below the tolerance.
 */
function check(
  model: ModelDef,
  config: Config,
  loss: (b: Batch, c: Config) => Var,
  names: (n: string) => boolean,
  prepare?: (b: Batch, c: Config, s: Session) => void,
) {
  const s = new Session(model, { config });
  const batch = s.dataset.train(new Rng(5), 2, s.config);
  prepare?.(batch, s.config, s);
  return gradCheck(() => loss(batch, s.config), s.params, {
    names: s.params.names().filter((n) => s.params.isTrainable(n) && names(n)),
    perParam: 3,
    eps: 1e-5,
  });
}

describe('point-mass environment', () => {
  it('follows the documented semi-implicit Euler dynamics, reward and rendering', () => {
    const s = [0.1, -0.2, 0.3, 0.5];
    const next = envStep(s, [0.5, -1]);
    const vx = 0.3 + 0.1 * (2 * 0.5 - 1 * 0.3);
    const vy = 0.5 + 0.1 * (2 * -1 - 1 * 0.5);
    expect(next[2]).toBeCloseTo(vx, 12);
    expect(next[3]).toBeCloseTo(vy, 12);
    expect(next[0]).toBeCloseTo(0.1 + 0.1 * vx, 12);
    expect(next[1]).toBeCloseTo(-0.2 + 0.1 * vy, 12);
    // Wall: clamped position, zero velocity.
    const w = envStep([0.99, 0, 1, 0], [1, 0]);
    expect(w[0]).toBe(1);
    expect(w[2]).toBe(0);
    expect(envReward([ENV.goal[0], ENV.goal[1], 0, 0])).toBe(1);
    const img = new Float32Array(256);
    render([0, 0, 0, 0], img);
    expect(Math.max(...img)).toBeGreaterThan(0.8);
    const { train, val } = episodes();
    expect(train.length).toBe(96);
    expect(val[0].obs.length).toBe(41 * 256);
  });
});

describe('RSSM (PlaNet)', () => {
  it('backpropagation matches finite differences (float64), MLP and convolutional vision', () => {
    for (const vision of ['mlp', 'conv']) {
      const res = check(
        rssm,
        { ...TINY, vision, channels: 2 },
        (b, c) => worldModelLoss(b, c as never).loss,
        () => true,
      );
      expect(res.maxRelError).toBeLessThan(1e-5);
    }
  });

  it('KL of diagonal Gaussians is non-negative, zero for equal arguments and equals the closed form', () => {
    const rng = new Rng(3);
    const t = (f: () => number) => {
      const x = Tensor.zeros([5, 4]);
      for (let i = 0; i < x.size; i++) x.data[i] = f();
      return x;
    };
    const muQ = t(() => rng.normal());
    const sdQ = t(() => 0.1 + rng.uniform() * 2);
    const muP = t(() => rng.normal());
    const sdP = t(() => 0.1 + rng.uniform() * 2);
    const g = new Graph({ record: false, grad: false });
    const [kl, self] = g.run(() => [
      klDiag({ mean: leaf(muQ), std: leaf(sdQ) }, { mean: leaf(muP), std: leaf(sdP) }).value,
      klDiag({ mean: leaf(muQ), std: leaf(sdQ) }, { mean: leaf(muQ), std: leaf(sdQ) }).value,
    ]);
    expect(kl.shape).toEqual([5]);
    for (let r = 0; r < 5; r++) {
      let ref = 0;
      for (let d = 0; d < 4; d++) {
        const k = r * 4 + d;
        const [mq, sq, mp, sp] = [muQ.data[k], sdQ.data[k], muP.data[k], sdP.data[k]];
        ref += Math.log(sp / sq) + (sq * sq + (mq - mp) ** 2) / (2 * sp * sp) - 0.5;
      }
      expect(kl.data[r]).toBeGreaterThanOrEqual(0);
      expect(kl.data[r]).toBeCloseTo(ref, 4);
      expect(Math.abs(self.data[r])).toBeLessThan(1e-6);
    }
  });

  it('reparameterised sampling is reproducible under the seeded graph generator', () => {
    const s = new Session(rssm);
    const batch = s.dataset.train(new Rng(1), 2, s.config);
    const loss = (seed: number) =>
      new Graph({ params: s.params, record: false, grad: false, rng: new Rng(seed) }).run(
        () => worldModelLoss(batch, s.config as never).loss.value.data[0],
      );
    expect(loss(4)).toBe(loss(4));
    expect(loss(4)).not.toBe(loss(5));
  });

  it('training lowers the held-out loss; the open-loop procedure returns aligned tensors', () => {
    const s = new Session(rssm, { seed: 2 });
    const before = s.evaluate()!;
    for (let i = 0; i < 60; i++) s.trainStep();
    const after = s.evaluate()!;
    expect(after.loss).toBeLessThan(before.loss * 0.75);
    expect(after.metrics.kl).toBeGreaterThanOrEqual(0);
    const r = s.runProcedure('openloop', { episode: 2, context: 4, horizon: 6 }, noop);
    expect(r.tensors.predicted.shape).toEqual([10, 1, 16, 16]);
    expect(r.tensors.observed.shape).toEqual([10, 1, 16, 16]);
    expect(r.tensors.reward.shape).toEqual([10]);
    expect(Array.from(r.tensors.observed.data.slice(0, 256))).toEqual(
      Array.from(episodes().val[1].obs.slice(0, 256)),
    );
  });
});

describe('Dreamer', () => {
  const cfg = { ...TINY, horizon: 4, starts: 4 };

  it('backpropagation matches finite differences in all three phases (float64)', () => {
    const wm = check(
      dreamer,
      cfg,
      (b, c) => worldModelLoss(b, c as never).loss,
      (n) => !isActor(n) && !isCritic(n),
    );
    expect(wm.maxRelError).toBeLessThan(1e-5);
    // Actor: analytic gradients through the imagined dynamics, reward head and critic.
    const actor = check(dreamer, cfg, (b, c) => actorLoss(b, c as never).loss, isActor);
    expect(actor.maxRelError).toBeLessThan(1e-5);
    // Critic: targets are stop-gradient values, taken from one actor pass on the same batch.
    const critic = check(
      dreamer,
      cfg,
      (b, c) => criticLoss(b, c as never).loss,
      isCritic,
      (b, c, s) => {
        new Graph({ params: s.params, record: false, grad: false, rng: new Rng(1) }).run(() =>
          actorLoss(b, c as never),
        );
      },
    );
    expect(critic.maxRelError).toBeLessThan(1e-5);
  });

  it('λ-returns: λ = 0 is one-step TD, λ = 1 is the discounted reward sum bootstrapped at H', () => {
    const H = 5;
    const r = Tensor.zeros([H, 1, 1]);
    const v = Tensor.zeros([H, 1, 1]);
    for (let t = 0; t < H; t++) {
      r.data[t] = t + 1;
      v.data[t] = 10 * (t + 1);
    }
    const g = new Graph({ record: false, grad: false });
    const [td, mc] = g.run(() => [
      lambdaReturns(leaf(r), leaf(v), 0.9, 0).value,
      lambdaReturns(leaf(r), leaf(v), 0.9, 1).value,
    ]);
    for (let t = 0; t < H - 1; t++) {
      expect(td.data[t]).toBeCloseTo(r.data[t] + 0.9 * v.data[t + 1], 5);
      let ret = 0;
      for (let k = t; k < H - 1; k++) ret += 0.9 ** (k - t) * r.data[k];
      ret += 0.9 ** (H - 1 - t) * v.data[H - 1];
      expect(mc.data[t]).toBeCloseTo(ret, 4);
    }
  });

  it('the actor phase leaves world-model and critic parameters unchanged', () => {
    const actorOnly = {
      ...dreamer,
      training: { ...dreamer.training, phases: [dreamer.training.phases![1]] },
    };
    const s = new Session(actorOnly, { config: cfg });
    const before = new Map(s.params.entries().map(([k, v]) => [k, Array.from(v.data)]));
    s.trainStep();
    let actorChanged = false;
    for (const [k, v] of s.params.entries()) {
      const same = Array.from(v.data).every((x, i) => x === before.get(k)![i]);
      if (isActor(k)) actorChanged ||= !same;
      else expect(same, k).toBe(true);
    }
    expect(actorChanged).toBe(true);
  });

  it('three-phase training lowers the world-model loss and runs its procedures', () => {
    const s = new Session(dreamer, { seed: 2, config: { starts: 16 } });
    const before = s.evaluate()!;
    for (let i = 0; i < 30; i++) {
      const r = s.trainStep() as { losses: Record<string, number> };
      expect(Object.keys(r.losses)).toEqual(['world', 'actor', 'critic']);
    }
    expect(s.evaluate()!.loss).toBeLessThan(before.loss);
    const im = s.runProcedure('imagine', {}, noop);
    expect(im.tensors.predicted.shape).toEqual([10, 1, 16, 16]);
    expect(im.tensors.actions.shape).toEqual([10, 2]);
    for (const a of im.tensors.actions.data) expect(Math.abs(a)).toBeLessThanOrEqual(1);
    const ctl = s.runProcedure('control', { episode: 1 }, noop);
    expect(ctl.tensors.reward.shape).toEqual([41]);
  });
});

describe('TD-MPC', () => {
  const cfg = { latent: 3, hidden: 6, horizon: 3 };

  it('backpropagation matches finite differences (float64), with targets held fixed', () => {
    // The TD and consistency targets are stop-gradient values; finite differences
    // are only meaningful with them held constant while θ is perturbed.
    let targets: ReturnType<typeof computeTargets> | undefined;
    const model = check(
      tdmpc,
      cfg,
      (b, c) => modelLoss(b, c as never, targets).loss,
      (n) => !isPolicy(n),
      (b, c, s) => {
        targets = withDType('float64', () =>
          new Graph({ params: s.params.clone('float64'), record: false, grad: false }).run(() =>
            computeTargets(b, c as never),
          ),
        );
      },
    );
    expect(model.maxRelError).toBeLessThan(1e-5);
    const pol = check(tdmpc, cfg, (b, c) => policyLoss(b, c as never).loss, isPolicy);
    expect(pol.maxRelError).toBeLessThan(1e-5);
  });

  it('MPPI improves the planned return over its iterations on a known reward', () => {
    const H = 4;
    const target = [0.6, -0.3, 0.2, 0.9, -0.7, 0.4, 0.1, -0.5];
    const evaluate = (acts: Float64Array, n: number) => {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++)
        for (let k = 0; k < H * 2; k++) out[i] -= (acts[i * H * 2 + k] - target[k]) ** 2;
      return out;
    };
    const res = mppi(
      evaluate,
      {
        horizon: H,
        actionDim: 2,
        samples: 64,
        elites: 8,
        iterations: 8,
        temperature: 2,
        minStd: 0.02,
        maxStd: 1,
      },
      new Rng(1),
    );
    const m = res.meanReturn;
    expect(m.length).toBe(9);
    expect(m[8]).toBeGreaterThan(m[0]);
    expect(m[8]).toBeGreaterThan(-0.05);
    const elites = res.iterations.map((it) => it.eliteMean);
    expect(elites[elites.length - 1]).toBeGreaterThan(elites[0]);
  });

  it('training lowers the loss, and MPPI raises the model’s planned return', () => {
    const s = new Session(tdmpc, { seed: 1 });
    const before = s.evaluate()!;
    for (let i = 0; i < 150; i++) s.trainStep();
    expect(s.evaluate()!.loss).toBeLessThan(before.loss * 0.5);
    const r = s.runProcedure('plan', {}, noop);
    expect(r.scalars!.plannedReturnLast).toBeGreaterThan(r.scalars!.plannedReturnFirst);
    expect(r.tensors.returns.shape).toEqual([6, 68]);
    expect(r.tensors.meanAction.shape).toEqual([7, 2]);
  });
});

describe('shipped checkpoints', () => {
  async function loaded(m: ModelDef) {
    const s = new Session(m);
    const ck = m.checkpoints![0];
    const { file, data } = await fetchCheckpoint(fsLoader, ck.path);
    s.loadCheckpoint(ck.id, file.params, data);
    return s;
  }

  it('RSSM: open-loop prediction stays close to the simulator for 5 steps', async () => {
    const s = await loaded(rssm);
    const r = s.runProcedure('openloop', { episode: 1, context: 5, horizon: 15 }, noop);
    // Per-pixel MSE; predicting only the background would give about 0.03.
    expect(r.tensors.pixelError.data[9]).toBeLessThan(0.006);
  });

  it('Dreamer and TD-MPC: the trained controller collects more reward than the scripted data policy', async () => {
    const d = (await loaded(dreamer)).runProcedure('control', { episode: 1 }, noop);
    expect(d.scalars!.actorReturn).toBeGreaterThan(d.scalars!.scriptedReturn);
    const t = (await loaded(tdmpc)).runProcedure('control', { episode: 1, iterations: 4 }, noop);
    expect(t.scalars!.mpcReturn).toBeGreaterThan(t.scalars!.scriptedReturn);
  });
});

describe('world models: speed at the default configuration', () => {
  it.each([rssm, dreamer, tdmpc])('$id: forward < 100 ms, training step < 250 ms', (m) => {
    const s = new Session(m);
    for (let i = 0; i < 2; i++) s.trainStep();
    s.inspect({ kind: 'none' });
    const t0 = performance.now();
    for (let i = 0; i < 3; i++) s.inspect({ kind: 'none' });
    const t1 = performance.now();
    for (let i = 0; i < 3; i++) s.trainStep();
    const t2 = performance.now();
    expect((t1 - t0) / 3).toBeLessThan(100);
    expect((t2 - t1) / 3).toBeLessThan(250);
  });
});

describe('in-browser gradient check (Session.gradCheck)', () => {
  it('passes for RSSM, Dreamer and TD-MPC with their declared settings; TD-MPC holds its targets fixed', () => {
    for (const [model, config] of [
      [rssm, TINY],
      [dreamer, TINY],
      [tdmpc, { latent: 3, hidden: 6, horizon: 3 }],
    ] as const) {
      const s = new Session(model, { config });
      const r = s.gradCheck(undefined, 2);
      expect(r.maxRelError, model.id).toBeLessThan(1e-5);
    }
  });
});
