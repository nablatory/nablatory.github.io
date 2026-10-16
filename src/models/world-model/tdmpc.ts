import {
  ParamBuilder,
  Tensor,
  add,
  concat,
  declareLinear,
  defaultOptimizer,
  dense,
  elu,
  maximum,
  mean,
  mulScalar,
  neg,
  noGrad,
  reshape,
  scope,
  select,
  slice,
  square,
  stack,
  sub,
  sum,
  tanh,
  type Rng,
  type Var,
} from '../../engine';
import { defineModel, type Batch, type Procedure } from '../types';
import { constant, input } from '../util';
import {
  ENV,
  ENV_EXTENT,
  ENV_MATH,
  envReward,
  envStep,
  episodes,
  pathScatter,
  pointMassDataset,
} from './env';
import { mppi, type MppiOptions } from './mppi';

export type TdmpcConfig = {
  [key: string]: number | string | boolean;
  latent: number;
  hidden: number;
  horizon: number;
  discount: number;
  rho: number;
  consistencyCoef: number;
  rewardCoef: number;
  valueCoef: number;
};

const A = ENV.actionDim;
const S = ENV.stateDim;

export const isPolicy = (n: string) => n.startsWith('pi.');

function mlp2(x: Var, name: string): Var {
  return dense(elu(dense(x, `${name}.fc1`)), `${name}.fc2`);
}

export function encodeState(s: Var): Var {
  return scope('encoder', () => mlp2(s, 'encoder'));
}

/** z' = d(z, a), r̂ = R(z, a), Q_i(z, a) all read [z ‖ a]. */
export function dynamics(za: Var): Var {
  return scope('dynamics', () => mlp2(za, 'dynamics'));
}

export function rewardFn(za: Var): Var {
  return scope('reward', () => mlp2(za, 'reward'));
}

export function qFns(za: Var): [Var, Var] {
  return scope('q', () => [mlp2(za, 'q1'), mlp2(za, 'q2')]);
}

/** min(Q₁, Q₂) = −max(−Q₁, −Q₂) */
export function minQ(za: Var): Var {
  const [q1, q2] = qFns(za);
  return neg(maximum(neg(q1), neg(q2)));
}

export function policy(z: Var): Var {
  return scope('pi', () => tanh(mlp2(z, 'pi')));
}

function za(z: Var, a: Var): Var {
  return concat([z, a], -1);
}

/** Per-row squared error averaged over the last axis, then over rows. */
function mseRows(a: Var, b: Var): Var {
  return mean(square(sub(a, b)));
}

export interface TdTargets {
  /** sg(h(s_{t+1})) for t = 0 … H−1, each [B, latent]. */
  zNext: Tensor[];
  /** y_t = r_t + γ min_i Q_i(z⁺_{t+1}, π(z⁺_{t+1})), each [B, 1]. */
  y: Tensor[];
}

/** Consistency and TD targets, computed without gradient from the current networks. */
export function computeTargets(batch: Batch, c: TdmpcConfig): TdTargets {
  return noGrad(() => {
    const st = constant(batch.inputs.state, 'state');
    const rew = constant(batch.inputs.reward, 'reward');
    const out: TdTargets = { zNext: [], y: [] };
    for (let t = 0; t < st.shape[1] - 1; t++) {
      const zn = encodeState(select(st, 1, t + 1));
      const r = reshape(select(rew, 1, t + 1), [-1, 1]);
      out.zNext.push(zn.value);
      out.y.push(add(r, mulScalar(minQ(za(zn, policy(zn))), c.discount)).value);
    }
    return out;
  });
}

/**
 * TD-MPC model objective over a horizon of H transitions (Hansen et al. 2022, eq. 7):
 *   L = (1/H) Σ_t ρ^t [ c₁ ‖d(ẑ_t, a_t) − sg(h(s_{t+1}))‖² + c₂ (R(ẑ_t, a_t) − r_t)² + c₃ Σ_i (Q_i(ẑ_t, a_t) − y_t)² ],
 *   y_t = r_t + γ min_i Q_i(z⁺, π(z⁺)),  z⁺ = sg(h(s_{t+1})),  ẑ_0 = h(s_0).
 * Squared errors are averaged over the batch and the latent dimensions.
 * `targets` may be supplied to hold them fixed (used by the gradient check).
 */
export function modelLoss(batch: Batch, c: TdmpcConfig, targets?: TdTargets) {
  const st = input('state', batch.inputs.state, false);
  const act = input('action', batch.inputs.action, false);
  const rew = input('reward', batch.inputs.reward, false);
  const T = st.shape[1];
  const H = T - 1;
  const tg = scope('targets', () => targets ?? computeTargets(batch, c));
  let z = scope('encoder', () => encodeState(select(st, 1, 0)));
  const zs: Var[] = [z];
  const terms = { consistency: [] as Var[], reward: [] as Var[], value: [] as Var[] };
  const rPred: Var[] = [];
  const qPred: Var[] = [];
  for (let t = 0; t < H; t++) {
    const zt = z;
    const inp = scope('dynamics', () => za(zt, select(act, 1, t)));
    const [q1, q2] = qFns(inp);
    const rh = rewardFn(inp);
    z = dynamics(inp);
    zs.push(z);
    const zt1 = z;
    const w = Math.pow(c.rho, t);
    scope('loss', () => {
      const r = reshape(select(rew, 1, t + 1), [-1, 1]);
      const zNext = constant(tg.zNext[t], 'sg(z⁺)');
      const y = constant(tg.y[t], 'sg(y)');
      terms.consistency.push(mulScalar(mseRows(zt1, zNext), w));
      terms.reward.push(mulScalar(mseRows(rh, r), w));
      terms.value.push(mulScalar(add(mseRows(q1, y), mseRows(q2, y)), w));
    });
    rPred.push(rh);
    qPred.push(q1);
  }
  return scope('loss', () => {
    const total = (xs: Var[]) => sum(stack(xs, 0));
    const cons = total(terms.consistency);
    const rl = total(terms.reward);
    const vl = total(terms.value);
    const loss = mulScalar(
      add(
        add(mulScalar(cons, c.consistencyCoef), mulScalar(rl, c.rewardCoef)),
        mulScalar(vl, c.valueCoef),
      ),
      1 / H,
    );
    return {
      loss,
      zs,
      rPred,
      qPred,
      metrics: {
        consistency: cons.value.data[0],
        reward: rl.value.data[0],
        value: vl.value.data[0],
      },
    };
  });
}

/** Policy objective: −(1/(H+1)) Σ_t ρ^t min_i Q_i(sg(ẑ_t), π(sg(ẑ_t))), Q frozen. */
export function policyLoss(batch: Batch, c: TdmpcConfig) {
  const zs = scope('rollout', () =>
    noGrad(() => {
      const st = constant(batch.inputs.state, 'state');
      const act = constant(batch.inputs.action, 'action');
      let z = encodeState(select(st, 1, 0));
      const out = [z];
      for (let t = 0; t < st.shape[1] - 1; t++) {
        z = dynamics(za(z, select(act, 1, t)));
        out.push(z);
      }
      return out.map((v) => constant(v.value, 'sg(ẑ)'));
    }),
  );
  const terms = zs.map((z, t) => mulScalar(mean(minQ(za(z, policy(z)))), Math.pow(c.rho, t)));
  const loss = scope('loss', () => mulScalar(sum(stack(terms, 0)), -1 / zs.length));
  return { loss, metrics: { policyQ: -loss.value.data[0] } };
}

/**
 * Returns of n action sequences [n · H · A] from latent state z₀ [1, L]:
 *   G = Σ_{t<H} γ^t R(z_t, a_t) + γ^H min_i Q_i(z_H, π(z_H)),  z_{t+1} = d(z_t, a_t).
 */
export function latentReturns(
  z0: Var,
  actions: Float64Array,
  n: number,
  c: TdmpcConfig,
): Float64Array {
  const H = c.horizon;
  let z = add(constant(Tensor.zeros([n, z0.shape[1]]), 'broadcast'), z0);
  let G: Var | null = null;
  let disc = 1;
  for (let t = 0; t < H; t++) {
    const a = Tensor.zeros([n, A]);
    for (let i = 0; i < n; i++)
      for (let k = 0; k < A; k++) a.data[i * A + k] = actions[(i * H + t) * A + k];
    const inp = za(z, constant(a, 'a_t'));
    const r = mulScalar(rewardFn(inp), disc);
    G = G ? add(G, r) : r;
    z = dynamics(inp);
    disc *= c.discount;
  }
  G = add(G!, mulScalar(minQ(za(z, policy(z))), disc));
  return Float64Array.from(G.value.data);
}

/** Action sequences [H · A] from rolling the policy through the latent dynamics (plus optional noise). */
function policySequence(z0: Var, c: TdmpcConfig, rng: Rng, noise: number): Float64Array {
  const out = new Float64Array(c.horizon * A);
  let z = z0;
  for (let t = 0; t < c.horizon; t++) {
    const a = policy(z).value;
    const an = Tensor.zeros([1, A]);
    for (let k = 0; k < A; k++) {
      const v = Math.max(-1, Math.min(1, a.data[k] + noise * rng.normal()));
      an.data[k] = v;
      out[t * A + k] = v;
    }
    z = dynamics(za(z, constant(an, 'a_t')));
  }
  return out;
}

function mppiOptions(c: TdmpcConfig, args: Record<string, unknown>): MppiOptions {
  return {
    horizon: c.horizon,
    actionDim: A,
    samples: Number(args.samples),
    elites: Number(args.elites),
    iterations: Number(args.iterations),
    temperature: Number(args.temperature),
    minStd: 0.05,
    maxStd: 2,
  };
}

function plan(
  state: number[],
  c: TdmpcConfig,
  args: Record<string, unknown>,
  rng: Rng,
  initMean?: Float64Array,
) {
  const z0 = encodeState(constant(new Tensor([1, S], state), 's₀'));
  const nPi = Number(args.policySamples);
  return mppi(
    (acts, n) => latentReturns(z0, acts, n, c),
    { ...mppiOptions(c, args), initMean },
    rng,
    () => Array.from({ length: nPi }, (_, i) => policySequence(z0, c, rng, i === 0 ? 0 : 0.2)),
  );
}

const mppiArgs = [
  {
    key: 'episode',
    label: 'Start state of held-out episode',
    kind: 'int' as const,
    min: 1,
    max: 8,
    affects: 'data' as const,
  },
  {
    key: 'samples',
    label: 'Gaussian samples N',
    kind: 'int' as const,
    min: 8,
    max: 256,
    affects: 'forward' as const,
  },
  {
    key: 'policySamples',
    label: 'Policy samples',
    kind: 'int' as const,
    min: 0,
    max: 16,
    affects: 'forward' as const,
  },
  {
    key: 'elites',
    label: 'Elites k',
    kind: 'int' as const,
    min: 2,
    max: 64,
    affects: 'forward' as const,
  },
  {
    key: 'iterations',
    label: 'Iterations J',
    kind: 'int' as const,
    min: 1,
    max: 12,
    affects: 'forward' as const,
  },
  {
    key: 'temperature',
    label: 'Temperature τ',
    kind: 'float' as const,
    min: 0.1,
    max: 10,
    step: 0.1,
    affects: 'forward' as const,
  },
];
const mppiDefaults = {
  episode: 1,
  samples: 64,
  policySamples: 4,
  elites: 8,
  iterations: 6,
  temperature: 0.5,
};

function startState(e: number): number[] {
  const eps = episodes().val;
  const ep = eps[Math.max(0, Math.min(eps.length - 1, Math.round(e) - 1))];
  return Array.from(ep.states.subarray(0, 4));
}

const planProcedure: Procedure<TdmpcConfig> = {
  id: 'plan',
  label: 'MPPI plan',
  description:
    'Plans one action sequence from the start of a held-out episode: samples sequences from a Gaussian, scores them in latent space with the learned reward and Q, and refits the Gaussian to the elites. The plan is then executed open loop in the simulator.',
  args: mppiArgs,
  defaults: mppiDefaults,
  run({ config: c, args, rng }) {
    const s0 = startState(Number(args.episode));
    const res = plan(s0, c, args, rng);
    const J = res.iterations.length;
    const n = res.iterations[0].returns.length;
    const returns = new Float32Array(J * n);
    res.iterations.forEach((it, j) => {
      const sorted = Array.from(it.returns).sort((a, b) => b - a);
      returns.set(sorted, j * n);
    });
    const meanAction = new Float32Array((J + 1) * A);
    res.iterations.forEach((it, j) => meanAction.set([it.mean[0], it.mean[1]], (j + 1) * A));
    const H = c.horizon;
    let s = s0;
    const states = [s];
    const rTrue = new Float32Array(H);
    for (let t = 0; t < H; t++) {
      s = envStep(s, [res.mean[t * A], res.mean[t * A + 1]]);
      states.push(s);
      rTrue[t] = envReward(s);
    }
    // Model's reward predictions along the final plan.
    const z0 = encodeState(constant(new Tensor([1, S], s0), 's₀'));
    let z = z0;
    const rModel = new Float32Array(H);
    for (let t = 0; t < H; t++) {
      const inp = za(
        z,
        constant(new Tensor([1, A], [res.mean[t * A], res.mean[t * A + 1]]), 'a_t'),
      );
      rModel[t] = rewardFn(inp).value.data[0];
      z = dynamics(inp);
    }
    const { points, colors } = pathScatter(states);
    return {
      tensors: {
        returns: new Tensor([J, n], returns),
        eliteMean: new Tensor([J], Float32Array.from(res.iterations.map((it) => it.eliteMean))),
        meanReturn: new Tensor([J + 1], Float32Array.from(res.meanReturn)),
        meanAction: new Tensor([J + 1, A], meanAction),
        plan: new Tensor([H, A], Float32Array.from(res.mean)),
        path: points,
        pathColor: colors,
        rewardModel: new Tensor([H], rModel),
        rewardTrue: new Tensor([H], rTrue),
      },
      scalars: { plannedReturnFirst: res.meanReturn[0], plannedReturnLast: res.meanReturn[J] },
      meta: {
        caption: `Planned return of the mean sequence: ${res.meanReturn[0].toFixed(3)} before, ${res.meanReturn[J].toFixed(3)} after ${J} iterations (model estimate).`,
      },
    };
  },
};

const controlProcedure: Procedure<TdmpcConfig> = {
  id: 'control',
  label: 'Closed-loop control',
  description:
    'Model predictive control in the simulator: at every step, plan with MPPI (warm-started from the previous plan shifted by one step), apply the first action of the mean, observe the new state, repeat. Compares the rewards with the scripted policy from the same start.',
  args: mppiArgs,
  defaults: mppiDefaults,
  run({ config: c, args, rng, shouldStop, progress }) {
    const e = Math.max(
      0,
      Math.min(episodes().val.length - 1, Math.round(Number(args.episode)) - 1),
    );
    const ep = episodes().val[e];
    const L = ENV.episodeLength;
    let s = startState(Number(args.episode));
    const states = [s];
    const rewards = new Float32Array(L + 1);
    rewards[0] = envReward(s);
    const actions = new Float32Array(L * A);
    let prev: Float64Array | undefined;
    for (let t = 0; t < L; t++) {
      if (shouldStop()) break;
      const res = plan(s, c, args, rng, prev);
      const a = [res.mean[0], res.mean[1]];
      actions.set(a, t * A);
      prev = new Float64Array(res.mean.length);
      prev.set(res.mean.subarray(A));
      s = envStep(s, a);
      states.push(s);
      rewards[t + 1] = envReward(s);
      progress((t + 1) / L);
    }
    const { points, colors } = pathScatter(states);
    let mpc = 0;
    let scripted = 0;
    for (let t = 1; t <= L; t++) {
      mpc += rewards[t];
      scripted += ep.rewards[t];
    }
    return {
      tensors: {
        path: points,
        pathColor: colors,
        reward: new Tensor([L + 1], rewards),
        scriptedReward: new Tensor([L + 1], ep.rewards.slice()),
        actions: new Tensor([L, A], actions),
      },
      scalars: { mpcReturn: mpc, scriptedReturn: scripted },
      meta: {
        caption: `Return Σr: MPPI with the learned model ${mpc.toFixed(2)}, scripted policy ${scripted.toFixed(2)}.`,
      },
    };
  },
};

const tdmpc = defineModel<TdmpcConfig>({
  id: 'td-mpc',
  family: 'world-model',
  name: 'TD-MPC',
  summary:
    'A task-oriented latent dynamics model without a decoder: an encoder, latent dynamics, reward and twin Q heads trained by latent consistency, reward and temporal-difference losses, used for planning with MPPI over short horizons and a learned value beyond them.',
  signature:
    '\\mathcal L = \\sum_{t} \\rho^{t}\\Big[c_1 (R(\\hat z_t, a_t) - r_t)^2 + c_2 \\sum_i \\big(Q_i(\\hat z_t, a_t) - r_t - \\gamma \\min_j Q_j(z_{t+1}, \\pi(z_{t+1}))\\big)^2 + c_3 \\lVert d(\\hat z_t, a_t) - \\operatorname{sg}(h(s_{t+1}))\\rVert^2\\Big]',
  fidelity: 'reduced',
  scaleNote:
    'Encoder h, latent dynamics d, reward R, twin Q, policy π, the temporally weighted reward + TD + consistency objective (c₁ = 0.5, c₂ = 0.1, c₃ = 2 and temporal weight ρ = 0.5, the paper’s λ) and MPPI planning (6 iterations, temperature 0.5, σ clipped to [0.05, 2], horizon 5) follow TD-MPC. Reduced: 16-dimensional latent and one-hidden-layer MLPs of 32 units instead of 50 and two hidden layers of 512, state observations of a point mass, discount 0.9 instead of 0.99, 64 + 4 candidates and 8 elites instead of 512 + 25 and 64, and no momentum on μ (paper 0.1).',
  references: [
    {
      authors: 'N. Hansen, X. Wang, H. Su',
      title: 'Temporal difference learning for model predictive control',
      year: 2022,
      url: 'https://arxiv.org/abs/2203.04955',
    },
    {
      authors: 'G. Williams, A. Aldrich, E. A. Theodorou',
      title: 'Model predictive path integral control: from theory to parallel computation',
      year: 2017,
      url: 'https://doi.org/10.2514/1.G001921',
    },
  ],
  limitations: [
    'Trained offline on the fixed scripted-policy dataset; there is no replay buffer filled by the planner, no seed-step exploration and no prioritised replay.',
    'TD targets and consistency targets use the current networks with gradients stopped, not the slowly updated (exponential-moving-average) target networks of the paper.',
    'No layer normalisation in the Q networks and no gradient-scale tricks of the reference implementation.',
    'MPPI here executes the first action of the final mean μ^J. Algorithm 2 of the paper samples a ~ N(μ^J, (σ^J)²); the reference implementation draws one elite trajectory with probability proportional to its weight and adds σ noise during training. μ is refitted without momentum.',
    'Policy candidates for MPPI are one noise-free policy rollout plus rollouts with Gaussian noise of standard deviation 0.2 on the actions; the reference implementation adds noise of 0.05 to every policy trajectory and to π in the TD target, whereas the TD target here uses π(z⁺) without noise.',
    'The model is trained on 5-step windows, so long-horizon planning relies on the learned Q beyond them; after a few hundred browser steps Q is still inaccurate and closed-loop control is usually worse than the scripted policy. The shipped checkpoint (4,000 steps) reaches a mean closed-loop return of 37.4 over 40 simulator steps from the held-out start states, against 14.8 for the scripted policy (40 is the maximum).',
  ],
  guide: {
    what: 'The input is the four-number state of a point mass (position and velocity) and the force, in windows of six steps. An encoder maps each state to a 16-dimensional latent vector, and small networks predict the next latent, the reward, and a value Q (the discounted reward expected from here on); there is no decoder, so the latent only has to be useful for predicting reward and value. To act, a planner samples many 5-step force sequences, scores them inside the model, and refits its distribution to the best eight.',
    watch:
      'Opens with the shipped checkpoint, already trained: its predicted reward follows the observed one (0.15–0.18 against 0.15–0.17 in sequence 1). Run “MPPI plan”: the return of the mean plan jumps in the first iteration (episode 1: 4.9 to 6.9) and the elite return then rises slowly over the remaining five (6.8 to 7.2). Press “Start from random weights” to train from scratch: the model loss falls from about 0.15 to 0.025 within 100 steps, but after the default 400 steps the predicted reward is still a nearly flat line well above the observed one (about 0.36 against 0.15–0.17), and “Closed-loop control” collects a return of about 4 instead of the checkpoint’s 37.',
    try: [
      {
        label: 'No consistency loss',
        note: 'Without the latent-consistency term the rolled-out latent drifts from the encoded states: over 200 steps consistency rises from 0.001 to between 0.01 and 0.04. Reward still fits (about 0.003, as when training on with the default), value is about twice the default’s (0.20 against 0.09), and closed-loop control barely suffers (return 35.9 against 37.2).',
        patch: { consistencyCoef: 0 },
        train: 200,
      },
      {
        label: 'Horizon 1',
        note: 'One-step windows: plans are one action plus Q, and the MPPI iterations barely change the plan after the first. Over 150 steps the reward fit loosens (reward 0.0007 → about 0.008), and the closed-loop return over the eight held-out starts falls from 37.4 to 27.4, still above the scripted policy’s 14.8.',
        patch: { horizon: 1 },
        train: 150,
      },
      {
        label: 'Discount 0.99',
        note: 'The true Q can now reach 1/(1 − γ) = 100 instead of 10, but the TD targets, computed from the current Q, chase it upwards: within 200 steps Q passes 600, far above any possible return, value grows from 2.7 to the hundreds and dominates the loss, and closed-loop control collapses (return about 4.5 against 37.4).',
        patch: { discount: 0.99 },
        train: 200,
      },
    ],
    views: [
      'Predicted vs. observed reward (sequence 1)',
      'MPPI: return of the mean plan and of the elites per iteration',
    ],
  },
  configSchema: [
    {
      key: 'latent',
      label: 'Latent dimension',
      kind: 'int',
      min: 2,
      max: 64,
      affects: 'architecture',
    },
    {
      key: 'hidden',
      label: 'MLP hidden units',
      kind: 'int',
      min: 8,
      max: 128,
      affects: 'architecture',
    },
    {
      key: 'horizon',
      label: 'Horizon H (training and planning)',
      kind: 'int',
      min: 1,
      max: 12,
      affects: 'data',
    },
    {
      key: 'discount',
      label: 'Discount γ',
      kind: 'float',
      min: 0.5,
      max: 0.999,
      step: 0.01,
      affects: 'forward',
    },
    {
      key: 'rho',
      label: 'Temporal weight ρ (λ in the paper)',
      kind: 'float',
      min: 0.1,
      max: 1,
      step: 0.05,
      affects: 'forward',
    },
    {
      key: 'consistencyCoef',
      label: 'Consistency coefficient c₃',
      kind: 'float',
      min: 0,
      max: 10,
      step: 0.1,
      affects: 'forward',
    },
    {
      key: 'rewardCoef',
      label: 'Reward coefficient c₁',
      kind: 'float',
      min: 0,
      max: 10,
      step: 0.1,
      affects: 'forward',
    },
    {
      key: 'valueCoef',
      label: 'Value coefficient c₂',
      kind: 'float',
      min: 0,
      max: 10,
      step: 0.05,
      affects: 'forward',
    },
  ],
  defaults: {
    latent: 16,
    hidden: 32,
    horizon: 5,
    discount: 0.9,
    rho: 0.5,
    consistencyCoef: 2,
    rewardCoef: 0.5,
    valueCoef: 0.1,
  },
  init(c, rng) {
    const b = new ParamBuilder(rng);
    declareLinear(b, 'encoder.fc1', S, c.hidden);
    declareLinear(b, 'encoder.fc2', c.hidden, c.latent);
    for (const name of ['dynamics', 'reward', 'q1', 'q2']) {
      declareLinear(b, `${name}.fc1`, c.latent + A, c.hidden);
      declareLinear(b, `${name}.fc2`, c.hidden, name === 'dynamics' ? c.latent : 1);
    }
    declareLinear(b, 'pi.fc1', c.latent, c.hidden);
    declareLinear(b, 'pi.fc2', c.hidden, A);
    return b.store;
  },
  datasets: [pointMassDataset<TdmpcConfig>((c) => c.horizon + 1)],
  forward(batch, c) {
    const r = modelLoss(batch, c);
    const pick = (vs: Var[]) => reshape(select(stack(vs, 1), 0, 0), [vs.length, -1]);
    const T = batch.inputs.state.shape[1];
    // Sequence 1 of the batch, for the views; each in the scope of its producer.
    return {
      outputs: {
        latentPredicted: scope('dynamics', () => pick(r.zs)),
        latentEncoded: scope('encoder', () =>
          encodeState(select(constant(batch.inputs.state, 'state'), 0, 0)),
        ),
        rewardPredicted: scope('reward', () => reshape(pick(r.rPred), [T - 1])),
        rewardObserved: scope('reward', () =>
          slice(select(constant(batch.inputs.reward, 'reward'), 0, 0), 0, 1, T),
        ),
        q: scope('q', () => reshape(pick(r.qPred), [T - 1])),
      },
      loss: r.loss,
      metrics: r.metrics,
    };
  },
  // These losses sum over 256 pixels (or many latent terms); with ε = 1e-6 the float64
  // round-off in L(θ ± ε) is comparable to the smallest gradients.
  gradCheck: { eps: 1e-5 },
  training: {
    support: 'full',
    note: 'Encoder, dynamics, reward, Q and policy all train here, on a fixed offline dataset; there is no environment-interaction loop or replay buffer filled by the planner.',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 3e-3, clip: 'norm', clipValue: 10 },
    batchSize: 32,
    steps: 400,
    phases: [
      {
        id: 'model',
        label: 'Model (consistency + reward + TD)',
        params: (n) => !isPolicy(n),
        loss: (b, c) => modelLoss(b, c),
        checkLoss: (b, c) => {
          const targets = computeTargets(b, c);
          return () => modelLoss(b, c, targets).loss;
        },
      },
      { id: 'policy', label: 'Policy (maximise Q)', params: isPolicy, loss: policyLoss },
    ],
    validateEvery: 25,
  },
  procedures: [planProcedure, controlProcedure],
  views: [
    {
      kind: 'series',
      title: 'MPPI: return of the mean plan and of the elites per iteration',
      sources: [
        { ref: 'proc:plan:meanReturn', label: 'mean sequence μ (before iteration j)' },
        { ref: 'proc:plan:eliteMean', label: 'elite mean' },
      ],
      xLabel: 'iteration',
    },
    {
      kind: 'matrix',
      title: 'MPPI: sampled returns per iteration (rows), sorted',
      source: 'proc:plan:returns',
    },
    {
      kind: 'series',
      title: 'MPPI: first action of μ per iteration',
      sources: [{ ref: 'proc:plan:meanAction', label: 'μ₀ (pick dimension)' }],
      xLabel: 'iteration',
    },
    {
      kind: 'scatter',
      title: 'Planned sequence executed in the simulator (start 1, goal 2)',
      source: 'proc:plan:path',
      colorBy: 'proc:plan:pathColor',
      extent: ENV_EXTENT,
    },
    {
      kind: 'series',
      title: 'Reward along the plan: model vs. simulator',
      sources: [
        { ref: 'proc:plan:rewardModel', label: 'R(ẑ_t, a_t)' },
        { ref: 'proc:plan:rewardTrue', label: 'simulator r(s_{t+1})' },
      ],
      xLabel: 't',
    },
    {
      kind: 'scatter',
      title: 'Closed-loop MPPI in the simulator (start 1, goal 2)',
      source: 'proc:control:path',
      colorBy: 'proc:control:pathColor',
      extent: ENV_EXTENT,
    },
    {
      kind: 'series',
      title: 'Closed-loop reward: MPPI vs. scripted data policy',
      sources: [
        { ref: 'proc:control:reward', label: 'MPPI' },
        { ref: 'proc:control:scriptedReward', label: 'scripted policy' },
      ],
      xLabel: 't',
    },
    {
      kind: 'series',
      title: 'Predicted vs. observed reward (sequence 1)',
      sources: [
        { ref: 'out:rewardPredicted', label: 'R(ẑ_t, a_t)' },
        { ref: 'out:rewardObserved', label: 'r(s_{t+1})' },
      ],
      xLabel: 't',
    },
    { kind: 'matrix', title: 'Latent rollout ẑ_t (rows: t)', source: 'out:latentPredicted' },
  ],
  math: {
    env: ENV_MATH,
    encoder: {
      tex: 'z_t = h_\\theta(s_t)',
      text: 'Two dense layers with ELU on the 4-dimensional state.',
    },
    dynamics: {
      tex: 'z_{t+1} = d_\\theta(z_t, a_t)',
      text: 'Deterministic latent transition; no decoder, so the latent only needs to predict rewards and values.',
    },
    reward: { tex: '\\hat r_t = R_\\theta(z_t, a_t)' },
    q: { tex: 'Q_{\\theta,i}(z_t, a_t),\\ i = 1, 2' },
    pi: {
      tex: '\\pi_\\theta(z_t) = \\tanh(\\mathrm{MLP}(z_t))',
      text: 'Trained to maximise min_i Q_i with Q frozen; used for TD targets and as extra MPPI candidates.',
    },
    targets: {
      tex: 'y_t = r_t + \\gamma \\min_i Q_i(z^+_{t+1}, \\pi(z^+_{t+1})),\\quad z^+_{t+1} = \\operatorname{sg}(h_\\theta(s_{t+1}))',
      text: 'Computed with gradients stopped (current networks; the paper uses a slow-moving target copy).',
    },
    loss: {
      tex: '\\mathcal L = \\frac{1}{H}\\sum_{t=0}^{H-1} \\rho^t \\big[c_1 (\\hat r_t - r_t)^2 + c_2 \\textstyle\\sum_i (Q_i - y_t)^2 + c_3 \\lVert \\hat z_{t+1} - z^+_{t+1}\\rVert^2\\big]',
      text: 'Latent consistency replaces image reconstruction: the predicted latent is pulled towards the encoding of the observed next state. r_t is the reward received after a_t (r(s_{t+1}) in the environment’s notation), and ‖·‖² is averaged over the latent dimensions, as in the reference implementation.',
    },
  },
  checkpoints: [
    {
      id: 'point-mass',
      label: 'Trained on point-mass trajectories',
      path: 'weights/td-mpc-point-mass',
      description:
        'Default configuration trained offline by scripts/train/td-mpc.ts (4,000 two-phase steps, batch 64) on the same dataset.',
    },
  ],
});

export default tdmpc;
