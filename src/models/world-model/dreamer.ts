import {
  ParamBuilder,
  Tensor,
  add,
  addScalar,
  declareLinear,
  defaultOptimizer,
  graphRng,
  mean,
  mul,
  mulScalar,
  neg,
  noGrad,
  reshape,
  scope,
  select,
  slice,
  softplus,
  square,
  stack,
  sub,
  take,
  tanh,
  type Var,
} from '../../engine';
import { defineModel, type Procedure, type TrainPhase } from '../types';
import { constant } from '../util';
import {
  ENV,
  ENV_EXTENT,
  ENV_MATH,
  OBS_DIM,
  envReward,
  envStep,
  episodes,
  pathScatter,
  pointMassDataset,
  renderStates,
} from './env';
import {
  declareRssm,
  decode,
  encode,
  feature,
  mlp,
  observe,
  posterior,
  prior,
  rewardHead,
  rssmDefaults,
  rssmFields,
  rssmMath,
  sampleGauss,
  transition,
  worldModelLoss,
  type RssmConfig,
} from './rssm-core';
import { framesOf, openLoopProcedure } from './rssm';

export type DreamerConfig = RssmConfig & {
  starts: number;
  horizon: number;
  discount: number;
  lambda: number;
};

const A = ENV.actionDim;
/** softplus(0.54) ≈ 1: the actor starts with σ ≈ 1.1 before the tanh. */
const ACTOR_STD_SHIFT = 0.54;
const ACTOR_MIN_STD = 0.1;

export const isActor = (n: string) => n.startsWith('actor.');
export const isCritic = (n: string) => n.startsWith('value.');
export const isWorldModel = (n: string) => !isActor(n) && !isCritic(n);

/** a = tanh(μ + σ ε) (sample) or tanh(μ) (mode), σ = softplus(· + 0.54) + 0.1. */
export function actorAction(feat: Var, sample: boolean): Var {
  return scope('actor', () => {
    const out = mlp(feat, 'actor');
    const ax = out.value.rank - 1;
    const mu = slice(out, ax, 0, A);
    if (!sample) return tanh(mu);
    const std = addScalar(
      softplus(addScalar(slice(out, ax, A, 2 * A), ACTOR_STD_SHIFT)),
      ACTOR_MIN_STD,
    );
    const rng = graphRng();
    const eps = Tensor.zeros(mu.shape);
    for (let i = 0; i < eps.size; i++) eps.data[i] = rng.normal();
    return tanh(add(mu, mul(std, constant(eps, 'ε'))));
  });
}

export function valueHead(feat: Var): Var {
  return scope('value', () => mlp(feat, 'value'));
}

export interface Imagined {
  /** [H, N, D] features of s_1 … s_H */
  feat: Var;
  /** [H, N, A] actions a_0 … a_{H−1} */
  actions: Var;
  reward: Var;
  value: Var;
}

/**
 * Imagine H steps from start states with the actor and the prior:
 *   a_τ = π(s_τ),  h_{τ+1} = f(h_τ, z_τ, a_τ),  z_{τ+1} ~ p(z | h_{τ+1}).
 */
export function imagine(
  h0: Var,
  z0: Var,
  c: DreamerConfig,
  sampleAction: boolean,
  samplePrior = true,
): Imagined {
  return scope('imagination', () => {
    let h = h0;
    let z = z0;
    const feats: Var[] = [];
    const acts: Var[] = [];
    for (let t = 0; t < c.horizon; t++) {
      const a = actorAction(feature(h, z), sampleAction);
      h = transition(h, z, a);
      z = scope('prior', () => sampleGauss(prior(h, c.stoch), samplePrior));
      feats.push(feature(h, z));
      acts.push(a);
    }
    const feat = stack(feats, 0);
    return { feat, actions: stack(acts, 0), reward: rewardHead(feat), value: valueHead(feat) };
  });
}

/**
 * λ-returns (Hafner et al. 2020, eq. 6), computed backwards:
 *   V^λ_H = v(s_H),  V^λ_τ = r_τ + γ ((1 − λ) v(s_{τ+1}) + λ V^λ_{τ+1}).
 * Returns [H − 1, N, 1] for τ = 1 … H − 1.
 */
export function lambdaReturns(reward: Var, value: Var, gamma: number, lambda: number): Var {
  return scope('returns', () => {
    const H = reward.shape[0];
    let next = select(value, 0, H - 1);
    const out: Var[] = [];
    for (let t = H - 2; t >= 0; t--) {
      const boot = add(mulScalar(select(value, 0, t + 1), 1 - lambda), mulScalar(next, lambda));
      next = add(select(reward, 0, t), mulScalar(boot, gamma));
      out.unshift(next);
    }
    return stack(out, 0);
  });
}

/** Indices of `k` start states spread evenly over n = B·T posterior states. */
export function startIndices(n: number, k: number): number[] {
  const m = Math.max(1, Math.min(n, k));
  return Array.from({ length: m }, (_, i) => Math.floor((i * n) / m));
}

/**
 * Within one training step the three phases see the same batch object.
 * As in Dreamer's implementation, the imagination starts from the posterior
 * states of the world-model pass, and the critic regresses on the trajectories
 * and λ-returns of the actor pass. These caches carry them between phases;
 * when absent (inspection, gradient checks) they are recomputed.
 */
const startCache = new WeakMap<object, { h: Tensor; z: Tensor }>();
const imaginationCache = new WeakMap<object, { feat: Tensor; target: Tensor }>();

function selectStarts(hs: Var[], zs: Var[], c: DreamerConfig): { h: Tensor; z: Tensor } {
  return noGrad(() => {
    const B = hs[0].shape[0];
    const n = B * hs.length;
    const idx = startIndices(n, c.starts);
    return {
      h: take(reshape(stack(hs, 1), [n, c.deter]), idx).value,
      z: take(reshape(stack(zs, 1), [n, c.stoch]), idx).value,
    };
  });
}

/**
 * Posterior states of the batch, detached: the imagination start states.
 * `starts` of the B·T states are used, spread evenly over sequences and steps.
 */
export function startStates(
  batch: { inputs: Record<string, Tensor> },
  c: DreamerConfig,
): { h: Var; z: Var } {
  return scope('start', () => {
    let st = startCache.get(batch);
    if (!st) {
      st = noGrad(() => {
        const obs = constant(batch.inputs.obs, 'observations');
        const act = constant(batch.inputs.action, 'actions');
        const o = observe(encode(obs, c), act, c, true);
        return selectStarts(o.h, o.z, c);
      });
    }
    return { h: constant(st.h, 'sg(h)'), z: constant(st.z, 'sg(z)') };
  });
}

export function worldPhaseLoss(batch: { inputs: Record<string, Tensor> }, c: DreamerConfig) {
  const r = worldModelLoss(batch as never, c);
  startCache.set(batch, selectStarts(r.obs.h, r.obs.z, c));
  return { loss: r.loss, metrics: r.metrics };
}

export function actorLoss(batch: { inputs: Record<string, Tensor> }, c: DreamerConfig) {
  const s = startStates(batch, c);
  const im = imagine(s.h, s.z, c, true);
  const ret = lambdaReturns(im.reward, im.value, c.discount, c.lambda);
  const H = im.feat.shape[0];
  imaginationCache.set(batch, { feat: slice(im.feat, 0, 0, H - 1).value, target: ret.value });
  const loss = scope('loss', () => neg(mean(ret)));
  return { loss, metrics: { imaginedReturn: -loss.value.data[0] } };
}

export function criticLoss(batch: { inputs: Record<string, Tensor> }, c: DreamerConfig) {
  let cached = imaginationCache.get(batch);
  if (!cached) {
    const s = startStates(batch, c);
    cached = noGrad(() => {
      const im = imagine(s.h, s.z, c, true);
      const H = im.feat.shape[0];
      return {
        feat: slice(im.feat, 0, 0, H - 1).value,
        target: lambdaReturns(im.reward, im.value, c.discount, c.lambda).value,
      };
    });
  }
  const pred = valueHead(constant(cached.feat, 'sg(s_τ)'));
  const loss = scope('loss', () =>
    mean(mulScalar(square(sub(pred, constant(cached.target, 'sg(V^λ)'))), 0.5)),
  );
  return { loss, metrics: { valueLoss: loss.value.data[0] } };
}

const phases: TrainPhase<DreamerConfig>[] = [
  { id: 'world', label: 'World model', params: isWorldModel, loss: worldPhaseLoss },
  { id: 'actor', label: 'Actor (through imagined dynamics)', params: isActor, loss: actorLoss },
  { id: 'critic', label: 'Critic (λ-return regression)', params: isCritic, loss: criticLoss },
];

function clampEpisode(e: number): number {
  return Math.max(0, Math.min(episodes().val.length - 1, Math.round(e) - 1));
}

/** Filter the first K frames of an episode; returns the last posterior state. */
function filterEpisode(
  c: DreamerConfig,
  ep: number,
  K: number,
  sample: boolean,
): { h: Var; z: Var } {
  const e = episodes().val[ep];
  const obs = constant(new Tensor([1, K, OBS_DIM], e.obs.slice(0, K * OBS_DIM)), 'observations');
  const act = constant(new Tensor([1, K, A], e.actions.slice(0, K * A)), 'actions');
  const o = observe(encode(obs, c), act, c, sample);
  return { h: o.h[K - 1], z: o.z[K - 1] };
}

const imagineProcedure: Procedure<DreamerConfig> = {
  id: 'imagine',
  label: 'Imagined trajectory',
  description:
    'Filters the first frames of a held-out episode, then lets the actor choose actions (its mode, tanh μ) inside the learned model for H steps. The same actions are replayed in the simulator from the true state for comparison.',
  args: [
    { key: 'episode', label: 'Held-out episode', kind: 'int', min: 1, max: 8, affects: 'data' },
    { key: 'context', label: 'Context frames', kind: 'int', min: 1, max: 20, affects: 'forward' },
    {
      key: 'sample',
      label: 'Sample z from the prior (else its mean)',
      kind: 'bool',
      affects: 'forward',
    },
  ],
  defaults: { episode: 1, context: 5, sample: false },
  run({ config: c, args }) {
    const ep = clampEpisode(Number(args.episode));
    const K = Math.max(1, Math.min(Number(args.context), ENV.episodeLength));
    const start = filterEpisode(c, ep, K, false);
    const im = imagine(start.h, start.z, c, false, Boolean(args.sample));
    const H = c.horizon;
    const frames = decode(im.feat, c).value;
    const acts = im.actions.value;
    const e = episodes().val[ep];
    let s = Array.from(e.states.subarray((K - 1) * 4, K * 4));
    const states: number[][] = [];
    const rTrue = new Float32Array(H);
    for (let t = 0; t < H; t++) {
      s = envStep(s, [acts.data[t * A], acts.data[t * A + 1]]);
      states.push(s);
      rTrue[t] = envReward(s);
    }
    const ret = lambdaReturns(im.reward, im.value, c.discount, c.lambda).value;
    return {
      tensors: {
        predicted: new Tensor([H, 1, ENV.size, ENV.size], Float32Array.from(frames.data)),
        observed: renderStates(states),
        reward: new Tensor([H], Float32Array.from(im.reward.value.data)),
        rewardObserved: new Tensor([H], rTrue),
        value: new Tensor([H], Float32Array.from(im.value.value.data)),
        lambdaReturn: new Tensor([H - 1], Float32Array.from(ret.data)),
        actions: new Tensor([H, A], Float32Array.from(acts.data)),
      },
      meta: {
        caption: `Imagination from step ${K - 1} of held-out episode ${ep + 1}; the simulator replays the actor’s actions from the true state.`,
      },
    };
  },
};

const controlProcedure: Procedure<DreamerConfig> = {
  id: 'control',
  label: 'Act in the simulator',
  description:
    'Runs the actor (mode actions) in the true simulator for one episode, updating the posterior from each new rendered frame, and compares its rewards with the scripted policy that generated the training data from the same start.',
  args: [
    {
      key: 'episode',
      label: 'Start state of held-out episode',
      kind: 'int',
      min: 1,
      max: 8,
      affects: 'data',
    },
  ],
  defaults: { episode: 1 },
  run({ config: c, args, shouldStop, progress }) {
    const ep = clampEpisode(Number(args.episode));
    const e = episodes().val[ep];
    const L = ENV.episodeLength;
    let s = Array.from(e.states.subarray(0, 4));
    const states: number[][] = [s];
    const rewards = new Float32Array(L + 1);
    rewards[0] = envReward(s);
    let h = constant(Tensor.zeros([1, c.deter]), 'h₀');
    let z = constant(Tensor.zeros([1, c.stoch]), 'z₀');
    let a = constant(Tensor.zeros([1, A]), 'a₀');
    const actions = new Float32Array(L * A);
    for (let t = 0; t < L; t++) {
      if (shouldStop()) break;
      const o = renderStates([s]);
      h = transition(h, z, a);
      const e1 = encode(constant(new Tensor([1, OBS_DIM], o.data), 'o_t'), c);
      z = posterior(h, e1, c.stoch).mean;
      a = actorAction(feature(h, z), false);
      const act = [a.value.data[0], a.value.data[1]];
      actions.set(act, t * A);
      s = envStep(s, act);
      states.push(s);
      rewards[t + 1] = envReward(s);
      progress((t + 1) / L);
    }
    const { points, colors } = pathScatter(states);
    let sumA = 0;
    let sumB = 0;
    for (let t = 1; t <= L; t++) {
      sumA += rewards[t];
      sumB += e.rewards[t];
    }
    return {
      tensors: {
        path: points,
        pathColor: colors,
        reward: new Tensor([L + 1], rewards),
        scriptedReward: new Tensor([L + 1], e.rewards.slice()),
        actions: new Tensor([L, A], actions),
      },
      scalars: { actorReturn: sumA, scriptedReturn: sumB },
      meta: { caption: `Return Σr: actor ${sumA.toFixed(2)}, scripted policy ${sumB.toFixed(2)}.` },
    };
  },
};

const dreamer = defineModel<DreamerConfig>({
  id: 'dreamer',
  family: 'world-model',
  name: 'Dreamer',
  summary:
    'An RSSM world model learned from image sequences, plus an actor and a critic trained entirely inside the model: the actor maximises λ-returns of imagined trajectories by backpropagating through the learned dynamics.',
  signature:
    '\\max_{\\phi}\\ \\mathbb E_{q_\\theta, q_\\phi}\\Big[\\sum_{\\tau=t}^{t+H} V_\\lambda(s_\\tau)\\Big],\\quad V_\\lambda(s_\\tau) = r_\\tau + \\gamma\\big((1-\\lambda) v_\\psi(s_{\\tau+1}) + \\lambda V_\\lambda(s_{\\tau+1})\\big)',
  fidelity: 'reduced',
  scaleNote:
    'The RSSM world model (as on the RSSM page), the tanh-Gaussian actor trained by analytic gradients through imagined dynamics, and the critic regressed on λ-returns follow Dreamer (v1), as do the three Adam optimisers (world model, actor, critic) that each step once per training step. Reduced: h 32 / z 6 units, MLP image encoder and decoder on 16 × 16 frames, one-hidden-layer actor and critic of 64 units, 32 imagination start states per batch instead of all, imagination horizon 10 instead of 15, discount 0.95 instead of 0.99, and one learning rate for all three optimisers.',
  references: [
    {
      authors: 'D. Hafner, T. Lillicrap, J. Ba, M. Norouzi',
      title: 'Dream to control: learning behaviors by latent imagination',
      year: 2020,
      url: 'https://arxiv.org/abs/1912.01603',
    },
    {
      authors: 'D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson',
      title: 'Learning latent dynamics for planning from pixels',
      year: 2019,
      url: 'https://arxiv.org/abs/1811.04551',
    },
  ],
  limitations: [
    'All three components train in the browser, but on a fixed offline dataset from a scripted policy: Dreamer’s loop of collecting new episodes with the current actor is not run, so the actor can exploit model errors in regions the data never visited.',
    'Each training step runs three phases on the same batch: world model, then actor (world-model and critic parameters frozen, gradients flow through them), then critic. As in the reference implementation, imagination starts from the posterior states of the world-model pass, and the critic regresses on the actor pass’s imagined states and λ-returns (gradients stopped). Unlike it, the phases run one after another, so the actor imagines with the world model already updated in this step. Each phase has its own Adam state.',
    'Imagination starts from 32 of the 96 posterior states of a batch (at the default 8 sequences of 12 steps; spread evenly and configurable up to all of them, which the paper uses) to keep a training step fast; the start states are detached.',
    'The actor is a tanh-squashed Gaussian as in Dreamer v1, but with σ = softplus(x + 0.54) + 0.1 (initial σ ≈ 1.1, minimum 0.1) and an unscaled mean, where the reference implementation uses an initial σ of 5, a minimum of 10⁻⁴ and the mean 5 tanh(μ/5). In imagination the actor reads the imagined state without the stop-gradient that the reference implementation places on its input, so the return gradient also flows through the actor’s own input features: the paper’s objective taken literally.',
    'No episode termination or learned discount; the actor has no entropy bonus, as in Dreamer v1 for continuous actions.',
    'The point-mass task is far simpler than the DeepMind Control Suite tasks of the paper. With a few hundred browser steps the actor improves its imagined return long before it improves in the simulator. The shipped checkpoint (5,000 steps) collects a mean return of 35.5 over 40 simulator steps from the held-out start states, against 14.8 for the scripted policy that produced the data (40 is the maximum).',
  ],
  guide: {
    what: 'Dreamer adds two networks to the RSSM world model: an actor that picks a force from the latent state (the model’s compressed memory) and a critic that estimates the reward still to come. Neither touches the simulator during training. From states inferred on recorded 12-frame sequences, the actor imagines 10 steps inside the learned model and is trained to raise the λ-return, a mix of predicted rewards and the critic’s value, by gradients flowing back through the model; the critic learns to predict those returns.',
    watch:
      'Opens with the shipped checkpoint, already trained; press “Start from random weights” to train from scratch. Each step runs three losses in turn: world model, actor, critic. Over the default 400 steps the imagined frames stay blurry (their brightest pixel reaches only about 0.4–0.6, against about 0.9 at the checkpoint and 1 in real frames). The actor’s imagined return climbs from about 4 to 11–15 (seeds 1–2), while its return in the simulator (“Act in the simulator”, mean over the eight held-out starts) first falls from 16–19 for the untrained actor to about 3 and then recovers only erratically, to 7 and 12 at step 400: below the scripted policy’s 14.8 and far from the checkpoint’s 35.5.',
    try: [
      {
        label: 'Imagination horizon 20',
        note: 'Twenty imagined steps. In “Imagined trajectory” the later frames drift further from the simulator replay as model errors compound (mean pixel error about 0.001 over the first five steps and 0.012 over the last five, held-out episodes 1–8), and a training step takes about 1.5 times as long.',
        patch: { horizon: 20 },
      },
      {
        label: 'λ = 0',
        note: 'The λ-return becomes r + γ v(s′): each point is one predicted reward plus the critic’s discounted value one step ahead, instead of a sum of rewards over the horizon. At the checkpoint the critic already agrees with the summed rewards, so the λ-return curve moves by only about 0.25, on values of 12–18.',
        patch: { lambda: 0 },
      },
      {
        label: 'All start states',
        note: 'Every posterior state of the batch starts an imagined path, as in the paper. Each step takes about twice as long; after 50 steps the imagined and simulator returns are the same as with 32 start states (16.4 and 35.3, against 16.3 and 35.1).',
        patch: { starts: 96 },
        train: 50,
      },
    ],
    views: ['Training-time imagination (sampled actions)', 'Training-time imagination: values'],
  },
  configSchema: [
    ...rssmFields,
    {
      key: 'starts',
      label: 'Imagination start states',
      kind: 'int',
      min: 1,
      max: 240,
      affects: 'forward',
      help: 'How many of the B·T posterior states of a batch start an imagined trajectory (the paper uses all of them).',
    },
    {
      key: 'horizon',
      label: 'Imagination horizon H',
      kind: 'int',
      min: 2,
      max: 20,
      affects: 'forward',
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
      key: 'lambda',
      label: 'λ (return mixing)',
      kind: 'float',
      min: 0,
      max: 1,
      step: 0.05,
      affects: 'forward',
    },
  ],
  defaults: { ...rssmDefaults, starts: 32, horizon: 10, discount: 0.95, lambda: 0.95 },
  init(c, rng) {
    const b = new ParamBuilder(rng);
    declareRssm(b, c);
    declareLinear(b, 'actor.fc1', c.deter + c.stoch, c.hidden);
    declareLinear(b, 'actor.fc2', c.hidden, 2 * A);
    declareLinear(b, 'value.fc1', c.deter + c.stoch, c.hidden);
    declareLinear(b, 'value.fc2', c.hidden, 1);
    return b.store;
  },
  datasets: [pointMassDataset<DreamerConfig>((c) => c.seqLen)],
  forward(batch, c) {
    const r = worldModelLoss(batch, c);
    const T = r.recon.shape[1];
    // Imagination from the last posterior state of the first sequence (detached).
    const { h0, z0 } = scope('start', () => ({
      h0: constant(reshape(select(r.obs.h[T - 1], 0, 0), [1, c.deter]).value, 'sg(h)'),
      z0: constant(reshape(select(r.obs.z[T - 1], 0, 0), [1, c.stoch]).value, 'sg(z)'),
    }));
    const im = imagine(h0, z0, c, true);
    const ret = lambdaReturns(im.reward, im.value, c.discount, c.lambda);
    const H = c.horizon;
    const outputs: Record<string, Var> = {
      reconFrames: scope('decoder', () => framesOf(r.recon)),
      obsFrames: scope('decoder', () => framesOf(r.obsIn)),
      rewardPred: scope('reward', () => select(r.rewardPred, 0, 0)),
      reward: scope('reward', () => select(r.rewardIn, 0, 0)),
      imaginedFrames: scope('imagination', () =>
        reshape(decode(im.feat, c), [H, 1, ENV.size, ENV.size]),
      ),
      imaginedReward: scope('imagination', () => reshape(im.reward, [H])),
      imaginedValue: scope('imagination', () => reshape(im.value, [H])),
      actions: scope('imagination', () => reshape(im.actions, [H, A])),
      lambdaReturn: scope('returns', () => reshape(ret, [H - 1])),
    };
    const meanReturn = scope('returns', () => mean(ret)).value.data[0];
    return { outputs, loss: r.loss, metrics: { ...r.metrics, imaginedReturn: meanReturn } };
  },
  // These losses sum over 256 pixels (or many latent terms); with ε = 1e-6 the float64
  // round-off in L(θ ± ε) is comparable to the smallest gradients.
  gradCheck: { eps: 1e-5 },
  training: {
    support: 'full',
    note: 'World model, actor and critic all train here, on a fixed offline dataset; the environment-interaction loop of Dreamer is not run during training.',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 3e-3, clip: 'norm', clipValue: 100 },
    batchSize: 8,
    steps: 400,
    phases,
    validateEvery: 25,
  },
  procedures: [
    imagineProcedure,
    controlProcedure,
    openLoopProcedure as unknown as Procedure<DreamerConfig>,
  ],
  views: [
    {
      kind: 'rollout',
      title: 'Imagined under the actor vs. simulator with the same actions',
      predicted: 'proc:imagine:predicted',
      observed: 'proc:imagine:observed',
      reward: 'proc:imagine:reward',
      rewardObserved: 'proc:imagine:rewardObserved',
    },
    {
      kind: 'series',
      title: 'Imagined values v(s_τ) and λ-returns',
      sources: [
        { ref: 'proc:imagine:value', label: 'critic v(s_τ)' },
        { ref: 'proc:imagine:lambdaReturn', label: 'λ-return' },
      ],
      xLabel: 'τ',
    },
    {
      kind: 'series',
      title: 'Actor actions in imagination',
      sources: [{ ref: 'proc:imagine:actions', label: 'a_τ (pick dimension)' }],
      xLabel: 'τ',
    },
    {
      kind: 'scatter',
      title: 'Actor in the simulator: path (start 1, goal 2)',
      source: 'proc:control:path',
      colorBy: 'proc:control:pathColor',
      extent: ENV_EXTENT,
    },
    {
      kind: 'series',
      title: 'Simulator reward: actor vs. scripted data policy',
      sources: [
        { ref: 'proc:control:reward', label: 'actor' },
        { ref: 'proc:control:scriptedReward', label: 'scripted policy' },
      ],
      xLabel: 't',
    },
    {
      kind: 'rollout',
      title: 'Posterior reconstruction (sequence 1)',
      predicted: 'out:reconFrames',
      observed: 'out:obsFrames',
      reward: 'out:rewardPred',
      rewardObserved: 'out:reward',
    },
    {
      kind: 'rollout',
      title: 'Training-time imagination (sampled actions)',
      predicted: 'out:imaginedFrames',
      reward: 'out:imaginedReward',
    },
    {
      kind: 'series',
      title: 'Training-time imagination: values',
      sources: [
        { ref: 'out:imaginedValue', label: 'v(s_τ)' },
        { ref: 'out:lambdaReturn', label: 'λ-return' },
      ],
      xLabel: 'τ',
    },
    {
      kind: 'series',
      title: 'Training-time imagination: sampled actions',
      sources: [{ ref: 'out:actions', label: 'a_τ (pick dimension)' }],
      xLabel: 'τ',
    },
  ],
  math: {
    ...rssmMath,
    env: ENV_MATH,
    start: {
      tex: 's_t = (h_t, z_t),\\ z_t \\sim q(z_t \\mid h_t, o_t),\\quad \\text{gradients stopped}',
      text: 'A subset of the posterior states of the batch (32 of 96 at the defaults; the paper uses all of them) each start one imagined trajectory.',
    },
    imagination: {
      tex: 'a_\\tau = \\tanh(\\mu_\\phi(s_\\tau) + \\sigma_\\phi(s_\\tau)\\,\\epsilon),\\quad h_{\\tau+1} = f(h_\\tau, z_\\tau, a_\\tau),\\quad z_{\\tau+1} \\sim p(z \\mid h_{\\tau+1})',
      text: 'The actor acts inside the model; rewards come from the reward head and values from the critic. The whole chain is differentiable in the actor’s parameters.',
    },
    actor: {
      tex: 'a = \\tanh(\\mu + \\sigma \\epsilon),\\quad \\sigma = \\operatorname{softplus}(\\cdot + 0.54) + 0.1',
    },
    value: { tex: 'v_\\psi(s_\\tau) = \\mathrm{MLP}_\\psi([h_\\tau \\Vert z_\\tau])' },
    returns: {
      tex: 'V_\\lambda(s_\\tau) = r_\\tau + \\gamma\\big((1-\\lambda)\\, v_\\psi(s_{\\tau+1}) + \\lambda V_\\lambda(s_{\\tau+1})\\big),\\quad V_\\lambda(s_H) = v_\\psi(s_H)',
      text: 'An exponentially weighted mix of n-step returns that bootstraps with the critic beyond the horizon.',
    },
  },
  checkpoints: [
    {
      id: 'point-mass',
      label: 'Trained on point-mass trajectories',
      path: 'weights/dreamer-point-mass',
      description:
        'World model, actor and critic in the default configuration, trained offline by scripts/train/dreamer.ts (5,000 three-phase steps, batch 16) on the same dataset.',
    },
  ],
});

export default dreamer;
