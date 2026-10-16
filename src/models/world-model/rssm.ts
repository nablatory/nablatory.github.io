import {
  ParamBuilder,
  Tensor,
  defaultOptimizer,
  reshape,
  scope,
  select,
  stack,
  type Var,
} from '../../engine';
import { defineModel, type Procedure, type ProcedureResult } from '../types';
import { constant } from '../util';
import { ENV, ENV_MATH, OBS_DIM, episodes, pointMassDataset } from './env';
import {
  declareRssm,
  decode,
  encode,
  feature,
  imagineWithActions,
  observe,
  rewardHead,
  rssmDefaults,
  rssmFields,
  rssmMath,
  worldModelLoss,
  type RssmConfig,
} from './rssm-core';

export type RssmModelConfig = RssmConfig;

/** [T, 1, 16, 16] frames of sequence 0 of a [B, T, 256] tensor. */
export function framesOf(v: Var): Var {
  const T = v.shape[1];
  return reshape(select(v, 0, 0), [T, 1, ENV.size, ENV.size]);
}

/**
 * Filter the first `context` frames of a held-out episode with the posterior,
 * then predict open loop with the prior under the episode's recorded actions.
 */
export function openLoop(
  c: RssmConfig,
  args: { episode: number; context: number; horizon: number; sample: boolean },
): ProcedureResult {
  const ep = episodes().val[Math.max(0, Math.min(episodes().val.length - 1, args.episode - 1))];
  const K = Math.max(1, Math.min(args.context, ep.length));
  const H = Math.max(1, Math.min(args.horizon, ep.length + 1 - K));
  const obs = constant(new Tensor([1, K, OBS_DIM], ep.obs.slice(0, K * OBS_DIM)), 'observations');
  const actsAll = new Tensor([1, ep.length, 2], ep.actions.slice());
  const ctxActs = constant(new Tensor([1, K, 2], ep.actions.slice(0, K * 2)), 'actions');
  const filt = observe(encode(obs, c), ctxActs, c, args.sample);
  // Open loop from the last filtered state: a_{K−1}, …, a_{K+H−2}.
  const acts: Var[] = [];
  for (let t = K - 1; t < K - 1 + H; t++) {
    const a = new Tensor([1, 2], actsAll.data.slice(t * 2, t * 2 + 2));
    acts.push(constant(a, 'action'));
  }
  const im = imagineWithActions(filt.h[K - 1], filt.z[K - 1], acts, c.stoch, args.sample);
  const hs = [...filt.h, ...im.h];
  const zs = [...filt.z, ...im.z];
  const feat = feature(stack(hs, 1), stack(zs, 1));
  const frames = decode(feat, c).value;
  const rewards = rewardHead(feat).value;
  const N = K + H;
  const observed = new Float32Array(N * OBS_DIM);
  observed.set(ep.obs.subarray(0, N * OBS_DIM));
  const errors = new Float32Array(N);
  for (let t = 0; t < N; t++) {
    let s = 0;
    for (let i = 0; i < OBS_DIM; i++)
      s += (frames.data[t * OBS_DIM + i] - observed[t * OBS_DIM + i]) ** 2;
    errors[t] = s / OBS_DIM;
  }
  return {
    tensors: {
      predicted: new Tensor([N, 1, ENV.size, ENV.size], Float32Array.from(frames.data)),
      observed: new Tensor([N, 1, ENV.size, ENV.size], observed),
      reward: new Tensor([N], Float32Array.from(rewards.data)),
      rewardObserved: new Tensor([N], ep.rewards.slice(0, N)),
      pixelError: new Tensor([N], errors),
    },
    scalars: { context: K, horizon: H },
    meta: {
      caption: `Steps 0–${K - 1} filtered with the posterior; steps ${K}–${N - 1} predicted open loop with the prior.`,
    },
  };
}

export const openLoopProcedure: Procedure<RssmConfig> = {
  id: 'openloop',
  label: 'Open-loop prediction',
  description:
    'Filters the first frames of a held-out episode with the posterior q(z|h, o), then rolls the prior p(z|h) forward under the episode’s recorded actions without seeing further frames, and compares the decoded images and rewards with the simulator.',
  args: [
    { key: 'episode', label: 'Held-out episode', kind: 'int', min: 1, max: 8, affects: 'data' },
    { key: 'context', label: 'Context frames', kind: 'int', min: 1, max: 20, affects: 'forward' },
    { key: 'horizon', label: 'Open-loop steps', kind: 'int', min: 1, max: 30, affects: 'forward' },
    { key: 'sample', label: 'Sample z (else use means)', kind: 'bool', affects: 'forward' },
  ],
  defaults: { episode: 1, context: 5, horizon: 15, sample: false },
  run({ config, args }) {
    return openLoop(config, {
      episode: Number(args.episode),
      context: Number(args.context),
      horizon: Number(args.horizon),
      sample: Boolean(args.sample),
    });
  },
};

const rssm = defineModel<RssmModelConfig>({
  id: 'rssm',
  family: 'world-model',
  name: 'Recurrent state-space model (PlaNet)',
  summary:
    'A latent dynamics model learned from image sequences: a deterministic recurrent state carries memory, a Gaussian stochastic state captures what the next frame adds, and decoders reconstruct images and rewards.',
  signature:
    'h_t = f(h_{t-1}, z_{t-1}, a_{t-1}),\\quad z_t \\sim p(z_t \\mid h_t),\\quad o_t \\sim p(o_t \\mid h_t, z_t),\\quad r_t \\sim p(r_t \\mid h_t, z_t)',
  fidelity: 'reduced',
  scaleNote:
    'The RSSM structure, Gaussian prior and posterior with minimum σ 0.1, reparameterised sampling and the free-nats ELBO are as in PlaNet (with ELU activations, as in Dreamer, rather than PlaNet’s ReLU). Sizes are reduced: h has 32 GRU units instead of 200, z 6 dimensions instead of 30, the default encoder and decoder are two-layer MLPs (a reduced two-layer convolutional pair is selectable) instead of PlaNet’s four-layer convolutional networks, frames are 16 × 16 greyscale instead of 64 × 64 colour, sequences are 12 steps instead of 50, and free nats are 1 instead of 3.',
  references: [
    {
      authors: 'D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson',
      title: 'Learning latent dynamics for planning from pixels',
      year: 2019,
      url: 'https://arxiv.org/abs/1811.04551',
    },
  ],
  limitations: [
    'Trained offline on a fixed dataset from a scripted policy; PlaNet’s data-collection loop with the cross-entropy-method planner is not included (see TD-MPC for latent planning).',
    'Latent overshooting is not used; the loss is the one-step ELBO (the objective Dreamer later adopted).',
    'Free nats are applied per time step to the KL summed over latent dimensions, as max(KL, free nats); Dreamer v1’s reference implementation clips the batch-mean KL instead, which penalises every step once the average exceeds the threshold.',
    'The point-mass environment is a stand-in for the DeepMind Control Suite: its dynamics are simple and its images are 16 × 16.',
    'A few hundred browser steps lower the loss visibly but leave open-loop predictions blurred after a few steps. The shipped checkpoint (6,000 steps) reaches a per-pixel open-loop error of about 0.002 five steps and 0.010 fifteen steps after 5 context frames, averaged over the held-out episodes; predicting the background alone gives about 0.03.',
  ],
  guide: {
    what: 'The input is 12 consecutive 16 × 16 greyscale frames of a point mass, plus the force applied at each step. The model keeps a two-part memory: a deterministic state h_t updated by a GRU (a gated recurrent cell), and a random state z_t drawn from a Gaussian predicted from h_t alone (the prior) or from h_t and the current frame (the posterior). Decoders read the frame and reward from both states; a KL term in the loss pulls the prior and the posterior together, so that the prior alone can predict frames the model has not seen.',
    watch:
      'This page opens with the shipped checkpoint, already trained; press “Start from random weights” to train from scratch. Over the default 400 steps the reconstructed frames go from a grey blur to a dot in the right place (by step 300 the brightest pixel lies within one pixel of the mass in 8–10 of the 12 frames, seeds 1–2) while the loss falls from about 7.5 to under 3; they stay blurrier than the checkpoint’s. Once training has started, the KL per step is nearly always largest at t = 0, where the first frame must be absorbed. Trained on from the checkpoint instead, the model does not improve: the held-out loss drifts from 1.76 to about 1.83 over 400 steps.',
    try: [
      {
        label: 'KL weight β = 0',
        note: 'Nothing pulls the posterior and the prior together: KL rises from about 1.2 to about 8 nats within 100 steps and then levels off. The posterior reconstruction gets sharper (reconstruction 0.38 → 0.24), but “Open-loop prediction” gets worse: the mean pixel error over its 15 open-loop steps, averaged over the eight held-out episodes, goes from 0.0039 to 0.0063 (0.0053 when training on with β = 1).',
        patch: { klScale: 0 },
        train: 200,
      },
      {
        label: 'Convolutional encoder',
        note: 'Parameters are re-initialised; the reconstruction re-forms from scratch through two stride-2 convolutions. Each step takes about twice as long as with the MLP, but by step 300 the dot is back in place in all 12 frames, with about half the MLP’s pixel error at the same step.',
        patch: { vision: 'conv' },
        train: 300,
      },
    ],
    views: ['Posterior reconstruction (sequence 1)', 'KL(q ‖ p) per step (sequence 1)'],
  },
  configSchema: rssmFields,
  defaults: { ...rssmDefaults },
  init(c, rng) {
    const b = new ParamBuilder(rng);
    declareRssm(b, c);
    return b.store;
  },
  datasets: [pointMassDataset<RssmModelConfig>((c) => c.seqLen)],
  forward(batch, c) {
    const r = worldModelLoss(batch, c);
    const T = r.recon.shape[1];
    // Sequence 1 of the batch, for the views; each in the scope of its producer.
    const outputs: Record<string, Var> = {
      reconFrames: scope('decoder', () => framesOf(r.recon)),
      obsFrames: scope('decoder', () => framesOf(r.obsIn)),
      rewardPred: scope('reward', () => select(r.rewardPred, 0, 0)),
      reward: scope('reward', () => select(r.rewardIn, 0, 0)),
      kl: scope('loss', () => select(r.kl, 0, 0)),
      h: scope('transition', () => reshape(select(stack(r.obs.h, 1), 0, 0), [T, c.deter])),
      postMean: scope('posterior', () =>
        reshape(
          select(
            stack(
              r.obs.post.map((q) => q.mean),
              1,
            ),
            0,
            0,
          ),
          [T, c.stoch],
        ),
      ),
      priorMean: scope('prior', () =>
        reshape(
          select(
            stack(
              r.obs.prior.map((q) => q.mean),
              1,
            ),
            0,
            0,
          ),
          [T, c.stoch],
        ),
      ),
    };
    return { outputs, loss: r.loss, metrics: r.metrics };
  },
  // These losses sum over 256 pixels (or many latent terms); with ε = 1e-6 the float64
  // round-off in L(θ ± ε) is comparable to the smallest gradients.
  gradCheck: { eps: 1e-5 },
  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 3e-3, clip: 'norm', clipValue: 100 },
    batchSize: 8,
    steps: 400,
    validateEvery: 25,
  },
  procedures: [openLoopProcedure],
  views: [
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
      title: 'Open-loop prediction vs. simulator',
      predicted: 'proc:openloop:predicted',
      observed: 'proc:openloop:observed',
      reward: 'proc:openloop:reward',
      rewardObserved: 'proc:openloop:rewardObserved',
    },
    {
      kind: 'series',
      title:
        'Pixel error per step (context frames: posterior reconstruction; later steps: open loop)',
      sources: [{ ref: 'proc:openloop:pixelError', label: 'mean squared error' }],
      xLabel: 'step',
    },
    {
      kind: 'series',
      title: 'KL(q ‖ p) per step (sequence 1)',
      sources: [{ ref: 'out:kl', label: 'KL, nats' }],
      xLabel: 't',
    },
    {
      kind: 'series',
      title: 'Posterior and prior means',
      sources: [
        { ref: 'out:postMean', label: 'posterior μ_q' },
        { ref: 'out:priorMean', label: 'prior μ_p' },
      ],
      xLabel: 't',
    },
    { kind: 'matrix', title: 'Deterministic state h_t (rows: t)', source: 'out:h' },
  ],
  math: { ...rssmMath, env: ENV_MATH },
  checkpoints: [
    {
      id: 'point-mass',
      label: 'Trained on point-mass trajectories',
      path: 'weights/rssm-point-mass',
      description:
        'Default configuration trained offline by scripts/train/rssm.ts (6,000 steps, batch 16) on the same dataset.',
    },
  ],
});

export default rssm;
