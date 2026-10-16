import {
  ParamBuilder,
  Tensor,
  add,
  addScalar,
  concat,
  conv,
  convTranspose2d,
  declareConv,
  declareLinear,
  dense,
  elu,
  exp,
  graphRng,
  gruCell,
  log,
  maximum,
  mean,
  mul,
  mulScalar,
  p,
  reshape,
  scope,
  select,
  slice,
  softplus,
  square,
  stack,
  sub,
  sum,
  hasParam,
  type Var,
} from '../../engine';
import type { Batch, ConfigField } from '../types';
import { constant, input } from '../util';
import { ENV, OBS_DIM } from './env';

export interface RssmConfig {
  [key: string]: number | string | boolean;
  deter: number;
  stoch: number;
  hidden: number;
  embed: number;
  freeNats: number;
  klScale: number;
  seqLen: number;
  vision: 'conv' | 'mlp';
  channels: number;
}

export const MIN_STD = 0.1;

export const rssmFields: ConfigField[] = [
  {
    key: 'deter',
    label: 'Deterministic state h (GRU units)',
    kind: 'int',
    min: 4,
    max: 64,
    affects: 'architecture',
  },
  {
    key: 'stoch',
    label: 'Stochastic state z (dims)',
    kind: 'int',
    min: 1,
    max: 16,
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
    key: 'vision',
    label: 'Image encoder / decoder',
    kind: 'select',
    affects: 'architecture',
    options: [
      { value: 'conv', label: 'Convolutional (two stride-2 layers, after PlaNet)' },
      { value: 'mlp', label: 'Two-layer MLP' },
    ],
  },
  {
    key: 'channels',
    label: 'Conv channels (first layer)',
    kind: 'int',
    min: 2,
    max: 16,
    affects: 'architecture',
  },
  {
    key: 'embed',
    label: 'Observation embedding',
    kind: 'int',
    min: 4,
    max: 64,
    affects: 'architecture',
  },
  {
    key: 'freeNats',
    label: 'Free nats',
    kind: 'float',
    min: 0,
    max: 3,
    step: 0.1,
    affects: 'forward',
    help: 'KL terms below this value per time step are not penalised (PlaNet and Dreamer used 3).',
  },
  {
    key: 'klScale',
    label: 'KL scale β',
    kind: 'float',
    min: 0,
    max: 2,
    step: 0.1,
    affects: 'forward',
  },
  {
    key: 'seqLen',
    label: 'Training sequence length T',
    kind: 'int',
    min: 2,
    max: 30,
    affects: 'data',
  },
];

export const rssmDefaults = {
  deter: 32,
  stoch: 6,
  hidden: 64,
  embed: 32,
  freeNats: 1,
  klScale: 1,
  seqLen: 12,
  vision: 'mlp' as const,
  channels: 8,
};

export function declareRssm(b: ParamBuilder, c: RssmConfig): void {
  const A = ENV.actionDim;
  const S = ENV.size / 4;
  if (c.vision === 'conv') {
    // 16×16 → c@8×8 → 2c@4×4 → embedding; mirrored by the decoder.
    declareConv(b, 'encoder.conv1', 1, c.channels, 4);
    declareConv(b, 'encoder.conv2', c.channels, 2 * c.channels, 4);
    declareLinear(b, 'encoder.fc', 2 * c.channels * S * S, c.embed);
  } else {
    declareLinear(b, 'encoder.fc1', OBS_DIM, c.hidden);
    declareLinear(b, 'encoder.fc2', c.hidden, c.embed);
  }
  // The transition MLP is as wide as h (both 200 in PlaNet).
  declareLinear(b, 'transition.input', c.stoch + A, c.deter);
  // GRU, PyTorch layout [r | z | n].
  declareLinear(b, 'transition.gru.ih', c.deter, 3 * c.deter);
  declareLinear(b, 'transition.gru.hh', c.deter, 3 * c.deter);
  declareLinear(b, 'prior.fc1', c.deter, c.hidden);
  declareLinear(b, 'prior.fc2', c.hidden, 2 * c.stoch);
  declareLinear(b, 'posterior.fc1', c.deter + c.embed, c.hidden);
  declareLinear(b, 'posterior.fc2', c.hidden, 2 * c.stoch);
  if (c.vision === 'conv') {
    declareLinear(b, 'decoder.fc', c.deter + c.stoch, 2 * c.channels * S * S);
    // Transposed-convolution weights [C_in, C_out, k, k].
    b.add('decoder.deconv1.weight', [2 * c.channels, c.channels, 4, 4], {
      kind: 'lecunUniform',
      fanIn: 2 * c.channels * 4,
    });
    b.add('decoder.deconv1.bias', [c.channels], { kind: 'zeros' });
    b.add('decoder.deconv2.weight', [c.channels, 1, 4, 4], {
      kind: 'lecunUniform',
      fanIn: c.channels * 4,
    });
    b.add('decoder.deconv2.bias', [1], { kind: 'zeros' });
  } else {
    declareLinear(b, 'decoder.fc1', c.deter + c.stoch, c.hidden);
    declareLinear(b, 'decoder.fc2', c.hidden, OBS_DIM);
  }
  declareLinear(b, 'reward.fc1', c.deter + c.stoch, c.hidden);
  declareLinear(b, 'reward.fc2', c.hidden, 1);
}

export interface Gauss {
  mean: Var;
  std: Var;
}

/** Two-layer MLP with ELU, as used for every RSSM head. */
export function mlp(x: Var, name: string): Var {
  return dense(elu(dense(x, `${name}.fc1`)), `${name}.fc2`);
}

/** Split the last axis into (μ, σ = softplus(·) + 0.1). */
function gaussHead(out: Var, S: number): Gauss {
  const ax = out.value.rank - 1;
  return {
    mean: slice(out, ax, 0, S),
    std: addScalar(softplus(slice(out, ax, S, 2 * S)), MIN_STD),
  };
}

/** e_t = enc(o_t) for observations [..., 256]. */
export function encode(obs: Var, c: RssmConfig): Var {
  return scope('encoder', () => {
    if (c.vision !== 'conv') return mlp(obs, 'encoder');
    const lead = obs.shape.slice(0, -1);
    const n = lead.reduce((a, b) => a * b, 1);
    const img = reshape(obs, [n, 1, ENV.size, ENV.size]);
    const h1 = elu(conv(img, 'encoder.conv1', { stride: 2, padding: 1 }));
    const h2 = elu(conv(h1, 'encoder.conv2', { stride: 2, padding: 1 }));
    return reshape(dense(reshape(h2, [n, -1]), 'encoder.fc'), [...lead, c.embed]);
  });
}

/** h_t = GRU(h_{t−1}, ELU(W [z_{t−1} ‖ a_{t−1}])) */
export function transition(h: Var, z: Var, a: Var): Var {
  return scope('transition', () => {
    const x = elu(dense(concat([z, a], -1), 'transition.input'));
    const bias = (n: string) => (hasParam(n) ? p(n) : undefined);
    return gruCell(
      x,
      h,
      p('transition.gru.ih.weight'),
      p('transition.gru.hh.weight'),
      bias('transition.gru.ih.bias'),
      bias('transition.gru.hh.bias'),
    ).h;
  });
}

/** p(z_t | h_t) */
export function prior(h: Var, S: number): Gauss {
  return scope('prior', () => gaussHead(mlp(h, 'prior'), S));
}

/** q(z_t | h_t, e_t) */
export function posterior(h: Var, e: Var, S: number): Gauss {
  return scope('posterior', () => gaussHead(mlp(concat([h, e], -1), 'posterior'), S));
}

/** Reparameterised sample μ + σ ⊙ ε, ε ~ N(0, I) from the graph generator; the mean when `sample` is false. */
export function sampleGauss(g: Gauss, sample = true): Var {
  if (!sample) return g.mean;
  const rng = graphRng();
  const eps = Tensor.zeros(g.mean.shape);
  for (let i = 0; i < eps.size; i++) eps.data[i] = rng.normal();
  return add(g.mean, mul(g.std, constant(eps, 'ε')));
}

export function feature(h: Var, z: Var): Var {
  return concat([h, z], -1);
}

/** Mean image ô_t = dec([h_t ‖ z_t]) as [..., 256]. */
export function decode(feat: Var, c: RssmConfig): Var {
  return scope('decoder', () => {
    if (c.vision !== 'conv') return mlp(feat, 'decoder');
    const lead = feat.shape.slice(0, -1);
    const n = lead.reduce((a, b) => a * b, 1);
    const S = ENV.size / 4;
    const x = reshape(elu(dense(reshape(feat, [n, -1]), 'decoder.fc')), [n, 2 * c.channels, S, S]);
    const u1 = elu(
      convTranspose2d(x, p('decoder.deconv1.weight'), p('decoder.deconv1.bias'), {
        stride: 2,
        padding: 1,
      }),
    );
    const u2 = convTranspose2d(u1, p('decoder.deconv2.weight'), p('decoder.deconv2.bias'), {
      stride: 2,
      padding: 1,
    });
    return reshape(u2, [...lead, OBS_DIM]);
  });
}

export function rewardHead(feat: Var): Var {
  return scope('reward', () => mlp(feat, 'reward'));
}

/**
 * KL(q ‖ p) between diagonal Gaussians, summed over the last axis:
 *   Σ_d [ log σp − log σq + (σq² + (μq − μp)²) / (2σp²) − ½ ].
 */
export function klDiag(q: Gauss, pr: Gauss): Var {
  const ratio = mul(
    add(square(q.std), square(sub(q.mean, pr.mean))),
    exp(mulScalar(log(pr.std), -2)),
  );
  const t = addScalar(add(sub(log(pr.std), log(q.std)), mulScalar(ratio, 0.5)), -0.5);
  return sum(t, -1);
}

export interface Observed {
  h: Var[];
  z: Var[];
  post: Gauss[];
  prior: Gauss[];
}

/**
 * Filter a batch of sequences: for t = 0 … T−1,
 *   h_t = f(h_{t−1}, z_{t−1}, a_{t−1})  (h_{−1}, z_{−1}, a_{−1} = 0),
 *   z_t ~ q(z_t | h_t, e_t).
 */
export function observe(
  embed: Var,
  actions: Var,
  c: RssmConfig,
  sample = true,
  init?: { h: Var; z: Var },
): Observed {
  const [B, T] = embed.shape;
  let { h, z, a } = scope('transition', () => ({
    h: init?.h ?? constant(Tensor.zeros([B, c.deter]), 'h₀'),
    z: init?.z ?? constant(Tensor.zeros([B, c.stoch]), 'z₀'),
    a: constant(Tensor.zeros([B, ENV.actionDim]), 'a₀'),
  }));
  const out: Observed = { h: [], z: [], post: [], prior: [] };
  for (let t = 0; t < T; t++) {
    h = transition(h, z, a);
    const pr = prior(h, c.stoch);
    const q = posterior(
      h,
      scope('posterior', () => select(embed, 1, t)),
      c.stoch,
    );
    z = scope('posterior', () => sampleGauss(q, sample));
    out.h.push(h);
    out.z.push(z);
    out.post.push(q);
    out.prior.push(pr);
    a = scope('transition', () => select(actions, 1, t));
  }
  return out;
}

/** Open-loop prediction with the prior: h_t = f(h_{t−1}, z_{t−1}, a_{t−1}), z_t ~ p(z_t | h_t). */
export function imagineWithActions(
  h0: Var,
  z0: Var,
  actions: Var[],
  S: number,
  sample: boolean,
): { h: Var[]; z: Var[] } {
  let h = h0;
  let z = z0;
  const out = { h: [] as Var[], z: [] as Var[] };
  for (const a of actions) {
    h = transition(h, z, a);
    z = scope('prior', () => sampleGauss(prior(h, S), sample));
    out.h.push(h);
    out.z.push(z);
  }
  return out;
}

export interface WorldModelResult {
  loss: Var;
  obsIn: Var;
  rewardIn: Var;
  obs: Observed;
  feat: Var;
  recon: Var;
  rewardPred: Var;
  kl: Var;
  metrics: Record<string, number>;
}

/**
 * Negative ELBO of the sequence batch (Gaussian observation and reward
 * likelihoods with unit variance, constants dropped):
 *   L = mean_{b,t} [ ½‖ô_t − o_t‖² + ½(r̂_t − r_t)² + β max(KL_t, free nats) ].
 */
export function worldModelLoss(batch: Batch, c: RssmConfig): WorldModelResult {
  const obsIn = input('obs', batch.inputs.obs, false);
  const act = input('action', batch.inputs.action, false);
  const rew = input('reward', batch.inputs.reward, false);
  const [B, T] = obsIn.shape;
  const embed = encode(obsIn, c);
  const obs = observe(embed, act, c, true);
  const feat = scope('features', () => feature(stack(obs.h, 1), stack(obs.z, 1)));
  const recon = decode(feat, c);
  const rewardPred = scope('reward', () => reshape(rewardHead(feat), [B, T]));
  return scope('loss', () => {
    const recLoss = mean(mulScalar(sum(square(sub(recon, obsIn)), -1), 0.5));
    const rewLoss = mean(mulScalar(square(sub(rewardPred, rew)), 0.5));
    const kl = stack(
      obs.post.map((q, t) => klDiag(q, obs.prior[t])),
      1,
    ); // [B, T]
    const klFree = mean(maximum(kl, constant(Tensor.full([1], c.freeNats), 'free nats')));
    const loss = add(add(recLoss, rewLoss), mulScalar(klFree, c.klScale));
    const klMean = mean(kl).value.data[0];
    return {
      loss,
      obsIn,
      rewardIn: rew,
      obs,
      feat,
      recon,
      rewardPred,
      kl,
      metrics: { reconstruction: recLoss.value.data[0], reward: rewLoss.value.data[0], kl: klMean },
    };
  });
}

export const rssmMath = {
  encoder: {
    tex: 'e_t = \\operatorname{enc}_\\theta(o_t)',
    text: 'Two dense layers with ELU map the 256-pixel image to an embedding; with the convolutional option, two stride-2 4 × 4 convolutions with ELU followed by a dense layer.',
  },
  transition: {
    tex: 'h_t = \\mathrm{GRU}\\big(h_{t-1},\\, \\mathrm{ELU}(W [z_{t-1} \\Vert a_{t-1}])\\big)',
    text: 'The deterministic path carries information across time; it is what lets the model infer velocity, which no single frame shows.',
  },
  prior: {
    tex: 'p(z_t \\mid h_t) = \\mathcal N\\big(\\mu_p(h_t),\\, \\operatorname{diag}\\sigma_p^2(h_t)\\big),\\quad \\sigma = \\operatorname{softplus}(\\cdot) + 0.1',
  },
  posterior: {
    tex: 'q(z_t \\mid h_t, e_t) = \\mathcal N\\big(\\mu_q(h_t, e_t),\\, \\operatorname{diag}\\sigma_q^2\\big),\\quad z_t = \\mu_q + \\sigma_q \\odot \\epsilon,\\ \\epsilon \\sim \\mathcal N(0, I)',
    text: 'The reparameterised sample uses noise from the seeded graph generator, so a forward pass is reproducible and differentiable in μ and σ.',
  },
  decoder: {
    tex: '\\hat o_t = \\operatorname{dec}_\\theta([h_t \\Vert z_t])',
    text: 'Mean of a unit-variance Gaussian over pixels: two dense layers with ELU, or with the convolutional option a dense layer and two stride-2 transposed convolutions.',
  },
  reward: { tex: '\\hat r_t = \\mathrm{MLP}_{r}([h_t \\Vert z_t])' },
  features: {
    tex: 's_t = [h_t \\Vert z_t]',
    text: 'The model state: deterministic and stochastic parts concatenated, read by every head.',
  },
  loss: {
    tex: 'L = \\frac{1}{BT}\\sum_{b,t} \\Big[\\tfrac12\\lVert \\hat o_t - o_t\\rVert^2 + \\tfrac12(\\hat r_t - r_t)^2 + \\beta \\max\\big(\\mathrm{KL}[q(z_t \\mid h_t, o_t) \\Vert p(z_t \\mid h_t)],\\ \\text{free nats}\\big)\\Big]',
    text: 'Negative evidence lower bound up to constants. Free nats stop the KL term from collapsing the posterior onto the prior early in training.',
  },
};
