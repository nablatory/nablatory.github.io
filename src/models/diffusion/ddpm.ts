import { defaultOptimizer } from '../../engine';
import { defineModel } from '../types';
import { schedule } from './schedule';
import {
  TOY_EXTENT,
  ancestralSample,
  clampT,
  diffuseProcedure,
  gaussian,
  initToyNet,
  mmd2,
  overlay,
  predictEps,
  scoreField,
  toyConfigSchema,
  toyData,
  toyDataset,
  toyDefaults,
  toyForward,
  toyMath,
  toyViews,
  validateToy,
  SCORE_NOTE,
  type ToyConfig,
} from './toy';

type DdpmConfig = ToyConfig & { variance: 'beta' | 'betaTilde' };

export const TOY_CHECKPOINT = {
  id: 'moons',
  label: 'Trained on two moons',
  path: 'weights/ddpm-toy-moons',
  description:
    'Default network trained offline on the two-moons distribution (T = 100, linear schedule) by scripts/train/ddpm-toy.ts.',
};

const ddpm = defineModel<DdpmConfig>({
  id: 'ddpm',
  family: 'diffusion',
  name: 'Denoising diffusion probabilistic model',
  summary:
    'Data is destroyed by T small Gaussian noising steps; a network trained to predict the added noise runs the chain backwards to generate new points.',
  signature:
    'L = \\mathbb E_{t,x_0,\\varepsilon}\\big\\|\\varepsilon - \\varepsilon_\\theta(\\sqrt{\\bar\\alpha_t}x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon,\\ t)\\big\\|^2',
  fidelity: 'reduced',
  scaleNote:
    'The forward process, the ε-prediction objective L_simple, the sinusoidal time embedding (applied to the rescaled time 1000·t/T) and ancestral sampling are as published. The data are two-dimensional point clouds instead of images, the denoiser is a residual MLP (30,530 parameters at the default size) instead of a U-Net with attention, and the default T is 100 instead of 1,000 (β rescaled by 1000/T).',
  references: [
    {
      authors: 'J. Ho, A. Jain, P. Abbeel',
      title: 'Denoising diffusion probabilistic models',
      year: 2020,
      url: 'https://arxiv.org/abs/2006.11239',
    },
    {
      authors: 'J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, S. Ganguli',
      title: 'Deep unsupervised learning using nonequilibrium thermodynamics',
      year: 2015,
      url: 'https://arxiv.org/abs/1503.03585',
    },
    {
      authors: 'A. Nichol, P. Dhariwal',
      title: 'Improved denoising diffusion probabilistic models',
      year: 2021,
      url: 'https://arxiv.org/abs/2102.09672',
    },
  ],
  limitations: [
    'Two-dimensional toy data only, so that samples, noised data, trajectories and the learned score can be drawn exactly.',
    'The network is an MLP, not the U-Net of the paper; there is no attention, group normalisation or dropout.',
    'The variance is fixed (β_t or β̃_t); the learned variances and hybrid loss of Nichol & Dhariwal are not implemented.',
    'Sample quality is measured by the squared maximum mean discrepancy (Gaussian kernel, bandwidth 0.3; the biased estimator that includes the diagonal terms) against 400 data points, not by FID or likelihood.',
    'The shipped checkpoint was trained with T = 100 and the linear schedule. Changing T or the schedule keeps the weights, which then see noise levels they were not trained on.',
  ],
  guide: {
    what: 'A generative model for points in the plane. Training takes a data point, picks a random step t out of T = 100, mixes the point with random Gaussian noise in a proportion fixed by the noise schedule, and asks a small network to guess the noise that was added. Sampling starts from pure noise and applies that guess backwards for T steps, turning noise into a new point that looks like the data. The only thing learned is the noise predictor ε_θ(x_t, t).',
    watch:
      'Opens with weights trained offline on two moons: run “Ancestral sampling” to see samples over the data. “Start from random weights” trains from scratch; the loss (squared error in predicting the noise) falls from about 1 to under 0.3 within 100 steps, but samples stay a blob for hundreds of steps and need one to two thousand to form crescents.',
    try: [
      {
        label: 'Cosine schedule',
        note: 'The weights were trained with the linear schedule, so samples first degrade (MMD² about 0.02); 300 steps of training recover them.',
        patch: { schedule: 'cosine' },
        train: 300,
      },
      {
        label: 'Two spirals',
        note: 'The samples still look like moons until training reshapes them; after 300 steps MMD² falls from about 0.05 to 0.01.',
        patch: { dataset: 'spirals' },
        train: 300,
      },
      {
        label: 'Twenty reverse steps',
        note: 'The same weights with β rescaled for T = 20: sampling is five times faster and the samples only slightly worse.',
        patch: { T: 20 },
      },
    ],
    views: ['Noise schedule: β_t and ᾱ_t', 'Data (blue) and DDPM samples (red)'],
  },
  configSchema: [
    ...toyConfigSchema,
    {
      key: 'variance',
      label: 'Reverse variance σ_t²',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'beta', label: 'σ_t² = β_t' },
        { value: 'betaTilde', label: 'σ_t² = β̃_t (posterior)' },
      ],
      help: 'Ho et al. report similar results for both; β_t is the upper and β̃_t the lower bound on the reverse-process entropy.',
    },
  ],
  defaults: { ...toyDefaults, variance: 'beta' },
  validate: validateToy,
  init: (c, rng) => initToyNet(c, rng).store,
  datasets: [toyDataset<DdpmConfig>()],
  forward: (batch, c) => toyForward(batch, c),
  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 2e-3 },
    batchSize: 128,
    steps: 600,
    validateEvery: 50,
  },
  procedures: [
    diffuseProcedure<DdpmConfig>(),
    {
      id: 'sample',
      label: 'Ancestral sampling',
      description:
        'Draws x_T ~ N(0, I) and runs all T reverse steps of the learned Markov chain (T network evaluations per point). Also evaluates the learned score at the chosen t on a 21 × 21 grid.',
      args: [
        {
          key: 'count',
          label: 'Samples',
          kind: 'int',
          min: 16,
          max: 1000,
          step: 16,
          affects: 'forward',
        },
        {
          key: 'fieldT',
          label: 'Score field at t',
          kind: 'int',
          min: 1,
          max: 1000,
          affects: 'forward',
        },
      ],
      defaults: { count: 400, fieldT: 5 },
      run(ctx) {
        const c = ctx.config;
        const s = schedule(c.schedule, c.T);
        const n = Number(ctx.args.count);
        const xT = gaussian([n, 2], ctx.rng);
        const chain = ancestralSample((x, t) => predictEps(x, t, c), xT, s, c.variance, ctx.rng, {
          track: 24,
          hooks: ctx,
        });
        const fieldT = clampT(Number(ctx.args.fieldT), c.T);
        const { points, labels } = overlay(c, chain.x);
        const data = toyData(c, 'show').x;
        return {
          tensors: {
            points,
            labels,
            samples: chain.x,
            trajectories: chain.trajectories!,
            field: scoreField(c, fieldT),
          },
          scalars: {
            'MMD² (samples, data)': mmd2(chain.x.data, data, 2),
            'MMD² (x_T, data)': mmd2(xT.data, data, 2),
            NFE: chain.nfe,
            ...(chain.stoppedAt !== null ? { 'stopped at t': chain.stoppedAt } : {}),
          },
        };
      },
    },
  ],
  views: [
    {
      kind: 'scatter',
      title: 'Data (blue) and DDPM samples (red)',
      source: 'proc:sample:points',
      colorBy: 'proc:sample:labels',
      extent: TOY_EXTENT,
    },
    {
      kind: 'phase',
      title: 'Learned score s_θ(x, t) and 24 reverse-chain trajectories from x_T to x₀',
      field: 'proc:sample:field',
      trajectories: 'proc:sample:trajectories',
      extent: TOY_EXTENT,
      note: SCORE_NOTE,
    },
    ...toyViews,
  ],
  math: {
    ...toyMath,
    sample: {
      tex: 'x_{t-1} = \\frac{1}{\\sqrt{\\alpha_t}}\\Big(x_t - \\frac{\\beta_t}{\\sqrt{1-\\bar\\alpha_t}}\\,\\varepsilon_\\theta(x_t,t)\\Big) + \\sigma_t z,\\quad \\sigma_t^2\\in\\{\\beta_t,\\ \\tilde\\beta_t\\}',
      text: 'β̃_t = (1 − ᾱ_{t−1})/(1 − ᾱ_t) · β_t is the variance of the true posterior q(x_{t−1} | x_t, x₀).',
    },
  },
  checkpoints: [TOY_CHECKPOINT],
});

export default ddpm;
