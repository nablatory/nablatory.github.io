import { defaultOptimizer, Tensor } from '../../engine';
import { defineModel, type ViewSpec } from '../types';
import { ddimTimesteps, schedule } from './schedule';
import {
  TOY_EXTENT,
  clampT,
  ddimSample,
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
import { TOY_CHECKPOINT } from './ddpm';

type DdimConfig = ToyConfig & { eta: number; steps: number };

/** Step counts drawn as scatters; `0` stands for the full sequence (S = T). */
const SWEEP = [2, 5, 10, 0];
/** Step counts for the MMD² curve (those above T are dropped; T is always included). */
const CURVE = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000];

const sweepViews: ViewSpec[] = SWEEP.map((S) => ({
  kind: 'scatter' as const,
  title: S === 0 ? 'Same x_T, all T steps' : `Same x_T, ${S} steps`,
  source: `proc:sweep:steps${S || 'T'}`,
  colorBy: 'proc:sweep:labels',
  extent: TOY_EXTENT,
}));

const ddim = defineModel<DdimConfig>({
  id: 'ddim',
  family: 'diffusion',
  name: 'Denoising diffusion implicit model',
  summary:
    'A non-Markovian sampler for a DDPM-trained noise predictor: it skips timesteps, and with η = 0 it is a deterministic map from noise to data.',
  signature:
    'x_{\\tau_{i-1}} = \\sqrt{\\bar\\alpha_{\\tau_{i-1}}}\\,\\hat x_0 + \\sqrt{1-\\bar\\alpha_{\\tau_{i-1}}-\\sigma_{\\tau_i}^2}\\,\\varepsilon_\\theta(x_{\\tau_i},\\tau_i) + \\sigma_{\\tau_i} z',
  fidelity: 'reduced',
  scaleNote:
    'The DDIM update (Song et al., eq. 12), the η-family of noise scales (eq. 16) and evenly spaced timestep sub-sequences are as published. The noise predictor is the same two-dimensional residual MLP as the DDPM page, trained with the same L_simple objective; it is not an image U-Net.',
  references: [
    {
      authors: 'J. Song, C. Meng, S. Ermon',
      title: 'Denoising diffusion implicit models',
      year: 2021,
      url: 'https://arxiv.org/abs/2010.02502',
    },
    {
      authors: 'J. Ho, A. Jain, P. Abbeel',
      title: 'Denoising diffusion probabilistic models',
      year: 2020,
      url: 'https://arxiv.org/abs/2006.11239',
    },
  ],
  limitations: [
    'Training is identical to DDPM (DDIM changes only the sampler). This page trains its own copy of the network in the browser and loads the same offline checkpoint file as the DDPM page.',
    'Only evenly spaced sub-sequences τ are offered; the quadratic spacing used for CIFAR-10 in the paper is not.',
    'The predicted x̂₀ is not clipped (there is no natural data range for the toy distributions).',
    'Two-dimensional data only; sample quality is measured by MMD² against 400 data points, not FID.',
  ],
  guide: {
    what: 'The same noise predictor as the DDPM page, run with a different sampler. Instead of visiting all T noise levels, DDIM jumps along a short sub-sequence of S = 20 levels: at each one it estimates the clean point x̂₀ from the current noisy point and moves it to the next level. With η = 0 no new noise is added, so each starting noise x_T maps to one fixed sample. Nothing new is learned: training is identical to DDPM.',
    watch:
      'Opens with the DDPM checkpoint. Run “DDIM sampling”, then “Steps and η from one x_T” to compare samples from the same x_T after 2, 5, 10 and all 100 steps. “Start from random weights” trains the predictor with the DDPM loss, which falls from about 1 to under 0.3 within 100 steps; good samples need a thousand steps or more.',
    try: [
      {
        label: 'Two steps only',
        note: 'Two network evaluations per sample give a poor estimate of x̂₀: MMD² about 0.4 against 0.01 with 20 steps.',
        patch: { steps: 2 },
      },
      {
        label: 'Stochastic, η = 1',
        note: 'Noise is injected at each of the 20 steps: sample quality is about the same, but a chain no longer depends only on its x_T.',
        patch: { eta: 1 },
      },
      {
        label: 'Full chain, S = T',
        note: 'All 100 levels with η = 0: five times the cost of S = 20 for a barely lower MMD² (0.010 against 0.012).',
        patch: { steps: 100 },
      },
    ],
    views: [
      'Sample quality against the number of DDIM steps (same x_T)',
      'Data (blue) and DDIM samples (red)',
    ],
  },
  configSchema: [
    ...toyConfigSchema,
    {
      key: 'steps',
      label: 'Sampling steps S',
      kind: 'int',
      min: 1,
      max: 1000,
      affects: 'forward',
      help: 'Length of the timestep sub-sequence τ; S = T visits every step.',
    },
    {
      key: 'eta',
      label: 'η (stochasticity)',
      kind: 'float',
      min: 0,
      max: 1,
      step: 0.1,
      affects: 'forward',
      help: 'η = 0: deterministic DDIM. η = 1 with S = T: the DDPM reverse process with σ_t² = β̃_t.',
    },
  ],
  defaults: { ...toyDefaults, steps: 20, eta: 0 },
  validate(c) {
    const v = validateToy(c);
    if (v) return v;
    if (c.steps > c.T)
      return `A sub-sequence of the T = ${c.T} timesteps has at most ${c.T} steps (got S = ${c.steps}).`;
    return null;
  },
  init: (c, rng) => initToyNet(c, rng).store,
  datasets: [toyDataset<DdimConfig>()],
  forward: (batch, c) => toyForward(batch, c),
  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 2e-3 },
    batchSize: 128,
    steps: 600,
    validateEvery: 50,
  },
  procedures: [
    diffuseProcedure<DdimConfig>(),
    {
      id: 'sample',
      label: 'DDIM sampling',
      description:
        'Draws x_T ~ N(0, I) and applies S DDIM updates with the configured η (S network evaluations per point). With η = 0 the trajectories are Euler steps of the probability-flow ODE in the variables x/√ᾱ.',
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
        const taus = ddimTimesteps(c.T, c.steps);
        const chain = ddimSample((x, t) => predictEps(x, t, c), xT, s, taus, c.eta, ctx.rng, {
          track: 24,
          hooks: ctx,
        });
        const { points, labels } = overlay(c, chain.x);
        const data = toyData(c, 'show').x;
        return {
          tensors: {
            points,
            labels,
            samples: chain.x,
            trajectories: chain.trajectories!,
            field: scoreField(c, clampT(Number(ctx.args.fieldT), c.T)),
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
    {
      id: 'sweep',
      label: 'Steps and η from one x_T',
      description:
        'Starts 400 chains from one fixed draw of x_T and samples with S = 1, 2, 3, 5, 10, 20, 50, … up to T steps at the configured η (MMD² against the data for each S), then with η = 0 and η = 1 at the configured S.',
      run(ctx) {
        const c = ctx.config;
        const s = schedule(c.schedule, c.T);
        const xT = gaussian([400, 2], ctx.rng.fork('xT'));
        const data = toyData(c, 'show').x;
        const eps = (x: Tensor, t: number) => predictEps(x, t, c);
        const curve = [...new Set([...CURVE.filter((S) => S < c.T), c.T])];
        const runs: { key: string | null; S: number; eta: number; label: string }[] = [
          ...curve.map((S) => {
            const shown = SWEEP.find((k) => (k || c.T) === S);
            return {
              key: shown !== undefined ? `steps${shown || 'T'}` : null,
              S,
              eta: c.eta,
              label: `S = ${S}`,
            };
          }),
          { key: 'eta0', S: c.steps, eta: 0, label: `η = 0, S = ${c.steps}` },
          { key: 'eta1', S: c.steps, eta: 1, label: `η = 1, S = ${c.steps}` },
        ];
        const total = runs.reduce((a, r) => a + Math.min(r.S, c.T), 0);
        let done = 0;
        const tensors: Record<string, Tensor> = {};
        const scalars: Record<string, number> = {};
        const curveS: number[] = [];
        const curveMmd: number[] = [];
        for (const r of runs) {
          if (ctx.shouldStop()) break;
          const taus = ddimTimesteps(c.T, r.S);
          const chain = ddimSample(eps, xT, s, taus, r.eta, ctx.rng.fork(r.label), {
            hooks: {
              shouldStop: ctx.shouldStop,
              progress: (f) => ctx.progress((done + f * taus.length) / total),
            },
          });
          done += taus.length;
          const m = mmd2(chain.x.data, data, 2);
          if (r.key) {
            const o = overlay(c, chain.x);
            tensors[r.key] = o.points;
            tensors.labels = o.labels;
          }
          if (r.key?.startsWith('eta')) scalars[`MMD² ${r.label}`] = m;
          else {
            curveS.push(taus.length);
            curveMmd.push(m);
          }
        }
        const floor = mmd2(toyData(c, 'val').x, data, 2);
        tensors.S = new Tensor([curveS.length], Float32Array.from(curveS));
        tensors.mmd = new Tensor([curveS.length], Float32Array.from(curveMmd));
        tensors.floor = new Tensor([curveS.length], new Float32Array(curveS.length).fill(floor));
        scalars['MMD² (held-out data)'] = floor;
        return { tensors, scalars };
      },
    },
  ],
  views: [
    {
      kind: 'scatter',
      title: 'Data (blue) and DDIM samples (red)',
      source: 'proc:sample:points',
      colorBy: 'proc:sample:labels',
      extent: TOY_EXTENT,
    },
    {
      kind: 'phase',
      title: 'Learned score s_θ(x, t) and 24 DDIM trajectories (deterministic when η = 0)',
      field: 'proc:sample:field',
      trajectories: 'proc:sample:trajectories',
      extent: TOY_EXTENT,
      note: `${SCORE_NOTE} With η = 0 each chain is a deterministic function of its x_T.`,
    },
    {
      kind: 'series',
      title: 'Sample quality against the number of DDIM steps (same x_T)',
      sources: [
        { ref: 'proc:sweep:mmd', label: 'MMD² (samples, data)' },
        { ref: 'proc:sweep:floor', label: 'MMD² between two data sets' },
      ],
      x: 'proc:sweep:S',
      xLabel: 'steps S',
      logY: true,
      note: 'Gaussian-kernel MMD² (bandwidth 0.3) between 400 samples and 400 data points, at the configured η. The dashed line is the value for a second, independent draw of 400 data points.',
    },
    ...sweepViews,
    {
      kind: 'scatter',
      title: 'Same x_T, η = 0 at S steps',
      source: 'proc:sweep:eta0',
      colorBy: 'proc:sweep:labels',
      extent: TOY_EXTENT,
    },
    {
      kind: 'scatter',
      title: 'Same x_T, η = 1 at S steps',
      source: 'proc:sweep:eta1',
      colorBy: 'proc:sweep:labels',
      extent: TOY_EXTENT,
    },
    ...toyViews,
  ],
  math: {
    ...toyMath,
    sample: {
      tex: '\\hat x_0 = \\frac{x_t - \\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon_\\theta(x_t,t)}{\\sqrt{\\bar\\alpha_t}},\\quad \\sigma_{\\tau_i} = \\eta\\sqrt{\\frac{1-\\bar\\alpha_{\\tau_{i-1}}}{1-\\bar\\alpha_{\\tau_i}}}\\sqrt{1-\\frac{\\bar\\alpha_{\\tau_i}}{\\bar\\alpha_{\\tau_{i-1}}}}',
      text: 'With η = 1 and τ = (1, …, T), σ_t² equals the DDPM posterior variance β̃_t.',
    },
  },
  checkpoints: [
    {
      ...TOY_CHECKPOINT,
      description:
        'The DDPM network trained offline on two moons (T = 100, linear schedule) by scripts/train/ddpm-toy.ts; DDIM reuses it unchanged.',
    },
  ],
});

export default ddim;
