import {
  ParamBuilder,
  Tensor,
  add,
  declareLinear,
  defaultOptimizer,
  dense,
  exp,
  graphRng,
  gruCell,
  klStandardNormal,
  mse,
  mul,
  mulScalar,
  p,
  scope,
  select,
  slice,
  stack,
  tanh,
  concat,
  type Var,
} from '../../engine';
import { defineModel } from '../types';
import { PENDULUM, pendulumDataset } from '../data/pendulum';
import { constant, input } from '../util';
import { METHOD_TEX, ODE_METHODS, STAGES, odeStep, type OdeMethod } from './solvers';

type LatentOdeConfig = {
  latent: number;
  encoder: number;
  width: number;
  method: OdeMethod;
  substeps: number;
  sigma: number;
  beta: number;
  noise: number;
  objective: 'reconstruct' | 'extrapolate';
};

const SPEC = PENDULUM;
/** Samples after the observed window that the 'extrapolate' objective is trained to predict. */
const PREDICT = 20;
const GRID = 21;

/** GRU run backwards over y_{K−1}, …, y_0, then a linear map to (μ, log σ²) of q(z₀ | y). */
function encode(y: Var, c: LatentOdeConfig): { mu: Var; logvar: Var } {
  return scope('encoder', () => {
    const [N, K] = y.shape;
    let h = constant(Tensor.zeros([N, c.encoder]), 'h₀');
    const wih = p('encoder.gru.weight_ih');
    const whh = p('encoder.gru.weight_hh');
    const bih = p('encoder.gru.bias_ih');
    const bhh = p('encoder.gru.bias_hh');
    for (let k = K - 1; k >= 0; k--) h = gruCell(select(y, 1, k), h, wih, whh, bih, bhh).h;
    const q = dense(h, 'encoder.out');
    return { mu: slice(q, 1, 0, c.latent), logvar: slice(q, 1, c.latent, 2 * c.latent) };
  });
}

/** Autonomous latent dynamics f_θ(z): two tanh hidden layers. */
function dynamics(z: Var): Var {
  return scope('f', () => dense(tanh(dense(tanh(dense(z, 'f.fc1')), 'f.fc2')), 'f.out'));
}

/** Integrate from z₀ over `intervals` sample intervals; returns the states at the sample times and every solver state. */
function rollout(
  z0: Var,
  intervals: number,
  c: LatentOdeConfig,
  hooks?: { shouldStop(): boolean; progress(f: number): void },
) {
  const field = (z: Var) => dynamics(z);
  const h = SPEC.dt / c.substeps;
  const atSamples: Var[] = [z0];
  const all: Var[] = [z0];
  let z = z0;
  for (let k = 0; k < intervals; k++) {
    if (hooks?.shouldStop()) break;
    for (let s = 0; s < c.substeps; s++) {
      const prev = z;
      z = odeStep(field, prev, (k * c.substeps + s) * h, h, c.method);
      all.push(z);
    }
    atSamples.push(z);
    hooks?.progress((k + 1) / intervals);
  }
  return {
    atSamples,
    all,
    nfe: atSamples.length > 1 ? (atSamples.length - 1) * c.substeps * STAGES[c.method] : 0,
  };
}

/** The first P time steps of a [N, T, 2] tensor. */
function firstSteps(t: Tensor, P: number): Tensor {
  const [N, T, D] = t.shape;
  const out = new Float32Array(N * P * D);
  for (let n = 0; n < N; n++) out.set(t.data.subarray(n * T * D, (n * T + P) * D), n * P * D);
  return new Tensor([N, P, D], out);
}

function decode(zs: Var): Var {
  return scope('decoder', () => dense(tanh(dense(zs, 'decoder.fc1')), 'decoder.out'));
}

const latentOde = defineModel<LatentOdeConfig>({
  id: 'latent-ode',
  family: 'neural-dynamics',
  name: 'Latent ODE',
  summary:
    'A recognition network reads an observed trajectory and infers an initial latent state; an ODE in latent space carries it forward, and a decoder maps the latent path back to observations, also beyond the observed window.',
  signature:
    'z_0 \\sim q_\\phi(z_0\\mid y_{0:K-1}),\\quad \\frac{dz}{dt} = f_\\theta(z),\\quad y_k \\sim \\mathcal N(g_\\psi(z(t_k)),\\ \\sigma^2 I)',
  fidelity: 'reduced',
  scaleNote:
    'The generative model and variational objective follow the latent ODE of Chen et al. (2018, §5): an RNN run backwards over the observations gives q(z₀), the latent path is an ODE solution, and the decoder is applied at the observation times; the four-dimensional latent space is the paper’s. Reduced: the recognition RNN is a GRU (the ODE-RNN encoder of Rubanova et al. 2019 is not implemented), observations are regularly sampled, and the solvers are fixed-step with backpropagation through the steps instead of adaptive with the adjoint.',
  references: [
    {
      authors: 'R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud',
      title: 'Neural ordinary differential equations',
      year: 2018,
      url: 'https://arxiv.org/abs/1806.07366',
    },
    {
      authors: 'Y. Rubanova, R. T. Q. Chen, D. Duvenaud',
      title: 'Latent ODEs for irregularly-sampled time series',
      year: 2019,
      url: 'https://arxiv.org/abs/1907.03907',
    },
    {
      authors: 'D. P. Kingma, M. Welling',
      title: 'Auto-encoding variational Bayes',
      year: 2014,
      url: 'https://arxiv.org/abs/1312.6114',
    },
  ],
  limitations: [
    `Observations are regularly sampled (every ${SPEC.dt} s); the irregular sampling that motivates Rubanova et al. is not exercised.`,
    'The encoder is a GRU, not an ODE-RNN, so the recognition network has no continuous-time dynamics between observations.',
    'Gradients come from backpropagating through the fixed-step solver; the adjoint method is not implemented.',
    'The observation noise σ in the likelihood is a fixed hyperparameter, not learned.',
    'With a latent dimension above two (the default is four), the latent phase portrait shows the first two coordinates, with the field evaluated where the remaining coordinates are zero; paths can therefore cross in the picture. Set the latent dimension to 2 for an exact portrait; whether that size trains within a few hundred steps depends on the seed.',
    'Gradient checks use central differences with ε = 1e-4 instead of 1e-6: with σ = 0.05 the likelihood scales squared errors by 1/(2σ²) = 200, and float64 round-off in L(θ ± ε) then dominates the difference quotient at ε = 1e-6.',
    'The latent space is not identified: any invertible reparameterisation of z gives the same likelihood, so the latent coordinates need not resemble (θ, θ′).',
  ],
  guide: {
    what: `The input is ${SPEC.observed} noisy samples of a swinging pendulum’s bob position (x, y), taken every ${SPEC.dt} s. A GRU reads them backwards and outputs a Gaussian guess of the initial latent state z₀, a four-number code for the state of the motion. A learned differential equation then carries z forward in time and a decoder turns each z(t) back into a position, also beyond the ${SPEC.observed} samples; training fits the observed and the next ${PREDICT} samples while keeping z₀ close to a standard normal.`,
    watch: `There is no checkpoint: training starts from random weights. The loss falls by a factor of about four in the first 25 steps while the decoded path is still almost flat (the model first learns the average position), then keeps falling, unevenly, to about 2 by step 600 as the path starts to swing with the pendulum. Run “Reconstruct and extrapolate”: around step 200 the model fits the ${SPEC.observed} observed samples but drifts afterwards; by step 600 it also follows the simulation through samples ${SPEC.observed}–${SPEC.observed + PREDICT - 1} (MSE 0.005–0.011 over eight validation trajectories, seeds 1–3). Beyond sample ${SPEC.observed + PREDICT}, never a training target, it holds in only one of three seeds; in seeds 1 and 3 its MSE there (0.04 and 0.06) is about that of a flat line (0.05).`,
    try: [
      {
        label: 'Window-only objective',
        note: `Trained on the observed window alone, as in Chen et al.: after 300 steps the fit inside the window is tighter than with the default target (MSE 0.009 against 0.030), but past sample ${SPEC.observed} the path runs away, with MSE 0.36 over the next ${PREDICT} samples and 0.97 beyond, worse than a flat line (about 0.11 and 0.05).`,
        patch: { objective: 'reconstruct' },
        train: 300,
      },
      {
        label: 'Likelihood σ = 0.5',
        note: 'The data term is 100× weaker, so the prior on z₀ weighs far more: after 200 steps the posterior stays much closer to the standard normal (mean σ about 0.5 against 0.07 with σ = 0.05; KL 3 against 23), and the observed window is fitted more loosely (MSE 0.077 against 0.045).',
        patch: { sigma: 0.5 },
        train: 200,
      },
      {
        label: 'Two latent dimensions',
        note: `Parameters re-initialise. The latent portrait is now exact rather than a projection, but whether this size trains in 300 steps depends on the seed. With this page’s seed 1 the decoded path is still almost flat (MSE 0.11 over samples ${SPEC.observed}–${SPEC.observed + PREDICT - 1}) and the latent paths do not spiral; with seed 2 the fit is close (MSE 0.001) and each path spirals inwards about four times, once per swing, as the pendulum damps.`,
        patch: { latent: 2 },
        train: 300,
      },
    ],
    views: [
      'Training target and model output (inspected example)',
      `Validation trajectory: simulated, model, and the ${SPEC.observed} noisy samples given to the encoder`,
    ],
  },
  configSchema: [
    {
      key: 'latent',
      label: 'Latent dimension',
      kind: 'int',
      min: 2,
      max: 6,
      affects: 'architecture',
    },
    {
      key: 'encoder',
      label: 'Encoder GRU width',
      kind: 'int',
      min: 8,
      max: 64,
      step: 8,
      affects: 'architecture',
    },
    {
      key: 'width',
      label: 'Width of f and decoder',
      kind: 'int',
      min: 8,
      max: 64,
      step: 8,
      affects: 'architecture',
    },
    { key: 'method', label: 'Solver', kind: 'select', options: ODE_METHODS, affects: 'forward' },
    {
      key: 'substeps',
      label: 'Solver steps per sample interval',
      kind: 'int',
      min: 1,
      max: 4,
      affects: 'forward',
    },
    {
      key: 'sigma',
      label: 'Likelihood σ',
      kind: 'float',
      min: 0.01,
      max: 0.5,
      step: 0.01,
      affects: 'forward',
    },
    {
      key: 'beta',
      label: 'KL weight',
      kind: 'float',
      min: 0,
      max: 1,
      step: 0.05,
      affects: 'forward',
      help: '1 gives the evidence lower bound; smaller values down-weight the prior on z₀.',
    },
    {
      key: 'noise',
      label: 'Observation noise',
      kind: 'float',
      min: 0,
      max: 0.2,
      step: 0.01,
      affects: 'data',
    },
    {
      key: 'objective',
      label: 'Training target',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'extrapolate', label: `Observed window + next ${PREDICT} samples` },
        { value: 'reconstruct', label: 'Observed window only' },
      ],
      help: 'The encoder always sees only the observed window. Reconstruction alone (Chen et al.) tends to fit the window without learning dynamics that extrapolate; predicting later samples from the same z₀ is the extrapolation setup of Rubanova et al.',
    },
  ],
  defaults: {
    latent: 4,
    encoder: 32,
    width: 32,
    method: 'rk4',
    substeps: 1,
    sigma: 0.05,
    beta: 1,
    noise: 0.02,
    objective: 'extrapolate',
  },
  init(c, rng) {
    const b = new ParamBuilder(rng);
    const H = c.encoder;
    b.add('encoder.gru.weight_ih', [3 * H, 2], { kind: 'lecunUniform', fanIn: H });
    b.add('encoder.gru.weight_hh', [3 * H, H], { kind: 'lecunUniform', fanIn: H });
    b.add('encoder.gru.bias_ih', [3 * H], { kind: 'lecunUniform', fanIn: H });
    b.add('encoder.gru.bias_hh', [3 * H], { kind: 'lecunUniform', fanIn: H });
    declareLinear(b, 'encoder.out', H, 2 * c.latent);
    declareLinear(b, 'f.fc1', c.latent, c.width);
    declareLinear(b, 'f.fc2', c.width, c.width);
    declareLinear(b, 'f.out', c.width, c.latent);
    declareLinear(b, 'decoder.fc1', c.latent, c.width);
    declareLinear(b, 'decoder.out', c.width, 2);
    return b.store;
  },
  datasets: [pendulumDataset<LatentOdeConfig>((c) => c.noise)],
  forward(batch, c) {
    const y = input('y', batch.inputs.y, false);
    const [N, K] = y.shape;
    const { mu, logvar } = encode(y, c);
    const z0 = scope('sample', () => {
      const rng = graphRng();
      const e = new Tensor(
        [N, c.latent],
        Float32Array.from({ length: N * c.latent }, () => rng.normal()),
      );
      return add(mu, mul(exp(mulScalar(logvar, 0.5)), constant(e, 'ε')));
    });
    const P = c.objective === 'extrapolate' ? PREDICT : 0;
    const H = K + P;
    const target =
      P > 0
        ? scope('target', () =>
            concat([y, constant(firstSteps(batch.inputs.future, P), 'y_future')], 1),
          )
        : y;
    const path = scope('ode', () => rollout(z0, H - 1, c));
    const zs = scope('trajectory', () => stack(path.atSamples, 1));
    const yhat = decode(zs);
    const recon = scope('reconstruction', () => mse(yhat, target));
    const kl = scope('kl', () => klStandardNormal(mu, logvar));
    const loss = scope('loss', () =>
      add(mulScalar(recon, 1 / (2 * c.sigma * c.sigma)), mulScalar(kl, c.beta / (H * 2))),
    );
    return {
      outputs: { observed: y, target, recon: yhat, mu, logvar, z: zs },
      loss,
      metrics: { mse: recon.value.data[0], kl: kl.value.data[0], NFE: path.nfe },
    };
  },
  // Finite differences: the 1/(2σ²) = 200 likelihood factor at σ = 0.05 amplifies float64 round-off in
  // L(θ ± ε) (~1e-15 relative after 20 GRU and 39 RK4 steps) to ≈1e-4 relative gradient error at ε = 1e-6.
  // At ε = 1e-4 the round-off is 100× smaller and the O(ε²) truncation error is still far below 1e-5.
  gradCheck: { eps: 1e-4 },
  training: {
    support: 'full',
    note: 'Gradients are obtained by backpropagating through the solver steps, not by the adjoint method.',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 0.01 },
    batchSize: 16,
    steps: 600,
    validateEvery: 50,
  },
  procedures: [
    {
      id: 'extrapolate',
      label: 'Reconstruct and extrapolate',
      description: `Encodes the first ${SPEC.observed} noisy samples of a validation trajectory with the posterior mean of z₀, integrates the latent ODE over ${SPEC.observed + SPEC.future} samples and decodes every state. Also integrates eight validation trajectories in latent space and evaluates f_θ on a 21 × 21 grid.`,
      args: [
        {
          key: 'example',
          label: 'Validation trajectory',
          kind: 'int',
          min: 0,
          max: 63,
          affects: 'data',
        },
      ],
      defaults: { example: 0 },
      run(ctx) {
        const c = ctx.config;
        const L = SPEC.observed + SPEC.future;
        const idx = Math.max(0, Math.min(63, Math.round(Number(ctx.args.example))));
        const val = ctx.dataset.validation!(c);
        const pick = (t: Tensor, i: number) => {
          const per = t.size / t.shape[0];
          return new Tensor(t.shape.slice(1), t.data.slice(i * per, (i + 1) * per));
        };
        // Main trajectory plus eight for the latent portrait (indices 0..7, or the chosen one first).
        const ids = [idx, ...[0, 1, 2, 3, 4, 5, 6, 7].filter((i) => i !== idx)].slice(0, 8);
        const yObs = new Tensor(
          [ids.length, SPEC.observed, 2],
          new Float32Array(ids.length * SPEC.observed * 2),
        );
        ids.forEach((i, r) => yObs.data.set(pick(val.inputs.y, i).data, r * SPEC.observed * 2));
        const { mu } = encode(constant(yObs, 'y'), c);
        const path = rollout(mu, L - 1, c, ctx);
        const decoded = decode(stack(path.atSamples, 1)).value;
        const T = path.atSamples.length;
        const predicted = new Tensor([T, 2], decoded.data.slice(0, T * 2));
        const truth = pick(val.inputs.clean, idx);
        const observed = pick(val.inputs.y, idx);
        const err = [0, 0, 0];
        const cnt = [0, 0, 0];
        for (let k = 0; k < Math.min(T, L); k++)
          for (let d = 0; d < 2; d++) {
            const w = k < SPEC.observed ? 0 : k < SPEC.observed + PREDICT ? 1 : 2;
            err[w] += (predicted.data[k * 2 + d] - truth.data[k * 2 + d]) ** 2;
            cnt[w]++;
          }
        const [mseObs, mseNext, mseBeyond] = err.map((e, i) => e / Math.max(cnt[i], 1));
        const S = path.all.length;
        const traj = new Float32Array(ids.length * S * 2);
        for (let r = 0; r < ids.length; r++)
          for (let s = 0; s < S; s++) {
            traj[(r * S + s) * 2] = path.all[s].value.data[r * c.latent];
            traj[(r * S + s) * 2 + 1] = path.all[s].value.data[r * c.latent + 1];
          }
        // Square extent fitted to the latent paths (10 % margin); the field is evaluated on the same extent.
        let lo0 = Infinity,
          hi0 = -Infinity,
          lo1 = Infinity,
          hi1 = -Infinity;
        for (let i = 0; i < traj.length; i += 2) {
          lo0 = Math.min(lo0, traj[i]);
          hi0 = Math.max(hi0, traj[i]);
          lo1 = Math.min(lo1, traj[i + 1]);
          hi1 = Math.max(hi1, traj[i + 1]);
        }
        const half = (Math.max(hi0 - lo0, hi1 - lo1, 1e-3) / 2) * 1.1;
        const cx = (lo0 + hi0) / 2;
        const cy = (lo1 + hi1) / 2;
        const [x0, x1, y0, y1] = [cx - half, cx + half, cy - half, cy + half];
        const g = new Float32Array(GRID * GRID * c.latent);
        for (let i = 0; i < GRID; i++)
          for (let j = 0; j < GRID; j++) {
            g[(i * GRID + j) * c.latent] = x0 + ((j + 0.5) / GRID) * (x1 - x0);
            g[(i * GRID + j) * c.latent + 1] = y0 + ((i + 0.5) / GRID) * (y1 - y0);
          }
        const fv = dynamics(constant(new Tensor([GRID * GRID, c.latent], g), 'grid')).value;
        const field = new Float32Array(GRID * GRID * 2);
        for (let i = 0; i < GRID * GRID; i++) {
          field[i * 2] = fv.data[i * c.latent];
          field[i * 2 + 1] = fv.data[i * c.latent + 1];
        }
        return {
          tensors: {
            truth,
            observed,
            predicted,
            trajectories: new Tensor([ids.length, S, 2], traj),
            field: new Tensor([GRID, GRID, 2], field),
            extent: new Tensor([4], Float32Array.from([x0, x1, y0, y1])),
          },
          scalars: {
            'MSE vs. noise-free simulation, observed window': mseObs,
            [`MSE vs. noise-free simulation, samples ${SPEC.observed}–${SPEC.observed + PREDICT - 1}`]:
              mseNext,
            [`MSE vs. noise-free simulation, samples ${SPEC.observed + PREDICT}–${L - 1} (never a training target)`]:
              mseBeyond,
            NFE: path.nfe,
          },
        };
      },
    },
  ],
  views: [
    {
      kind: 'series',
      title: `Validation trajectory: simulated, model, and the ${SPEC.observed} noisy samples given to the encoder`,
      sources: [
        { ref: 'proc:extrapolate:truth', label: 'simulated (noise-free)' },
        {
          ref: 'proc:extrapolate:predicted',
          label: 'latent ODE (reconstruction, then extrapolation)',
        },
        { ref: 'proc:extrapolate:observed', label: 'observed input' },
      ],
      xLabel: `sample k (Δ = ${SPEC.dt} s; k < ${SPEC.observed} observed, k ≥ ${SPEC.observed + PREDICT} never a training target)`,
    },
    {
      kind: 'phase',
      title:
        'Latent vector field f_θ(z) and latent paths of eight validation trajectories (first two latent coordinates)',
      field: 'proc:extrapolate:field',
      trajectories: 'proc:extrapolate:trajectories',
      extent: 'proc:extrapolate:extent',
      note: 'Arrows: f_θ(z) on a 21 × 21 grid over a square fitted to the paths shown (with more than two latent dimensions, on the plane where the other coordinates are zero). Curves: every solver state of the latent paths z(t) over the 60-sample horizon, starting from the posterior mean of z₀.',
    },
    {
      kind: 'series',
      title: 'Training target and model output (inspected example)',
      sources: [
        { ref: 'out:target', label: 'target (noisy)' },
        { ref: 'out:recon', label: 'decoded latent path' },
      ],
      xLabel: 'sample k',
    },
    {
      kind: 'series',
      title: 'Latent path z(t_k) (inspected example)',
      sources: [{ ref: 'out:z', label: 'z' }],
      xLabel: 'sample k',
    },
  ],
  math: {
    encoder: {
      tex: 'h_k = \\mathrm{GRU}(y_k, h_{k+1}),\\ k = K-1,\\dots,0;\\quad [\\mu;\\ \\log\\sigma_0^2] = W h_0 + b',
      text: 'The recognition RNN reads the observations backwards in time, so its final state summarises the trajectory from its start.',
    },
    sample: {
      tex: 'z_0 = \\mu + \\sigma_0 \\odot \\varepsilon,\\quad \\varepsilon\\sim\\mathcal N(0, I)',
      text: 'Reparameterisation; ε comes from the seeded generator of the step.',
    },
    ode: {
      tex: `z(t_{k+1}) = \\mathrm{ODESolve}(f_\\theta, z(t_k), t_k, t_{k+1}),\\qquad \\text{RK4: } ${METHOD_TEX.rk4}`,
      text: 'Fixed steps of size Δ / substeps; NFE = intervals × substeps × stages.',
    },
    f: { tex: 'f_\\theta(z) = W_3 \\tanh(W_2 \\tanh(W_1 z + b_1) + b_2) + b_3' },
    decoder: { tex: '\\hat y_k = g_\\psi(z(t_k)) = W_2 \\tanh(W_1 z(t_k) + b_1) + b_2' },
    target: {
      tex: 'y_{0:H} = [\\,y_{0:K}\\,;\\ y_{K:K+20}\\,]',
      text: 'The observed window followed by the next 20 simulated samples, which the encoder never sees.',
    },
    trajectory: {
      tex: 'Z = [\\,z(t_0), z(t_1), \\dots, z(t_{H-1})\\,]',
      text: 'The solver states at the sample times, stacked along the time axis for the decoder.',
    },
    reconstruction: {
      tex: '\\mathrm{MSE} = \\tfrac{1}{HD} \\textstyle\\sum_{k,d} (y_{k,d} - \\hat y_{k,d})^2',
    },
    kl: {
      tex: 'D_{\\mathrm{KL}} = \\tfrac12 \\textstyle\\sum_{j} \\big(\\mu_j^2 + \\sigma_j^2 - \\log \\sigma_j^2 - 1\\big)',
      text: 'Closed form for the diagonal Gaussian posterior of z₀ against the standard normal prior, averaged over the batch.',
    },
    loss: {
      tex: 'L = \\frac{1}{HD}\\Big[\\sum_{k<H,\\,d} \\frac{(y_{k,d} - \\hat y_{k,d})^2}{2\\sigma^2} + \\beta\\,\\mathrm{KL}\\big(q(z_0\\mid y)\\,\\|\\,\\mathcal N(0, I)\\big)\\Big]',
      text: 'H = K (observed window) or K + 20 (with the extrapolation target). With β = 1 and H = K this is the negative evidence lower bound per coordinate, up to a constant.',
    },
  },
});

export default latentOde;
