import {
  Graph,
  ParamBuilder,
  Rng,
  Tensor,
  activations,
  add,
  declareLinear,
  defaultOptimizer,
  dense,
  exp,
  flatten,
  graphRng,
  klStandardNormal,
  mul,
  mulScalar,
  reshape,
  scope,
  sigmoid,
  type ActivationName,
  type ParamStore,
  type Var,
} from '../../engine';
import { defineModel, type Batch, type Config } from '../types';
import { constant, input } from '../util';
import { bernoulliNll } from './ops';
import {
  PIXELS,
  SIDE,
  latentProcedure,
  latentView,
  mnistUnit,
  pixelMse,
  unitCustomBatch,
  unitCustomInput,
} from './common';

export type VaeConfig = {
  hidden: number;
  latent: number;
  activation: 'relu' | 'tanh';
  beta: number;
};

function act(c: VaeConfig) {
  return activations[c.activation as ActivationName];
}

function encode(x: Var, c: VaeConfig): { mu: Var; logvar: Var } {
  const h = scope('enc1', () => act(c)(dense(flatten(x), 'enc1')));
  const mu = scope('mu', () => dense(h, 'mu'));
  const logvar = scope('logvar', () => dense(h, 'logvar'));
  return { mu, logvar };
}

/** z = μ + σ ⊙ ε with ε ~ N(0, I) drawn from the graph's generator. */
export function reparameterize(mu: Var, logvar: Var): { z: Var; eps: Var; sigma: Var } {
  const rng = graphRng();
  const e = Tensor.zeros(mu.shape);
  for (let i = 0; i < e.size; i++) e.data[i] = rng.normal();
  const eps = constant(e, 'epsilon');
  const sigma = exp(mulScalar(logvar, 0.5));
  return { z: add(mu, mul(sigma, eps)), eps, sigma };
}

/** Pixel logits [N, 1, 28, 28] of the Bernoulli decoder p(x | z). */
function decode(z: Var, c: VaeConfig): Var {
  const h = scope('dec1', () => act(c)(dense(z, 'dec1')));
  return scope('dec2', () => reshape(dense(h, 'dec2'), [z.shape[0], 1, SIDE, SIDE]));
}

/** Decoder means σ(logits) for a batch of latent vectors (no-grad procedures). */
function decodeMeans(zData: Float32Array, n: number, c: VaeConfig): Tensor {
  const z = constant(new Tensor([n, c.latent], zData), 'z');
  return scope('output', () => sigmoid(decode(z, c))).value;
}

/** Inverse of the standard normal CDF (Acklam's rational approximation, relative error < 1.2e-9). */
export function probit(p: number): number {
  const a = [
    -39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716,
    2.506628277459239,
  ];
  const b = [
    -54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972,
    -13.28068155288572,
  ];
  const cc = [
    -0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734,
    4.374664141464968, 2.938163982698783,
  ];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((cc[0] * q + cc[1]) * q + cc[2]) * q + cc[3]) * q + cc[4]) * q + cc[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (p > 1 - lo) return -probit(1 - p);
  const q = p - 0.5;
  const r = q * q;
  return (
    ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  );
}

/**
 * Monte Carlo estimate of the negative ELBO (nats per image) on a batch:
 * reconstruction term and KL averaged over `samples` independent draws of ε
 * from Rng(seed). Used by the training script and the checkpoint test.
 */
export function vaeNegElbo(
  params: ParamStore,
  config: Config,
  batch: Batch,
  seed = 0,
  samples = 10,
): { negElbo: number; recon: number; kl: number } {
  const rng = new Rng(seed);
  let recon = 0;
  let kl = 0;
  for (let s = 0; s < samples; s++) {
    const g = new Graph({ params, record: false, grad: false, training: false, rng });
    const r = g.run(() => vae.forward(batch, config, { training: false, rng }));
    recon += r.metrics!.recon / samples;
    kl += r.metrics!.kl / samples;
  }
  return { negElbo: recon + kl, recon, kl };
}

const GRID = 10;

const vae = defineModel<VaeConfig>({
  id: 'vae',
  family: 'autoencoder',
  name: 'Variational autoencoder',
  summary:
    'The encoder outputs a Gaussian over codes, a sample is decoded through a Bernoulli likelihood, and training maximises the evidence lower bound.',
  signature:
    '\\mathcal L = \\mathbb E_{q_\\phi(z\\mid x)}[\\log p_\\theta(x\\mid z)] - \\beta\\, D_{\\mathrm{KL}}\\big(q_\\phi(z\\mid x)\\,\\|\\,\\mathcal N(0, I)\\big)',
  fidelity: 'reduced',
  scaleNote:
    'Gaussian MLP encoder and Bernoulli MLP decoder with one hidden layer each and one sample of ε per data point, as in Kingma & Welling (2014, appendix C). The paper used 500 tanh hidden units, Adagrad and minibatches of 100; the default here is 256 ReLU units (tanh is selectable) and Adam. β ≠ 1 gives the β-VAE objective of Higgins et al. (2017).',
  references: [
    {
      authors: 'D. P. Kingma, M. Welling',
      title: 'Auto-encoding variational Bayes',
      year: 2014,
      url: 'https://arxiv.org/abs/1312.6114',
    },
    {
      authors: 'D. J. Rezende, S. Mohamed, D. Wierstra',
      title: 'Stochastic backpropagation and approximate inference in deep generative models',
      year: 2014,
      url: 'https://arxiv.org/abs/1401.4082',
    },
    {
      authors:
        'I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, A. Lerchner',
      title: 'β-VAE: learning basic visual concepts with a constrained variational framework',
      year: 2017,
      url: 'https://openreview.net/forum?id=Sy2fzU9gl',
    },
  ],
  limitations: [
    'Grey-level MNIST intensities are used directly as Bernoulli targets; the paper does the same, but this is not a normalised likelihood for continuous pixels, so the reported −ELBO is not comparable with numbers on dynamically binarised MNIST.',
    'The ELBO is a single-sample Monte Carlo estimate on every forward pass. ε is redrawn for every training batch; evaluation reuses one fixed draw, so the validation −ELBO is the same one-sample estimate at every check rather than an average over ε.',
    'No importance-weighted bound or marginal-likelihood estimate is computed.',
    'Browser training uses the 6,000-image MNIST subset; the shipped checkpoint was trained on the full 60,000-image training set.',
  ],
  guide: {
    what: 'A digit goes in and the encoder outputs not one code but a Gaussian over codes: a mean μ and a standard deviation σ for each of its two dimensions. A code is sampled from that Gaussian and decoded into a probability for every pixel. Training trades reconstruction quality against keeping each encoder Gaussian close to a standard normal, so that points drawn from N(0, I) decode to digit-like images; that is what lets the model generate new images.',
    watch:
      'This page opens with a trained checkpoint (−ELBO about 152 nats per image on the 500 validation digits); press “Start from random weights” to train from scratch. Watch −ELBO in the metrics fall from about 550 to about 165 in 500 steps; the reconstruction sharpen from a uniform grey; and the posterior σ shrink from 1 to between 0.1 and 0.15 without reaching zero (the checkpoint’s is about 0.1). Run “Latent manifold” now and then: the decoded grid turns from 100 near-identical blobs into a smooth sheet of digits, in which neighbouring cells differ far less than distant ones.',
    try: [
      {
        label: 'Strong KL weight',
        note: 'Posteriors are pulled towards the prior: after 300 steps the mean σ has grown from 0.10 to about 0.18, KL has fallen from 5.6 to 3.6 nats, and reconstructions blur slightly (pixel MSE 0.043 → 0.046). Prior samples stay digit-like.',
        patch: { beta: 4 },
        train: 300,
      },
      {
        label: 'No KL term',
        note: 'Nothing now keeps σ away from zero: it collapses to about 0.01 within 100 steps, and KL doubles to about 12 nats as the codes spread a little further from the origin. In 300 steps the reconstructions barely change (pixel MSE stays at 0.043) and samples from the prior still look like digits.',
        patch: { beta: 0 },
        train: 300,
      },
      {
        label: 'Sixteen-dimensional latent',
        note: 'Re-initialises. The bound falls lower than with two dimensions (−ELBO about 134 after 300 steps, against 152 for the checkpoint), and samples from the prior become more varied as it falls: mostly 0s and 8s after 100 steps, nine different digits after 300. The manifold view now varies one coordinate per row, for the first ten of the sixteen.',
        patch: { latent: 16 },
        train: 300,
      },
    ],
  },
  configSchema: [
    {
      key: 'hidden',
      label: 'Hidden width',
      kind: 'int',
      min: 16,
      max: 512,
      step: 16,
      affects: 'architecture',
    },
    {
      key: 'latent',
      label: 'Latent size d',
      kind: 'int',
      min: 1,
      max: 32,
      affects: 'architecture',
      help: 'With d = 2 the manifold procedure decodes a 10 × 10 grid of codes; otherwise it varies one dimension at a time.',
    },
    {
      key: 'activation',
      label: 'Hidden activation',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'relu', label: 'ReLU' },
        { value: 'tanh', label: 'tanh (paper)' },
      ],
    },
    {
      key: 'beta',
      label: 'KL weight β',
      kind: 'float',
      min: 0,
      max: 8,
      step: 0.1,
      affects: 'forward',
      help: 'β = 1 is the ELBO; β > 1 is the β-VAE objective.',
    },
  ],
  defaults: { hidden: 256, latent: 2, activation: 'relu', beta: 1 },

  init(c, rng) {
    const b = new ParamBuilder(rng);
    declareLinear(b, 'enc1', PIXELS, c.hidden);
    declareLinear(b, 'mu', c.hidden, c.latent);
    declareLinear(b, 'logvar', c.hidden, c.latent);
    declareLinear(b, 'dec1', c.latent, c.hidden);
    declareLinear(b, 'dec2', c.hidden, PIXELS);
    return b.store;
  },

  datasets: [mnistUnit<VaeConfig>()],

  forward(batch, c) {
    const target = batch.inputs.image;
    const x = input('image', target);
    const { mu, logvar } = encode(x, c);
    const { z, eps, sigma } = scope('sample', () => reparameterize(mu, logvar));
    const logits = decode(z, c);
    const recon = scope('output', () => sigmoid(logits));
    const N = target.shape[0];
    const rec = scope('recon', () => bernoulliNll(logits, target, 1 / N));
    const kl = scope('kl', () => klStandardNormal(mu, logvar));
    // Per pixel: (recon + β KL) / 784. The metrics report nats per image.
    const loss = scope('loss', () => mulScalar(add(rec, mulScalar(kl, c.beta)), 1 / PIXELS));
    const r = rec.value.data[0];
    const k = kl.value.data[0];
    return {
      outputs: { input: x, mu, logvar, sigma, eps, z, logits, reconstruction: recon },
      loss,
      metrics: { recon: r, kl: k, negElbo: r + k, pixelMse: pixelMse(recon.value, target) },
    };
  },

  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 3e-3 },
    batchSize: 64,
    steps: 500,
    validateEvery: 50,
  },

  procedures: [
    {
      id: 'prior',
      label: 'Sample from the prior',
      description: 'Draws 32 codes z ~ N(0, I) and shows the decoder mean σ(g(z)) for each.',
      run({ config, rng }) {
        const n = 32;
        const z = new Float32Array(n * config.latent);
        for (let i = 0; i < z.length; i++) z[i] = rng.normal();
        return { tensors: { samples: decodeMeans(z, n, config) } };
      },
    },
    {
      id: 'manifold',
      label: 'Latent manifold',
      description:
        'd = 2: decodes a 10 × 10 grid of codes at the Gaussian quantiles Φ⁻¹(0.05 … 0.95) in each coordinate (Kingma & Welling, fig. 4). Otherwise: each row varies one latent coordinate over the same quantiles with the others at 0.',
      run({ config, shouldStop, progress }) {
        const d = config.latent;
        const q = Array.from({ length: GRID }, (_, i) => probit(0.05 + (0.9 * i) / (GRID - 1)));
        if (d === 2) {
          const z = new Float32Array(GRID * GRID * 2);
          for (let r = 0; r < GRID; r++)
            for (let col = 0; col < GRID; col++) {
              const k = r * GRID + col;
              z[k * 2] = q[col];
              z[k * 2 + 1] = q[GRID - 1 - r];
            }
          return {
            tensors: { grid: decodeMeans(z, GRID * GRID, config) },
            text: 'Columns: z₁ from low to high. Rows: z₂ from high (top) to low.',
          };
        }
        const rows = Math.min(d, GRID);
        const z = new Float32Array(rows * GRID * d);
        for (let r = 0; r < rows; r++) {
          if (shouldStop()) break;
          for (let col = 0; col < GRID; col++) z[(r * GRID + col) * d + r] = q[col];
          progress(r / rows);
        }
        return {
          tensors: { grid: decodeMeans(z, rows * GRID, config) },
          text: `Row k varies z_k from ${q[0].toFixed(2)} to ${q[GRID - 1].toFixed(2)}; the other ${d - 1} coordinates are 0.`,
        };
      },
    },
    latentProcedure<VaeConfig>((x, c) => encode(x, c).mu, 'posterior mean μ(x)'),
  ],

  views: [
    { kind: 'image', title: 'Input x', source: 'out:input', range: [0, 1] },
    {
      kind: 'image',
      title: 'Decoder mean σ(g(z)) at the sampled z',
      source: 'out:reconstruction',
      range: [0, 1],
    },
    { kind: 'vector', title: 'Posterior mean μ', source: 'out:mu' },
    { kind: 'vector', title: 'Posterior standard deviation σ', source: 'out:sigma' },
    {
      kind: 'imageGrid',
      title: 'Samples from the prior',
      source: 'proc:prior:samples',
      range: [0, 1],
      columns: 8,
    },
    {
      kind: 'imageGrid',
      title: 'Latent manifold',
      source: 'proc:manifold:grid',
      range: [0, 1],
      columns: GRID,
    },
    latentView('Posterior means of 500 validation digits'),
  ],

  math: {
    enc: {
      tex: 'h = \\phi(W_1 x + b_1)',
      text: 'Shared hidden layer of the Gaussian encoder q_φ(z | x); φ is ReLU or tanh.',
    },
    mu: { tex: '\\mu = W_\\mu h + b_\\mu' },
    logvar: {
      tex: '\\log \\sigma^2 = W_\\sigma h + b_\\sigma',
      text: 'The log-variance is unconstrained, so σ² = exp(log σ²) is always positive.',
    },
    sample: {
      tex: 'z = \\mu + \\sigma \\odot \\varepsilon,\\quad \\varepsilon \\sim \\mathcal N(0, I)',
      text: 'The reparameterisation trick: ε is drawn from the graph’s random generator and treated as a constant, so gradients flow to μ and σ.',
    },
    dec: {
      tex: 'p_\\theta(x \\mid z) = \\textstyle\\prod_i \\mathrm{Bernoulli}\\big(x_i;\\, \\sigma(W_5\\, \\phi(W_4 z + b_4) + b_5)_i\\big)',
    },
    output: { tex: '\\hat x = \\sigma(a)', text: 'The decoder mean, shown as an image.' },
    recon: {
      tex: '-\\log p_\\theta(x\\mid z) = \\textstyle\\sum_{i=1}^{784} \\big[-x_i \\log \\hat x_i - (1 - x_i)\\log(1 - \\hat x_i)\\big]',
      text: 'One-sample estimate of the expected negative log-likelihood, averaged over the batch.',
    },
    kl: {
      tex: 'D_{\\mathrm{KL}} = \\tfrac12 \\textstyle\\sum_{j=1}^{d} \\big(\\mu_j^2 + \\sigma_j^2 - \\log \\sigma_j^2 - 1\\big)',
      text: 'Closed form for a diagonal Gaussian against the standard normal prior; no sampling needed.',
    },
    loss: {
      tex: 'L = \\tfrac{1}{784}\\big(\\text{recon} + \\beta\\, D_{\\mathrm{KL}}\\big)',
      text: 'With β = 1 the bracket is the negative evidence lower bound, an upper bound on −log p(x), in nats per image. The training loss divides it by the 784 pixels; the metrics recon, kl and negElbo are per image.',
    },
  },

  checkpoints: [
    {
      id: 'mnist',
      label: 'Trained on MNIST',
      path: 'weights/vae-mnist',
      description:
        'Default configuration (d = 2, ReLU) trained for 4 epochs on the 60,000 MNIST training images by scripts/train/vae-mnist.ts.',
    },
  ],

  customInput: unitCustomInput,
  buildCustomBatch: (t) => unitCustomBatch(t),
});

export default vae;
