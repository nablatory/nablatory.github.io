import {
  ParamBuilder,
  activations,
  defaultOptimizer,
  dense,
  scope,
  sigmoid,
  type ActivationName,
  type Var,
} from '../../engine';
import { defineModel } from '../types';
import { input } from '../util';
import {
  ACT_OPTIONS,
  decoder,
  declareDecoder,
  declareTrunk,
  latentProcedure,
  latentView,
  mnistUnit,
  pixelMse,
  reconLoss,
  trunk,
  unitCustomBatch,
  unitCustomInput,
  type Arch,
} from './common';

type AeConfig = {
  arch: Arch;
  hidden: number;
  latent: number;
  activation: 'relu' | 'tanh' | 'sigmoid';
  loss: 'bce' | 'mse';
};

function encode(x: Var, c: AeConfig): Var {
  const h = trunk(x, c.arch, activations[c.activation as ActivationName]);
  return scope('code', () => dense(h, 'code'));
}

const autoencoder = defineModel<AeConfig>({
  id: 'autoencoder',
  family: 'autoencoder',
  name: 'Autoencoder',
  summary:
    'An encoder maps a digit to a short code z and a decoder maps z back to an image; both are trained to reproduce the input.',
  signature:
    'z = f_\\phi(x),\\quad \\hat x = \\sigma\\big(g_\\theta(z)\\big),\\quad L = \\textstyle\\sum_i \\ell(x_i, \\hat x_i)',
  fidelity: 'reduced',
  scaleNote:
    'Hinton & Salakhutdinov (2006) used a 784–1000–500–250–30 encoder with a mirrored decoder, pretrained layer by layer as RBMs. Here the encoder has one hidden layer (or two strided convolutions), the decoder mirrors it, and training is end to end with Adam from random initialisation.',
  references: [
    {
      authors: 'D. E. Rumelhart, G. E. Hinton, R. J. Williams',
      title: 'Learning internal representations by error propagation',
      year: 1986,
      url: 'https://doi.org/10.7551/mitpress/5236.001.0001',
    },
    {
      authors: 'G. E. Hinton, R. R. Salakhutdinov',
      title: 'Reducing the dimensionality of data with neural networks',
      year: 2006,
      url: 'https://doi.org/10.1126/science.1127647',
    },
  ],
  limitations: [
    'No RBM pretraining; the deep autoencoder of Hinton & Salakhutdinov is not reproduced at its published depth.',
    'The binary cross-entropy option treats grey-level intensities in [0, 1] as Bernoulli targets, as is common practice; it is not a normalised likelihood for continuous pixels.',
    'Browser training uses the 6,000-image MNIST subset; the latent plot uses the first 500 test images.',
  ],
  guide: {
    what: 'A 28×28 digit goes in, a code of two numbers comes out of the encoder, and the decoder turns that code back into a 28×28 image. Nothing tells the network which digit it is looking at: the only training signal is the difference between the input and its reconstruction, so the code must hold whatever two numbers describe the image best. The latent plot shows where each of 500 test digits lands in that two-dimensional code space.',
    watch:
      'The reconstruction starts as a uniform grey and within a few dozen steps becomes a blurred average digit. By step 400 it is a blurry version of the input (pixel MSE about 0.048, against 0.064 for always drawing the average digit) and often looks like a different digit: two numbers are too few to describe an image. The loss, in nats per pixel, falls from 0.70 to about 0.20. Run “Latent space” now and then: the 500 test digits, coloured by label, spread into loose, overlapping groups. After 400 steps a digit’s five nearest neighbours in the plot share its label about 40 % of the time, against 20 % at the start.',
    try: [
      {
        label: 'One-number code',
        note: 'Re-initialises. Reconstructions blur further (pixel MSE 0.058 after 300 steps, against 0.049 with two numbers) and fall into a few digit shapes; the latent plot collapses onto a line.',
        patch: { latent: 1 },
        train: 300,
      },
      {
        label: 'Thirty-two-number code',
        note: 'Re-initialises. Reconstructions become much sharper: pixel MSE 0.022 after 300 steps, against 0.049 with two numbers. The latent plot now shows the first two principal components of the 32 numbers, which hold only about a third of their variance.',
        patch: { latent: 32 },
        train: 300,
      },
      {
        label: 'Convolutional encoder',
        note: 'Re-initialises. Two strided convolutions replace the dense hidden layer; the code size is unchanged. After 300 steps its reconstructions are slightly worse than the dense version’s (pixel MSE 0.056 against 0.049), and each step is slower.',
        patch: { arch: 'conv' },
        train: 300,
      },
    ],
  },
  configSchema: [
    {
      key: 'arch',
      label: 'Encoder and decoder',
      kind: 'select',
      affects: 'architecture',
      options: [
        { value: 'mlp', label: 'Fully connected' },
        { value: 'conv', label: 'Convolutional (stride 2)' },
      ],
    },
    {
      key: 'hidden',
      label: 'Hidden width (fully connected)',
      kind: 'int',
      min: 16,
      max: 512,
      step: 16,
      affects: 'architecture',
    },
    {
      key: 'latent',
      label: 'Code size d',
      kind: 'int',
      min: 1,
      max: 64,
      affects: 'architecture',
      help: 'With d = 2 the latent plot shows the codes themselves; otherwise their first two principal components.',
    },
    {
      key: 'activation',
      label: 'Hidden activation',
      kind: 'select',
      options: ACT_OPTIONS,
      affects: 'forward',
    },
    {
      key: 'loss',
      label: 'Reconstruction loss',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'bce', label: 'Binary cross-entropy' },
        { value: 'mse', label: 'Squared error' },
      ],
    },
  ],
  defaults: { arch: 'mlp', hidden: 256, latent: 2, activation: 'relu', loss: 'bce' },

  init(c, rng) {
    const b = new ParamBuilder(rng);
    const f = declareTrunk(b, c.arch, c.hidden);
    b.add('code.weight', [c.latent, f], { kind: 'lecunUniform', fanIn: f });
    b.add('code.bias', [c.latent], { kind: 'lecunUniform', fanIn: f });
    declareDecoder(b, c.arch, c.latent, c.hidden);
    return b.store;
  },

  datasets: [mnistUnit<AeConfig>()],

  forward(batch, c) {
    const target = batch.inputs.image;
    const x = input('image', target);
    const z = encode(x, c);
    const logits = decoder(z, c.arch, activations[c.activation as ActivationName]);
    const recon = scope('output', () => sigmoid(logits));
    const loss = scope('loss', () => reconLoss(logits, recon, target, c.loss));
    return {
      outputs: { input: x, z, logits, reconstruction: recon },
      loss,
      metrics: { pixelMse: pixelMse(recon.value, target) },
    };
  },

  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 2e-3 },
    batchSize: 64,
    steps: 400,
    validateEvery: 50,
  },

  procedures: [latentProcedure<AeConfig>(encode)],

  views: [
    { kind: 'image', title: 'Input x', source: 'out:input', range: [0, 1] },
    { kind: 'image', title: 'Reconstruction x̂', source: 'out:reconstruction', range: [0, 1] },
    { kind: 'vector', title: 'Code z', source: 'out:z' },
    latentView(),
  ],

  math: {
    enc: {
      tex: 'h = \\phi(W_1 x + b_1) \\;\\text{or}\\; h = \\phi(W \\star_{2} x + b)',
      text: 'Fully connected: one affine map of the 784 pixels and a pointwise nonlinearity. Convolutional: two 4×4 convolutions with stride 2 (28 → 14 → 7).',
    },
    code: {
      tex: 'z = W_z h + b_z \\in \\mathbb{R}^d',
      text: 'The bottleneck. No nonlinearity, so z can take any real value.',
    },
    dec: {
      tex: 'a = W_{\\text{dec}}\\, \\phi(\\cdots) + b_{\\text{dec}}',
      text: 'The decoder mirrors the encoder (transposed convolutions in the convolutional variant) and outputs one logit per pixel.',
    },
    output: { tex: '\\hat x = \\sigma(a)', text: 'Each pixel of the reconstruction is in (0, 1).' },
    loss: {
      tex: 'L = \\tfrac{1}{784N}\\textstyle\\sum_{n,i} \\big[-x_i \\log \\hat x_i - (1-x_i)\\log(1-\\hat x_i)\\big] \\;\\text{or}\\; \\tfrac{1}{784N}\\sum_n \\lVert x - \\hat x \\rVert^2',
      text: 'Averaged over pixels and examples (nats per pixel for the cross-entropy). The cross-entropy is computed from the logits in a numerically stable form.',
    },
  },

  customInput: unitCustomInput,
  buildCustomBatch: (t) => unitCustomBatch(t),
});

export default autoencoder;
