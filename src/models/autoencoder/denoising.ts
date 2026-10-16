import {
  ParamBuilder,
  Tensor,
  T,
  activations,
  add,
  declareLinear,
  defaultOptimizer,
  flatten,
  graphRng,
  linear,
  mul,
  p,
  reshape,
  scope,
  sigmoid,
  type ActivationName,
  type Var,
} from '../../engine';
import { defineModel } from '../types';
import { constant, input } from '../util';
import {
  ACT_OPTIONS,
  PIXELS,
  SIDE,
  latentProcedure,
  latentView,
  mnistUnit,
  pixelMse,
  reconLoss,
  unitCustomBatch,
  unitCustomInput,
} from './common';

type DaeConfig = {
  hidden: number;
  tied: boolean;
  activation: 'sigmoid' | 'relu' | 'tanh';
  noise: 'masking' | 'gaussian' | 'saltpepper';
  fraction: number;
  sigma: number;
  loss: 'bce' | 'mse';
};

/**
 * Corruption q(x̃ | x), sampled from the graph's generator:
 *   masking     – exactly ⌊ν·784⌋ pixels per image, chosen uniformly, are set to 0
 *   saltpepper  – the same number of pixels are set to 0 or 1 with equal probability
 *   gaussian    – x̃ = x + σ ε, ε ~ N(0, I)
 */
export function corrupt(x: Var, c: DaeConfig): Var {
  const rng = graphRng();
  const N = x.shape[0];
  const D = x.value.size / N;
  if (c.noise === 'gaussian') {
    const e = Tensor.zeros(x.shape);
    for (let i = 0; i < e.size; i++) e.data[i] = c.sigma * rng.normal();
    return add(x, constant(e, 'noise'));
  }
  const k = Math.floor(c.fraction * D);
  const keep = Tensor.full(x.shape, 1);
  const salt = Tensor.zeros(x.shape);
  const idx = Array.from({ length: D }, (_, i) => i);
  for (let n = 0; n < N; n++) {
    // Partial Fisher–Yates: the first k entries of idx are a uniform k-subset.
    for (let j = 0; j < k; j++) {
      const r = j + rng.int(D - j);
      [idx[j], idx[r]] = [idx[r], idx[j]];
      keep.data[n * D + idx[j]] = 0;
      if (c.noise === 'saltpepper' && rng.uniform() < 0.5) salt.data[n * D + idx[j]] = 1;
    }
  }
  const masked = mul(x, constant(keep, 'mask'));
  return c.noise === 'saltpepper' ? add(masked, constant(salt, 'salt')) : masked;
}

function encode(x: Var, c: DaeConfig): Var {
  const act = activations[c.activation as ActivationName];
  return scope('enc', () => act(linear(flatten(x), p('enc.weight'), p('enc.bias'))));
}

function decode(h: Var, c: DaeConfig): Var {
  return scope('dec', () => {
    const w = c.tied ? T(p('enc.weight')) : p('dec.weight');
    return reshape(linear(h, w, p('dec.bias')), [h.shape[0], 1, SIDE, SIDE]);
  });
}

const denoising = defineModel<DaeConfig>({
  id: 'denoising-autoencoder',
  family: 'autoencoder',
  name: 'Denoising autoencoder',
  summary:
    'The input is corrupted at random and the network is trained to reconstruct the clean digit, so it must learn structure rather than copy pixels.',
  signature:
    "\\tilde x \\sim q(\\tilde x \\mid x),\\quad h = s(W\\tilde x + b),\\quad L = \\ell\\big(x,\\, s(W^{\\top} h + b')\\big)",
  fidelity: 'reduced',
  scaleNote:
    'One hidden layer with sigmoid units, tied weights, masking noise and cross-entropy as in Vincent et al. (2008). The paper used 1,000–2,000 hidden units, plain SGD, and stacked the layers before supervised fine-tuning; here the default is 128 units, Adam, and a single layer with no stacking.',
  references: [
    {
      authors: 'P. Vincent, H. Larochelle, Y. Bengio, P.-A. Manzagol',
      title: 'Extracting and composing robust features with denoising autoencoders',
      year: 2008,
      url: 'https://doi.org/10.1145/1390156.1390294',
    },
    {
      authors: 'P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, P.-A. Manzagol',
      title:
        'Stacked denoising autoencoders: learning useful representations in a deep network with a local denoising criterion',
      year: 2010,
      url: 'https://jmlr.org/papers/v11/vincent10a.html',
    },
  ],
  limitations: [
    'Single layer only; stacking and the supervised fine-tuning stage of the paper are not included.',
    'Grey-level intensities in [0, 1] are used as cross-entropy targets; the paper also used the [0, 1] scaled pixels.',
    'Salt-and-pepper noise is from the 2010 journal version; Gaussian noise is an option the 2008 paper discusses but did not use on MNIST.',
    'A fresh corruption is drawn for every training batch. Evaluation reuses one fixed draw (the same seed every time), so the validation loss measures the same corrupted digits at every check; it is a single sample of the expected loss, not its average over corruptions.',
  ],
  guide: {
    what: 'A digit is damaged before the network sees it: by default a random quarter of its pixels are set to zero. A single hidden layer of 128 sigmoid units encodes the damaged image, and the decoder, which reuses the same weights transposed, must reproduce the clean original. Copying the damaged input no longer gives a perfect score, so the network has to fill the gaps from what it has learned about digits. The filters view shows the input pattern each hidden unit responds to.',
    watch:
      'Compare the three images: corrupted input, reconstruction and clean target. At first the reconstruction is grey noise, then a blur; as the loss falls it fills in part of the erased strokes, and by step 300 its pixel MSE (0.025) is below the MSE of the corrupted input (0.026). The validation loss uses the same corruption of the same 500 digits every time, so it falls smoothly, from 0.22 at step 50 to 0.13 at step 400. Run “Encoder filters” to see the filters turn from random speckle into smooth patterns.',
    try: [
      {
        label: 'Corrupt most pixels',
        note: 'Three quarters of the pixels are erased, so most of each stroke is gone, yet after 300 steps most reconstructions are still recognisable digits, blurrier than at ν = 0.25 (pixel MSE 0.036 against 0.025).',
        patch: { fraction: 0.75 },
        train: 300,
      },
      {
        label: 'No corruption',
        note: 'With ν = 0 this is a plain autoencoder. At this size and in 300 steps its filters look much like the denoising ones, only slightly less smooth, and its reconstructions of clean digits are a little closer (pixel MSE 0.022 against 0.025).',
        patch: { fraction: 0 },
        train: 300,
      },
      {
        label: 'Gaussian noise',
        note: 'Every pixel is perturbed instead of some being erased (MSE of the corrupted input about 0.25); the reconstruction has to remove the grain rather than fill gaps, and after 300 steps its pixel MSE is about 0.03.',
        patch: { noise: 'gaussian', sigma: 0.5 },
        train: 300,
      },
    ],
    views: ['Corrupted input x̃', 'Reconstruction'],
  },
  configSchema: [
    {
      key: 'hidden',
      label: 'Hidden units d′',
      kind: 'int',
      min: 16,
      max: 512,
      step: 16,
      affects: 'architecture',
    },
    { key: 'tied', label: 'Tied weights W′ = Wᵀ', kind: 'bool', affects: 'architecture' },
    {
      key: 'activation',
      label: 'Hidden activation s',
      kind: 'select',
      options: [ACT_OPTIONS[2], ACT_OPTIONS[0], ACT_OPTIONS[1]],
      affects: 'architecture',
      help: 'Re-initialises. Sigmoid units start from U(±4√(6/(784 + d′))), the Glorot–Bengio bound scaled by 4 for sigmoids as in the reference Theano implementation; the other activations use the PyTorch default.',
    },
    {
      key: 'noise',
      label: 'Corruption',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'masking', label: 'Masking (set to 0)' },
        { value: 'saltpepper', label: 'Salt and pepper' },
        { value: 'gaussian', label: 'Additive Gaussian' },
      ],
    },
    {
      key: 'fraction',
      label: 'Corrupted fraction ν',
      kind: 'float',
      min: 0,
      max: 0.9,
      step: 0.05,
      affects: 'forward',
      help: 'Masking and salt-and-pepper noise.',
    },
    {
      key: 'sigma',
      label: 'Noise σ',
      kind: 'float',
      min: 0,
      max: 1,
      step: 0.05,
      affects: 'forward',
      help: 'Additive Gaussian noise.',
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
  defaults: {
    hidden: 128,
    tied: true,
    activation: 'sigmoid',
    noise: 'masking',
    fraction: 0.25,
    sigma: 0.3,
    loss: 'bce',
  },

  init(c, rng) {
    const b = new ParamBuilder(rng);
    // Glorot & Bengio (2010) bound ×4 for sigmoid units, as in the Theano reference implementation; PyTorch default otherwise.
    if (c.activation === 'sigmoid') {
      b.add('enc.weight', [c.hidden, PIXELS], {
        kind: 'uniform',
        bound: 4 * Math.sqrt(6 / (PIXELS + c.hidden)),
      });
      b.add('enc.bias', [c.hidden], { kind: 'zeros' });
    } else {
      declareLinear(b, 'enc', PIXELS, c.hidden);
    }
    if (c.tied) b.add('dec.bias', [PIXELS], { kind: 'zeros' });
    else declareLinear(b, 'dec', c.hidden, PIXELS);
    return b.store;
  },

  datasets: [mnistUnit<DaeConfig>()],

  forward(batch, c) {
    const clean = batch.inputs.image;
    const x = input('image', clean);
    const xt = scope('corrupt', () => corrupt(x, c));
    const h = encode(xt, c);
    const logits = decode(h, c);
    const recon = scope('output', () => sigmoid(logits));
    const loss = scope('loss', () => reconLoss(logits, recon, clean, c.loss));
    return {
      outputs: { clean: x, corrupted: xt, code: h, logits, reconstruction: recon },
      loss,
      metrics: { pixelMse: pixelMse(recon.value, clean), corruptedMse: pixelMse(xt.value, clean) },
    };
  },

  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 3e-3 },
    batchSize: 64,
    steps: 400,
    validateEvery: 50,
  },

  procedures: [
    {
      id: 'filters',
      label: 'Encoder filters',
      description:
        'Rows of W reshaped to 28 × 28: the input pattern each hidden unit responds to most strongly (first 64 units).',
      run({ params }) {
        const w = params.get('enc.weight');
        const k = Math.min(64, w.shape[0]);
        return {
          tensors: {
            filters: new Tensor(
              [k, 1, SIDE, SIDE],
              Float32Array.from(w.data.subarray(0, k * PIXELS)),
            ),
          },
        };
      },
    },
    latentProcedure<DaeConfig>(encode, 'hidden code h of the clean digit'),
  ],

  views: [
    { kind: 'image', title: 'Clean input x', source: 'out:clean', range: [0, 1] },
    { kind: 'image', title: 'Corrupted input x̃', source: 'out:corrupted', range: [0, 1] },
    { kind: 'image', title: 'Reconstruction', source: 'out:reconstruction', range: [0, 1] },
    {
      kind: 'imageGrid',
      title: 'Encoder filters (rows of W)',
      source: 'proc:filters:filters',
      columns: 8,
    },
    latentView('Hidden codes of 500 clean validation digits'),
  ],

  math: {
    corrupt: {
      tex: '\\tilde x = x \\odot m \\;(\\text{masking}),\\quad \\tilde x = x + \\sigma\\varepsilon \\;(\\text{Gaussian})',
      text: 'Masking sets a random ⌊ν·784⌋ of the pixels of each image to 0. The corruption is drawn from the graph’s random generator: a new draw for every training batch, one fixed draw for evaluation.',
    },
    enc: {
      tex: 'h = s(W \\tilde x + b)',
      text: 'The hidden code is computed from the corrupted input only.',
    },
    dec: {
      tex: "a = W' h + b',\\quad W' = W^{\\top} \\;(\\text{tied})",
      text: 'With tied weights the decoder reuses the encoder matrix, transposed; its gradient is the sum of both uses.',
    },
    output: { tex: 'z = \\sigma(a)' },
    loss: {
      tex: 'L_H(x, z) = -\\textstyle\\sum_{i} \\big[x_i \\log z_i + (1-x_i)\\log(1-z_i)\\big]',
      text: 'The target is the clean x, not the corrupted x̃. The training loss is this sum divided by 784 and averaged over the batch.',
    },
  },

  customInput: unitCustomInput,
  buildCustomBatch: (t) => unitCustomBatch(t),
});

export default denoising;
