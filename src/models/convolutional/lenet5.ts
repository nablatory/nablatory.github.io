import {
  Tensor,
  ParamBuilder,
  Rng,
  activations,
  add,
  mul,
  mulScalar,
  avgPool2d,
  maxPool2d,
  conv2d,
  flatten,
  softmax,
  crossEntropy,
  sum,
  square,
  sub,
  reshape,
  concat,
  p,
  dense,
  conv,
  scope,
  defaultOptimizer,
  type Var,
  type ActivationName,
} from '../../engine';
import { convOutSize } from '../../engine/ops/conv';
import { defineModel, type Batch, type ForwardResult } from '../types';
import { DIGITS, MNIST_PREPROCESSING, encodeDigit, mnistDataset } from '../data/mnist';
import { accuracy, constant, input } from '../util';

type LeNetConfig = {
  activation: 'scaledTanh' | 'tanh' | 'relu' | 'sigmoid';
  pooling: 'trainable' | 'avg' | 'max';
  c3: 'table' | 'full';
  output: 'linear' | 'rbf';
  c1Padding: number;
  c1Stride: number;
};

/**
 * C3 connection table (LeCun et al., 1998, Table I): which of the six S2 maps
 * feed each of the sixteen C3 maps. 60 connections → 1,516 parameters.
 */
export const C3_TABLE: number[][] = [
  [0, 1, 2],
  [1, 2, 3],
  [2, 3, 4],
  [3, 4, 5],
  [4, 5, 0],
  [5, 0, 1],
  [0, 1, 2, 3],
  [1, 2, 3, 4],
  [2, 3, 4, 5],
  [3, 4, 5, 0],
  [4, 5, 0, 1],
  [5, 0, 1, 2],
  [0, 1, 3, 4],
  [1, 2, 4, 5],
  [0, 2, 3, 5],
  [0, 1, 2, 3, 4, 5],
];

/** Spatial sizes through the network; null when a layer would be empty. */
export function lenetGeometry(
  c: LeNetConfig,
): { c1: number; s2: number; c3: number; s4: number; c5: number } | null {
  const c1 = convOutSize(32, 5, c.c1Stride, c.c1Padding);
  const s2 = Math.floor(c1 / 2);
  const c3 = s2 - 4;
  const s4 = Math.floor(c3 / 2);
  const c5 = s4 - 4;
  if (c1 < 2 || s2 < 5 || c3 < 2 || s4 < 5 || c5 < 1) return null;
  return { c1, s2, c3, s4, c5 };
}

function uniformInit(shape: number[], bound: number, rng: Rng): Tensor {
  const t = Tensor.zeros(shape);
  for (let i = 0; i < t.size; i++) t.data[i] = rng.range(-bound, bound);
  return t;
}

function subsample(name: string, x: Var, c: LeNetConfig, act: (v: Var) => Var): Var {
  if (c.pooling === 'max') return maxPool2d(x, 2);
  if (c.pooling === 'avg') return avgPool2d(x, 2);
  // Trainable subsampling: f(w_c · Σ_{2×2} x + b_c)
  const C = x.shape[1];
  const pooled = mulScalar(avgPool2d(x, 2), 4);
  const scaled = mul(pooled, reshape(p(`${name}.coef`), [1, C, 1, 1]));
  return act(add(scaled, reshape(p(`${name}.bias`), [1, C, 1, 1])));
}

const lenet5 = defineModel<LeNetConfig>({
  id: 'lenet-5',
  family: 'convolutional',
  name: 'LeNet-5',
  summary:
    'Seven-layer convolutional network for 32×32 digit images: alternating convolution and subsampling, then fully connected layers.',
  signature:
    'x \\xrightarrow{\\text{C1}} 6@28^2 \\xrightarrow{\\text{S2}} 6@14^2 \\xrightarrow{\\text{C3}} 16@10^2 \\xrightarrow{\\text{S4}} 16@5^2 \\to 120 \\to 84 \\to 10',
  fidelity: 'exact',
  scaleNote:
    'Layer sizes, C3 connection table, trainable subsampling and the scaled tanh are as published. The default output layer is a linear softmax classifier; the published Euclidean RBF output is selectable (its fixed codes are random ±1 vectors rather than the paper’s 7×12 character bitmaps).',
  references: [
    {
      authors: 'Y. LeCun, L. Bottou, Y. Bengio, P. Haffner',
      title: 'Gradient-based learning applied to document recognition',
      year: 1998,
      url: 'https://doi.org/10.1109/5.726791',
    },
  ],
  limitations: [
    'Trained here with Adam and cross-entropy, not the stochastic diagonal Levenberg–Marquardt method used in the paper.',
    'The shipped checkpoint was trained on the full MNIST training set by scripts/train/lenet5.ts; browser training uses the 6,000-image subset.',
  ],
  guide: {
    what: 'LeNet-5 reads a 32×32 image of a handwritten digit and outputs ten class probabilities. Its first layers are convolutions, small 5×5 filters slid across the image, each producing a feature map that responds where its pattern appears; subsampling layers then halve the maps, and fully connected layers at the end turn the last maps into the ten scores. Training learns the filters, the subsampling coefficients and the final weights.',
    watch:
      'This page opens with trained weights, which get 98 % of the test digits right. Press “Start from random weights” to train from scratch: the loss starts at ln 10 ≈ 2.30 (guessing among ten) and in 200 steps reaches about 0.5 with 85 % accuracy; watch the C1 feature maps change as the filters are learned.',
    try: [
      {
        label: 'Full C3 connections',
        note: 'Every C3 map reads all six S2 maps: 900 more parameters and about the same accuracy after 200 steps.',
        patch: { c3: 'full' },
        train: 200,
      },
      {
        label: 'Paper’s RBF output',
        note: 'Scores become negative squared distances to fixed codes; the loss is now tens rather than units, but accuracy climbs the same way.',
        patch: { output: 'rbf' },
        train: 200,
      },
    ],
    views: ['C1 feature maps', 'Class probabilities'],
  },
  configSchema: [
    {
      key: 'activation',
      label: 'Activation',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'scaledTanh', label: '1.7159 tanh(2a/3)' },
        { value: 'tanh', label: 'tanh' },
        { value: 'relu', label: 'ReLU' },
        { value: 'sigmoid', label: 'sigmoid' },
      ],
    },
    {
      key: 'pooling',
      label: 'Subsampling',
      kind: 'select',
      affects: 'architecture',
      options: [
        { value: 'trainable', label: 'Trainable (paper)' },
        { value: 'avg', label: 'Average pool' },
        { value: 'max', label: 'Max pool' },
      ],
    },
    {
      key: 'c3',
      label: 'C3 connections',
      kind: 'select',
      affects: 'architecture',
      options: [
        { value: 'table', label: 'Partial table (paper)' },
        { value: 'full', label: 'Full' },
      ],
    },
    {
      key: 'output',
      label: 'Output layer',
      kind: 'select',
      affects: 'architecture',
      options: [
        { value: 'linear', label: 'Linear + softmax' },
        { value: 'rbf', label: 'Euclidean RBF (paper)' },
      ],
    },
    { key: 'c1Padding', label: 'C1 padding', kind: 'int', min: 0, max: 4, affects: 'architecture' },
    { key: 'c1Stride', label: 'C1 stride', kind: 'int', min: 1, max: 2, affects: 'architecture' },
  ],
  defaults: {
    activation: 'scaledTanh',
    pooling: 'trainable',
    c3: 'table',
    output: 'linear',
    c1Padding: 0,
    c1Stride: 1,
  },

  validate(c) {
    const g = lenetGeometry(c);
    if (!g) {
      const c1 = convOutSize(32, 5, c.c1Stride, c.c1Padding);
      return `With stride ${c.c1Stride} and padding ${c.c1Padding}, C1 is ${c1}×${c1}; after two 5×5 convolutions and two 2×2 subsamplings no spatial extent remains for C5 (needs ≥ 5×5 at S4).`;
    }
    return null;
  },

  init(c, rng) {
    const g = lenetGeometry(c)!;
    const b = new ParamBuilder(rng);
    const s = b.store;
    // Paper initialisation: U(−2.4/F, 2.4/F), F = fan-in.
    const lecun98 = (name: string, shape: number[], fanIn: number) =>
      s.add(name, uniformInit(shape, 2.4 / fanIn, rng.fork(name)), `U(±2.4/F), F = ${fanIn}`);
    lecun98('c1.weight', [6, 1, 5, 5], 25);
    lecun98('c1.bias', [6], 25);
    if (c.pooling === 'trainable') {
      lecun98('s2.coef', [6], 4);
      lecun98('s2.bias', [6], 4);
    }
    const w3 = uniformInit([16, 6, 5, 5], 1, rng.fork('c3.weight'));
    const mask = Tensor.zeros([16, 6, 1, 1]);
    for (let o = 0; o < 16; o++) {
      const ins = c.c3 === 'table' ? C3_TABLE[o] : [0, 1, 2, 3, 4, 5];
      for (const i of ins) mask.data[o * 6 + i] = 1;
      const bound = 2.4 / (ins.length * 25);
      for (let i = 0; i < 6 * 25; i++) w3.data[o * 150 + i] *= bound;
    }
    s.add('c3.weight', w3, 'U(±2.4/F), F = 25 × connected maps');
    const b3 = uniformInit([16], 1, rng.fork('c3.bias'));
    for (let o = 0; o < 16; o++) {
      const F = (c.c3 === 'table' ? C3_TABLE[o].length : 6) * 25;
      b3.data[o] *= 2.4 / F;
    }
    s.add('c3.bias', b3, 'U(±2.4/F), F = 25 × connected maps');
    if (c.c3 === 'table') b.fixed('c3.mask', mask, 'Connection table (LeCun et al. 1998, Table I)');
    if (c.pooling === 'trainable') {
      lecun98('s4.coef', [16], 4);
      lecun98('s4.bias', [16], 4);
    }
    lecun98('c5.weight', [120, 16, 5, 5], 400);
    lecun98('c5.bias', [120], 400);
    const f6In = 120 * g.c5 * g.c5;
    lecun98('f6.weight', [84, f6In], f6In);
    lecun98('f6.bias', [84], f6In);
    if (c.output === 'linear') {
      lecun98('output.weight', [10, 84], 84);
      lecun98('output.bias', [10], 84);
    } else {
      const codes = Tensor.zeros([10, 84]);
      const r = rng.fork('rbf');
      for (let i = 0; i < codes.size; i++) codes.data[i] = r.uniform() < 0.5 ? -1 : 1;
      b.fixed('output.centers', codes, 'Fixed ±1 codes (random; the paper used 7×12 bitmaps)');
    }
    return s;
  },

  datasets: [mnistDataset<LeNetConfig>(() => 'lecun')],

  forward(batch, c, _ctx): ForwardResult {
    const act = activations[c.activation as ActivationName];
    const x = input('image', batch.inputs.image);
    const c1 = scope('c1', () => act(conv(x, 'c1', { stride: c.c1Stride, padding: c.c1Padding })));
    const s2 = scope('s2', () => subsample('s2', c1, c, act));
    const c3 = scope('c3', () => {
      const w = c.c3 === 'table' ? mul(p('c3.weight'), p('c3.mask')) : p('c3.weight');
      return act(conv2d(s2, w, p('c3.bias')));
    });
    const s4 = scope('s4', () => subsample('s4', c3, c, act));
    const c5 = scope('c5', () => act(conv(s4, 'c5')));
    const f6 = scope('f6', () => act(dense(flatten(c5), 'f6')));
    let logits: Var;
    let loss: Var | undefined;
    if (c.output === 'linear') {
      logits = scope('output', () => dense(f6, 'output'));
      if (batch.labels) loss = scope('loss', () => crossEntropy(logits, batch.labels!));
    } else {
      // y_i = Σ_j (x_j − w_ij)²; class scores are −y_i.
      const dist = scope('output', () => {
        const N = f6.shape[0];
        const d = sub(reshape(f6, [N, 1, 84]), reshape(p('output.centers'), [1, 10, 84]));
        return sum(square(d), -1);
      });
      logits = mulScalar(dist, -1);
      if (batch.labels) {
        // MAP criterion: E = y_D + log(e^{−j} + Σ_i e^{−y_i}), j = 1.
        loss = scope('loss', () => {
          const N = logits.shape[0];
          const garbage = Tensor.full([N, 1], -1);
          return crossEntropy(concat([logits, constant(garbage)], 1), batch.labels!);
        });
      }
    }
    const probs = softmax(logits, -1);
    const metrics = batch.labels ? { accuracy: accuracy(probs.value, batch.labels) } : undefined;
    return { outputs: { input: x, c1, s2, c3, s4, c5, f6, logits, probs }, loss, metrics };
  },

  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 1e-3 },
    batchSize: 32,
    steps: 200,
    validateEvery: 50,
  },

  views: [
    { kind: 'image', title: 'Input (32×32)', source: 'out:input', range: [-0.1, 1.175] },
    { kind: 'featureMaps', title: 'C1 feature maps', source: 'out:c1' },
    { kind: 'featureMaps', title: 'S2 subsampling', source: 'out:s2' },
    { kind: 'featureMaps', title: 'C3 feature maps', source: 'out:c3' },
    { kind: 'featureMaps', title: 'S4 subsampling', source: 'out:s4' },
    { kind: 'probs', title: 'Class probabilities', source: 'out:probs', classNames: DIGITS },
  ],

  math: {
    c1: {
      tex: 'C1_{o}(y,x) = f\\Big(b_o + \\sum_{i,j} w_{o,i,j}\\, X(s y - p + i,\\, s x - p + j)\\Big)',
      text: 'Six 5×5 kernels slide over the input. Each output pixel is a dot product of a kernel with one 5×5 patch, plus a bias, passed through f.',
    },
    s2: {
      tex: 'S2_{c}(y,x) = f\\Big(\\alpha_c \\sum_{i,j\\in\\{0,1\\}} C1_c(2y+i,\\,2x+j) + \\beta_c\\Big)',
      text: 'Trainable subsampling: each map sums non-overlapping 2×2 blocks, multiplies by one learned coefficient and adds one bias. With the average or max option the block is pooled directly, without coefficient, bias or f.',
    },
    c3: {
      tex: 'C3_{o} = f\\Big(b_o + \\sum_{c \\in T(o)} w_{o,c} \\star S2_c\\Big)',
      text: 'Each C3 map reads only the S2 maps listed in the connection table T(o). Implemented as a full convolution whose weights are multiplied by a fixed 0/1 mask, so masked weights receive zero gradient.',
    },
    s4: {
      tex: 'S4_{c}(y,x) = f\\Big(\\alpha_c \\sum_{i,j} C3_c(2y+i,2x+j) + \\beta_c\\Big)',
      text: 'Same trainable subsampling as S2, with its own coefficient and bias per map. With the average or max option the block is pooled directly, without coefficient, bias or f.',
    },
    c5: {
      tex: 'C5_o = f\\Big(b_o + \\sum_{c=1}^{16} w_{o,c} \\star S4_c\\Big)',
      text: '120 maps, each 5×5 kernel covers the whole 5×5 S4 map, so the output is 1×1.',
    },
    f6: { tex: 'F6 = f(W\\, C5 + b),\\quad W \\in \\mathbb{R}^{84\\times 120}' },
    output: {
      tex: 'z = W F6 + b,\\qquad p = \\operatorname{softmax}(z)',
      text: 'With the RBF option: y_i = Σ_j (F6_j − w_ij)², scores −y_i.',
    },
    loss: {
      tex: 'L = -\\frac{1}{N}\\sum_n \\log p_{n, y_n}',
      text: 'With the RBF output the loss is the paper’s MAP criterion E = y_D + log(e^{−j} + Σ_i e^{−y_i}) with j = 1: the cross-entropy of the scores −y_i against one extra constant logit −j standing for a rubbish class.',
    },
  },

  checkpoints: [
    {
      id: 'mnist',
      label: 'Trained on MNIST',
      path: 'weights/lenet-5-mnist',
      description: 'Default configuration trained on the 60,000 MNIST training images.',
    },
  ],

  customInput: {
    kind: 'image',
    channels: 1,
    height: 28,
    width: 28,
    invertUploads: true,
    preprocessing: MNIST_PREPROCESSING.lecun,
  },

  buildCustomBatch(input) {
    const u8 = new Float32Array(784);
    for (let i = 0; i < 784; i++) u8[i] = Math.max(0, Math.min(1, input.data[i])) * 255;
    const x = new Float32Array(32 * 32);
    encodeDigit(u8, 0, 'lecun', x, 0);
    return {
      inputs: { image: new Tensor([1, 1, 32, 32], x) },
      meta: { classNames: DIGITS, caption: 'Your drawing' },
    } satisfies Batch;
  },
});

export default lenet5;
