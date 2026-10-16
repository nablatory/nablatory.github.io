import {
  ParamBuilder,
  activations,
  crossEntropy,
  declareLinear,
  defaultOptimizer,
  dense,
  scope,
  softmax,
  type ActivationName,
  type Var,
} from '../../engine';
import { defineModel } from '../types';
import { PLANAR_OPTIONS, planarDataset, type PlanarKind } from '../data/planar';
import { accuracy, input } from '../util';
import { fieldProcedure, fieldViews, logitMargin } from './planar-common';

type MlpConfig = {
  dataset: PlanarKind;
  noise: number;
  layers: number;
  width: number;
  activation: ActivationName;
  init: 'torch' | 'he' | 'xavier';
};

const ACTS: { value: ActivationName; label: string }[] = [
  { value: 'tanh', label: 'tanh' },
  { value: 'relu', label: 'ReLU' },
  { value: 'gelu', label: 'GELU' },
  { value: 'silu', label: 'SiLU' },
  { value: 'sigmoid', label: 'sigmoid' },
  { value: 'leakyRelu', label: 'leaky ReLU' },
  { value: 'identity', label: 'none (linear)' },
];

function forward(x: Var, c: MlpConfig) {
  const act = activations[c.activation];
  let h = x;
  for (let l = 0; l < c.layers; l++)
    h = scope(`layer${l + 1}`, () => act(dense(h, `layer${l + 1}`)));
  return scope('output', () => dense(h, 'output'));
}

const mlp = defineModel<MlpConfig>({
  id: 'mlp',
  family: 'feedforward',
  name: 'Multilayer perceptron',
  summary:
    'Affine maps alternating with a pointwise nonlinearity, trained by backpropagation to separate two classes in the plane.',
  signature:
    'h_{\\ell} = \\sigma(W_{\\ell} h_{\\ell-1} + b_{\\ell}),\\quad p = \\operatorname{softmax}(W_{L} h_{L-1} + b_L)',
  fidelity: 'equations',
  scaleNote:
    'Rumelhart, Hinton and Williams (1986) define the architecture and the learning rule but no canonical size; depth and width here are chosen for two-dimensional inputs.',
  references: [
    {
      authors: 'D. E. Rumelhart, G. E. Hinton, R. J. Williams',
      title: 'Learning representations by back-propagating errors',
      year: 1986,
      url: 'https://doi.org/10.1038/323533a0',
    },
  ],
  limitations: [
    'Two-dimensional synthetic data only, so that the full decision function can be drawn.',
    'Softmax output with cross-entropy, trained with Adam; the 1986 paper used logistic output units with a squared-error criterion and gradient descent with momentum.',
  ],
  guide: {
    what: 'A multilayer perceptron stacks several layers: each multiplies its input by a weight matrix, adds a bias and applies a nonlinearity (here tanh) to every number. It takes a point in the plane and outputs two class probabilities. Training by backpropagation, which works out how much each weight contributed to the error, adjusts all the weights so the probabilities match the labels; the hidden layers let the boundary between the classes bend, which a single perceptron cannot do.',
    watch:
      'The decision field starts as a blur and the two spirals are half wrong (loss about 0.69, which is guessing). Over 300 steps the field bends around the arms; accuracy often sits near 75 % for the first 150 steps and then jumps, ending above 95 % with a loss below 0.15 in a typical run.',
    try: [
      {
        label: 'No hidden layers',
        note: 'With no hidden layer the boundary is one straight line: on the spirals accuracy tops out near 75 %.',
        patch: { layers: 0 },
        train: 300,
      },
      {
        label: 'ReLU instead of tanh',
        note: 'Piecewise-linear units make the field polygonal; this run reached 99 % within 150 steps.',
        patch: { activation: 'relu' },
        train: 300,
      },
      {
        label: 'No nonlinearity',
        note: 'Three linear layers compose to one line: the same 75 % as no hidden layers.',
        patch: { activation: 'identity' },
        train: 300,
      },
    ],
  },
  configSchema: [
    { key: 'dataset', label: 'Dataset', kind: 'select', options: PLANAR_OPTIONS, affects: 'data' },
    {
      key: 'noise',
      label: 'Point noise σ',
      kind: 'float',
      min: 0,
      max: 0.4,
      step: 0.02,
      affects: 'data',
    },
    { key: 'layers', label: 'Hidden layers', kind: 'int', min: 0, max: 5, affects: 'architecture' },
    { key: 'width', label: 'Width', kind: 'int', min: 1, max: 64, affects: 'architecture' },
    {
      key: 'init',
      label: 'Initialisation',
      kind: 'select',
      affects: 'architecture',
      help: 'U(±1/√fan_in) is PyTorch’s default for weights and biases. He normal and Glorot uniform start the biases at zero; with He normal the output layer uses Glorot uniform.',
      options: [
        { value: 'torch', label: 'U(±1/√fan_in)' },
        { value: 'he', label: 'He normal' },
        { value: 'xavier', label: 'Glorot uniform' },
      ],
    },
    { key: 'activation', label: 'Activation σ', kind: 'select', options: ACTS, affects: 'forward' },
  ],
  defaults: {
    dataset: 'spirals',
    noise: 0.08,
    layers: 3,
    width: 16,
    activation: 'tanh',
    init: 'torch',
  },
  init(c, rng) {
    const b = new ParamBuilder(rng);
    let nIn = 2;
    for (let l = 0; l < c.layers; l++) {
      declareLinear(b, `layer${l + 1}`, nIn, c.width, { init: c.init });
      nIn = c.width;
    }
    declareLinear(b, 'output', nIn, 2, { init: c.init === 'he' ? 'xavier' : c.init });
    return b.store;
  },
  datasets: [planarDataset<MlpConfig>((c) => c.dataset, { noise: (c) => c.noise })],
  forward(batch, c) {
    const x = input('x', batch.inputs.x);
    const logits = forward(x, c);
    const probs = softmax(logits, -1);
    const loss = batch.labels
      ? scope('loss', () => crossEntropy(logits, batch.labels!))
      : undefined;
    return {
      outputs: { logits, probs },
      loss,
      metrics: batch.labels ? { accuracy: accuracy(probs.value, batch.labels) } : undefined,
    };
  },
  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 0.01 },
    batchSize: 32,
    steps: 300,
    validateEvery: 25,
  },
  procedures: [
    fieldProcedure<MlpConfig>(
      (b, c) => ({ outputs: { logits: forward(input('x', b.inputs.x, false), c) } }),
      (c) => c.dataset,
      (r) => logitMargin(r.outputs.logits.value),
      (c) => c.noise,
    ),
  ],
  views: [
    ...fieldViews,
    {
      kind: 'probs',
      title: 'Class probabilities (first example)',
      source: 'out:probs',
      classNames: ['class 0', 'class 1'],
    },
  ],
  math: {
    layer1: {
      tex: 'h_1 = \\sigma(W_1 x + b_1)',
      text: 'Each hidden unit computes a weighted sum of its inputs plus a bias, then applies σ elementwise.',
    },
    layer2: { tex: 'h_2 = \\sigma(W_2 h_1 + b_2)' },
    layer3: { tex: 'h_3 = \\sigma(W_3 h_2 + b_3)' },
    layer4: { tex: 'h_4 = \\sigma(W_4 h_3 + b_4)' },
    layer5: { tex: 'h_5 = \\sigma(W_5 h_4 + b_5)' },
    output: {
      tex: 'z = W_{\\text{out}} h + b_{\\text{out}},\\quad p = \\operatorname{softmax}(z)',
      text: 'The decision field shows z₁ − z₀: positive (red) where class 1 is predicted.',
    },
    loss: { tex: 'L = -\\tfrac{1}{N}\\textstyle\\sum_n \\log p_{n,y_n}' },
  },
});

export default mlp;
