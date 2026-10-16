import {
  ParamBuilder,
  activations,
  add,
  crossEntropy,
  declareLinear,
  declareNorm,
  defaultOptimizer,
  dense,
  layerNorm,
  p,
  scope,
  softmax,
  type ActivationName,
  type Var,
} from '../../engine';
import { defineModel } from '../types';
import { PLANAR_OPTIONS, planarDataset, type PlanarKind } from '../data/planar';
import { accuracy, input } from '../util';
import { fieldProcedure, fieldViews, logitMargin } from './planar-common';

type ResMlpConfig = {
  dataset: PlanarKind;
  noise: number;
  blocks: number;
  width: number;
  hidden: number;
  activation: ActivationName;
  norm: boolean;
  residual: boolean;
};

function forward(x: Var, c: ResMlpConfig) {
  const act = activations[c.activation];
  let h = scope('embed', () => dense(x, 'embed'));
  for (let l = 0; l < c.blocks; l++) {
    const name = `block${l + 1}`;
    h = scope(name, () => {
      const u = c.norm
        ? scope('norm', () => layerNorm(h, p(`${name}.norm.weight`), p(`${name}.norm.bias`)))
        : h;
      const f = dense(act(dense(u, `${name}.fc1`)), `${name}.fc2`);
      return c.residual ? add(h, f) : f;
    });
  }
  return scope('output', () => dense(act(h), 'output'));
}

const resmlp = defineModel<ResMlpConfig>({
  id: 'resmlp',
  family: 'feedforward',
  name: 'Residual MLP',
  summary:
    'Fully connected blocks that add their output to their input, so each block learns a correction to the identity.',
  signature: 'h_{\\ell} = h_{\\ell-1} + W_2\\,\\sigma(W_1\\,\\mathrm{LN}(h_{\\ell-1}))',
  fidelity: 'reduced',
  scaleNote:
    'Fully connected residual blocks in the pre-activation ordering of He et al. (2016): the normalisation precedes the weights and the sum is not followed by a nonlinearity. Layer normalisation replaces batch normalisation, and each block has one normalisation and one nonlinearity (LN → W₁ → σ → W₂) rather than the paper’s two; depth and width are chosen for two-dimensional inputs.',
  references: [
    {
      authors: 'K. He, X. Zhang, S. Ren, J. Sun',
      title: 'Identity mappings in deep residual networks',
      year: 2016,
      url: 'https://arxiv.org/abs/1603.05027',
    },
  ],
  limitations: [
    'Two-dimensional synthetic data only. Turning off the skip connection keeps all parameters and only removes the addition, so the same weights can be compared with and without it.',
  ],
  guide: {
    what: 'A residual block computes a correction and adds it to its input instead of replacing it, so the signal has a straight path (a skip connection) through the whole network. This one maps a point in the plane through a first linear layer, six such blocks and an output layer to two class probabilities. Training learns all the weights; the skips also carry the training signal (the gradient) straight back to the early blocks, which matters once many blocks are stacked.',
    watch:
      'This network solves the spirals quickly: in this run accuracy passed 90 % by step 50 and the loss ended near 0.01 after 300 steps (it also has ten times the parameters of the MLP page). Watch the decision field tighten around the arms, then switch the skip connections off in the experiments below.',
    try: [
      {
        label: 'Skips off, 12 blocks',
        note: 'Without skips a 12-block stack sits at chance (loss 0.69) for over 200 steps before it begins to learn.',
        patch: { blocks: 12, residual: false },
        train: 300,
      },
      {
        label: 'Skips on, 12 blocks',
        note: 'The same depth with skips learns as fast as six blocks: 96 % by step 50 in this run.',
        patch: { blocks: 12, residual: true },
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
    {
      key: 'blocks',
      label: 'Residual blocks',
      kind: 'int',
      min: 1,
      max: 12,
      affects: 'architecture',
    },
    { key: 'width', label: 'Stream width', kind: 'int', min: 2, max: 64, affects: 'architecture' },
    {
      key: 'hidden',
      label: 'Block hidden width',
      kind: 'int',
      min: 2,
      max: 128,
      affects: 'architecture',
    },
    { key: 'norm', label: 'Layer normalisation', kind: 'bool', affects: 'architecture' },
    {
      key: 'activation',
      label: 'Activation σ',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'relu', label: 'ReLU' },
        { value: 'gelu', label: 'GELU' },
        { value: 'tanh', label: 'tanh' },
        { value: 'silu', label: 'SiLU' },
      ],
    },
    {
      key: 'residual',
      label: 'Skip connections',
      kind: 'bool',
      affects: 'forward',
      help: 'Off: h_ℓ = F(h_{ℓ−1}), with the same parameters.',
    },
  ],
  defaults: {
    dataset: 'spirals',
    noise: 0.08,
    blocks: 6,
    width: 16,
    hidden: 32,
    activation: 'relu',
    norm: true,
    residual: true,
  },
  init(c, rng) {
    const b = new ParamBuilder(rng);
    declareLinear(b, 'embed', 2, c.width);
    for (let l = 0; l < c.blocks; l++) {
      const name = `block${l + 1}`;
      if (c.norm) declareNorm(b, `${name}.norm`, c.width);
      declareLinear(b, `${name}.fc1`, c.width, c.hidden);
      declareLinear(b, `${name}.fc2`, c.hidden, c.width);
    }
    declareLinear(b, 'output', c.width, 2);
    return b.store;
  },
  datasets: [planarDataset<ResMlpConfig>((c) => c.dataset, { noise: (c) => c.noise })],
  forward(batch, c) {
    const logits = forward(input('x', batch.inputs.x), c);
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
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 0.005 },
    batchSize: 32,
    steps: 300,
    validateEvery: 25,
  },
  procedures: [
    fieldProcedure<ResMlpConfig>(
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
    embed: { tex: 'h_0 = W_e x + b_e' },
    block: {
      tex: 'h_{\\ell} = h_{\\ell-1} + W_{2}\\,\\sigma\\!\\left(W_{1}\\,\\mathrm{LN}(h_{\\ell-1}) + b_1\\right) + b_2',
      text: 'Because ∂h_ℓ/∂h_{ℓ−1} = I + ∂F/∂h_{ℓ−1}, the gradient reaching early blocks always contains an identity path.',
    },
    output: { tex: 'z = W_o\\,\\sigma(h_L) + b_o' },
    loss: { tex: 'L = -\\tfrac{1}{N}\\textstyle\\sum_n \\log p_{n,y_n}' },
  },
});

export default resmlp;
