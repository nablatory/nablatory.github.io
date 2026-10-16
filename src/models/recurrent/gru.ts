import { add, addScalar, defaultOptimizer, linear, mul, neg, p, sigmoid, tanh } from '../../engine';
import { defineModel } from '../types';
import { recurrentDataset } from '../data/recurrent-sequences';
import {
  block,
  commonMath,
  commonViews,
  gradientProcedure,
  hiddenField,
  initRecurrent,
  lengthField,
  taskField,
  unroll,
  type CellDef,
  type RecurrentBase,
} from './common';

type GruConfig = RecurrentBase & { updateBias: number };

/** PyTorch gate layout [r | z | n]. */
export const gruCellDef: CellDef<GruConfig> = {
  blocks: 3,
  gateNames: ['r', 'z', 'n'],
  stateNames: ['h'],
  gates(xp, prev, c) {
    const H = c.hidden;
    const gh = linear(prev.h, p('cell.weight_hh'), p('cell.bias_hh'));
    const r = sigmoid(add(block(xp, 0, H), block(gh, 0, H)));
    const z = sigmoid(add(block(xp, 1, H), block(gh, 1, H)));
    const n = tanh(add(block(xp, 2, H), mul(r, block(gh, 2, H))));
    return { r, z, n };
  },
  update(g, prev) {
    return { h: add(mul(addScalar(neg(g.z), 1), g.n), mul(g.z, prev.h)) };
  },
  adjustBias(bias, H, c) {
    if (!c.updateBias) return null;
    for (let j = H; j < 2 * H; j++) bias.data[j] += c.updateBias;
    return `update block + ${c.updateBias}`;
  },
};

const gru = defineModel<GruConfig>({
  id: 'gru',
  family: 'recurrent',
  name: 'Gated recurrent unit',
  summary:
    'A hidden state that is interpolated between its previous value and a candidate, with an update gate z and a reset gate r and no separate memory cell.',
  signature: 'h_t = (1 - z_t) \\odot n_t + z_t \\odot h_{t-1}',
  fidelity: 'equations',
  scaleNote:
    'The GRU in PyTorch’s formulation (reset applied after the recurrent product, as in cuDNN) and initialisation. Hidden size and tasks are small so it trains in the browser.',
  references: [
    {
      authors:
        'K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, Y. Bengio',
      title:
        'Learning phrase representations using RNN encoder–decoder for statistical machine translation',
      year: 2014,
      url: 'https://arxiv.org/abs/1406.1078',
    },
    {
      authors: 'J. Chung, C. Gulcehre, K. Cho, Y. Bengio',
      title: 'Empirical evaluation of gated recurrent neural networks on sequence modeling',
      year: 2014,
      url: 'https://arxiv.org/abs/1412.3555',
    },
  ],
  limitations: [
    'Single layer, unidirectional, trained by full backpropagation through time on short synthetic sequences.',
    'Cho et al. (2014) apply the reset gate before the recurrent product, n = tanh(W x + U(r ⊙ h)); this uses the PyTorch/cuDNN variant n = tanh(W x + r ⊙ (U h + b)), which differs slightly.',
    'Cho et al. write h_t = z ⊙ h_{t−1} + (1 − z) ⊙ h̃ with the same convention as here; some texts swap the roles of z and 1 − z.',
  ],
  guide: {
    what: 'A gated recurrent unit is a simpler gated cell than the LSTM, with no separate memory: an update gate z blends the previous hidden state with a freshly computed candidate (z near 1 keeps the old state, near 0 replaces it), and a reset gate r controls how much of the old state the candidate sees. It reads the same delayed-recall sequences (symbol, distractors, query), and training learns the gate and candidate weights by backpropagation through time.',
    watch:
      'The GRU solves the 20-step recall within about 50–100 steps: the loss falls from 1.39 to nearly zero. Watch the update gate matrix: with the +2 bias offset z starts near 0.9 everywhere, so the state is mostly carried forward from step to step. Compare with the plain RNN, which fails at this gap.',
    try: [
      {
        label: 'No update-gate offset',
        note: 'Without the offset the GRU still learns, later: chance until about step 100, then solved by 150 in this run.',
        patch: { updateBias: 0 },
        train: 300,
      },
      {
        label: 'Gap of 50 steps',
        note: 'Still solved, within about 150 steps in this run; the LSTM at the same gap did not learn in 300.',
        patch: { length: 50 },
        train: 300,
      },
    ],
    views: ['Answer at the query step', 'Update gate z_t (rows: t, columns: units)'],
  },
  configSchema: [
    taskField,
    lengthField,
    hiddenField,
    {
      key: 'updateBias',
      label: 'Initial update-gate bias offset',
      kind: 'float',
      min: 0,
      max: 3,
      step: 0.5,
      affects: 'architecture',
      help: 'Added to b_iz at initialisation. A positive offset starts with z ≈ 1, i.e. carrying the state forward by default.',
    },
  ],
  defaults: { task: 'recall', length: 20, hidden: 32, updateBias: 2 },
  validate: (c) =>
    c.length < 3
      ? 'The recall task needs at least three steps: symbol, one distractor, query.'
      : null,
  init: (c, rng) => initRecurrent(gruCellDef, c, rng),
  datasets: [recurrentDataset<GruConfig>()],
  forward: (batch, c) => unroll(gruCellDef, batch, c).result,
  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 1e-2, clip: 'norm', clipValue: 1 },
    batchSize: 32,
    steps: 300,
    validateEvery: 25,
  },
  procedures: [gradientProcedure(gruCellDef)],
  views: [
    ...commonViews,
    {
      kind: 'series',
      title: 'Gates of one unit over time (choose the unit)',
      sources: [
        { ref: 'out:z', label: 'update z' },
        { ref: 'out:r', label: 'reset r' },
      ],
      xLabel: 't',
    },
    { kind: 'matrix', title: 'Update gate z_t (rows: t, columns: units)', source: 'out:z' },
  ],
  math: {
    gates: {
      tex: '\\begin{aligned} r_t &= \\sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})\\\\ z_t &= \\sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz})\\\\ n_t &= \\tanh(W_{in} x_t + b_{in} + r_t \\odot (W_{hn} h_{t-1} + b_{hn})) \\end{aligned}',
      text: 'The reset gate r decides how much of the previous state enters the candidate n.',
    },
    state: {
      tex: 'h_t = (1 - z_t) \\odot n_t + z_t \\odot h_{t-1}',
      text: 'With z = 1 the state is copied unchanged and ∂h_t/∂h_{t−1} = I: the GRU’s route around vanishing gradients.',
    },
    ...commonMath,
  },
});

export default gru;
