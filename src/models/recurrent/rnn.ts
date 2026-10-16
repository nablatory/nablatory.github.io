import { add, defaultOptimizer, linear, p, relu, tanh } from '../../engine';
import { defineModel } from '../types';
import { recurrentDataset } from '../data/recurrent-sequences';
import {
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

type RnnConfig = RecurrentBase & { activation: 'tanh' | 'relu' };

export const rnnCellDef: CellDef<RnnConfig> = {
  blocks: 1,
  gateNames: ['a'],
  stateNames: ['h'],
  // a_t = W_ih x_t + b_ih + W_hh h_{t-1} + b_hh
  gates: (xp, prev) => ({ a: add(xp, linear(prev.h, p('cell.weight_hh'), p('cell.bias_hh'))) }),
  update: (g, _prev, c) => ({ h: c.activation === 'relu' ? relu(g.a) : tanh(g.a) }),
};

const rnn = defineModel<RnnConfig>({
  id: 'rnn',
  family: 'recurrent',
  name: 'Elman recurrent network',
  summary:
    'One hidden state updated from the previous state and the current input by the same affine map and nonlinearity at every step.',
  signature: 'h_t = \\tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})',
  fidelity: 'equations',
  scaleNote:
    'The Elman update with PyTorch’s parameterisation (two bias vectors) and initialisation. Hidden size and tasks are small so it trains in the browser.',
  references: [
    {
      authors: 'J. L. Elman',
      title: 'Finding structure in time',
      year: 1990,
      url: 'https://doi.org/10.1207/s15516709cog1402_1',
    },
    {
      authors: 'Y. Bengio, P. Simard, P. Frasconi',
      title: 'Learning long-term dependencies with gradient descent is difficult',
      year: 1994,
      url: 'https://doi.org/10.1109/72.279181',
    },
    {
      authors: 'R. Pascanu, T. Mikolov, Y. Bengio',
      title: 'On the difficulty of training recurrent neural networks',
      year: 2013,
      url: 'https://arxiv.org/abs/1211.5063',
    },
    {
      authors: 'Q. V. Le, N. Jaitly, G. E. Hinton',
      title: 'A simple way to initialize recurrent networks of rectified linear units',
      year: 2015,
      url: 'https://arxiv.org/abs/1504.00941',
    },
  ],
  limitations: [
    'Single layer, unidirectional, trained by full backpropagation through time on short synthetic sequences.',
    'The ReLU option uses the default initialisation, not the identity initialisation of Le et al. (2015), so it can diverge without gradient clipping.',
    'Elman’s original network fed back a copy of the hidden layer through fixed one-to-one context units, used logistic units, and was trained by ordinary backpropagation with the context units treated as inputs, so no error flowed back through time; this is the now-standard equivalent with a learned W_hh, trained through every step.',
  ],
  guide: {
    what: 'A recurrent network reads a sequence one step at a time and keeps a hidden state (32 numbers) that it updates from the current input and its previous value, using the same weights at every step. The input here is a sequence of tokens: a symbol, then distractors, then a query, and the output at the last step must name the symbol seen at the first. Training by backpropagation through time, which unrolls the steps and sends the error back through all of them, learns the shared weights.',
    watch:
      'With a 20-step gap the plain RNN struggles: typically only about half the answers are right after 300 steps (loss near 0.8, where guessing among four gives 1.39). Watch the answer at the query step, then run “Gradient through time”: at initialisation the error signal shrinks by about five orders of magnitude on its way back to step 0.',
    try: [
      {
        label: 'Shorter gap',
        note: 'With a 10-step gap every answer is right within about 50 steps.',
        patch: { length: 10 },
        train: 100,
      },
      {
        label: 'Sine continuation',
        note: 'Predict the next sample of a sine wave: the loss drops well below the copy-last baseline of 0.076 within 50 steps.',
        patch: { task: 'sine' },
        train: 300,
      },
    ],
    views: ['Answer at the query step', 'Hidden states h_t (rows: t, columns: units)'],
  },
  configSchema: [
    taskField,
    lengthField,
    hiddenField,
    {
      key: 'activation',
      label: 'Activation',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'tanh', label: 'tanh' },
        { value: 'relu', label: 'ReLU' },
      ],
    },
  ],
  defaults: { task: 'recall', length: 20, hidden: 32, activation: 'tanh' },
  validate: (c) =>
    c.length < 3
      ? 'The recall task needs at least three steps: symbol, one distractor, query.'
      : null,
  init: (c, rng) => initRecurrent(rnnCellDef, c, rng),
  datasets: [recurrentDataset<RnnConfig>()],
  forward: (batch, c) => unroll(rnnCellDef, batch, c).result,
  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 1e-2, clip: 'norm', clipValue: 1 },
    batchSize: 32,
    steps: 300,
    validateEvery: 25,
  },
  procedures: [gradientProcedure(rnnCellDef)],
  views: [
    ...commonViews,
    { kind: 'matrix', title: 'Pre-activations a_t (rows: t, columns: units)', source: 'out:a' },
  ],
  math: {
    gates: {
      tex: 'a_t = W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh}',
      text: 'The input projections for all steps are computed in one product; the recurrent term is added step by step.',
    },
    state: {
      tex: 'h_t = \\sigma(a_t),\\quad \\sigma \\in \\{\\tanh, \\operatorname{ReLU}\\},\\quad h_{-1} = 0',
      text: 'The dashed edge back to the gates is the recurrence. Backpropagation multiplies by Wₕₕᵀ diag(σ′(aₜ)) once per step, which is why ∂L/∂hₜ tends to shrink (or grow) geometrically with distance from the loss.',
    },
    ...commonMath,
  },
});

export default rnn;
