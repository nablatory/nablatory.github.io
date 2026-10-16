import { add, defaultOptimizer, linear, mul, p, sigmoid, tanh } from '../../engine';
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

type LstmConfig = RecurrentBase & { forgetBias: number };

/** PyTorch gate layout [i | f | g | o]. */
export const lstmCellDef: CellDef<LstmConfig> = {
  blocks: 4,
  gateNames: ['i', 'f', 'g', 'o'],
  stateNames: ['h', 'c'],
  gates(xp, prev, c) {
    const H = c.hidden;
    const z = add(xp, linear(prev.h, p('cell.weight_hh'), p('cell.bias_hh')));
    return {
      i: sigmoid(block(z, 0, H)),
      f: sigmoid(block(z, 1, H)),
      g: tanh(block(z, 2, H)),
      o: sigmoid(block(z, 3, H)),
    };
  },
  update(g, prev) {
    const c = add(mul(g.f, prev.c), mul(g.i, g.g));
    return { h: mul(g.o, tanh(c)), c };
  },
  adjustBias(bias, H, c) {
    if (!c.forgetBias) return null;
    for (let j = H; j < 2 * H; j++) bias.data[j] += c.forgetBias;
    return `forget block + ${c.forgetBias}`;
  },
};

const lstm = defineModel<LstmConfig>({
  id: 'lstm',
  family: 'recurrent',
  name: 'Long short-term memory',
  summary:
    'A memory cell updated additively and guarded by input, forget and output gates, so error signals can cross many steps without vanishing.',
  signature: 'c_t = f_t \\odot c_{t-1} + i_t \\odot g_t,\\quad h_t = o_t \\odot \\tanh(c_t)',
  fidelity: 'equations',
  scaleNote:
    'The LSTM with forget gate (Gers et al. 2000) in PyTorch’s layout and initialisation, without peephole connections. Hidden size and tasks are small so it trains in the browser.',
  references: [
    {
      authors: 'S. Hochreiter, J. Schmidhuber',
      title: 'Long short-term memory',
      year: 1997,
      url: 'https://doi.org/10.1162/neco.1997.9.8.1735',
    },
    {
      authors: 'F. A. Gers, J. Schmidhuber, F. Cummins',
      title: 'Learning to forget: continual prediction with LSTM',
      year: 2000,
      url: 'https://doi.org/10.1162/089976600300015015',
    },
    {
      authors: 'R. Jozefowicz, W. Zaremba, I. Sutskever',
      title: 'An empirical exploration of recurrent network architectures',
      year: 2015,
      url: 'https://proceedings.mlr.press/v37/jozefowicz15.html',
    },
  ],
  limitations: [
    'Single layer, unidirectional, trained by full backpropagation through time. The 1997 paper truncated the gradient at the cell; here it flows through every gate.',
    'No peephole connections (Gers & Schmidhuber 2000) and no projection layer.',
  ],
  guide: {
    what: 'An LSTM is a recurrent network with a separate memory cell: at each step a forget gate decides how much of the cell to keep, an input gate how much new content to write, and an output gate how much of the cell to show as the hidden state. Gates are numbers between 0 and 1 computed from the input and previous state. It reads the same delayed-recall sequences as the RNN page (symbol, distractors, query), and training learns the gate and cell weights by backpropagation through time.',
    watch:
      'Unlike the plain RNN, the LSTM solves the 20-step recall: the loss falls from 1.39 to almost zero within 50–200 steps. Watch the forget gate matrix: with the +2 bias offset gates start near 0.9, so the cell keeps its contents across the distractors. Run “Gradient through time”: along the cell about half the error signal survives at initialisation.',
    try: [
      {
        label: 'No forget-gate offset',
        note: 'Without the offset this run stayed at chance (loss 1.39) for all 300 steps.',
        patch: { forgetBias: 0 },
        train: 300,
      },
      {
        label: 'Sine continuation',
        note: 'Next-sample prediction: loss 0.005 after 300 steps, against 0.076 for repeating the last sample.',
        patch: { task: 'sine' },
        train: 300,
      },
    ],
    views: ['Answer at the query step', 'Cell state c_t (rows: t, columns: units)'],
  },
  configSchema: [
    taskField,
    lengthField,
    hiddenField,
    {
      key: 'forgetBias',
      label: 'Initial forget-gate bias offset',
      kind: 'float',
      min: 0,
      max: 3,
      step: 0.5,
      affects: 'architecture',
      help: 'Added to b_if at initialisation. A positive offset starts with f ≈ 1, i.e. remembering by default (Jozefowicz et al. 2015).',
    },
  ],
  defaults: { task: 'recall', length: 20, hidden: 32, forgetBias: 2 },
  validate: (c) =>
    c.length < 3
      ? 'The recall task needs at least three steps: symbol, one distractor, query.'
      : null,
  init: (c, rng) => initRecurrent(lstmCellDef, c, rng),
  datasets: [recurrentDataset<LstmConfig>()],
  forward: (batch, c) => unroll(lstmCellDef, batch, c).result,
  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 1e-2, clip: 'norm', clipValue: 1 },
    batchSize: 32,
    steps: 300,
    validateEvery: 25,
  },
  procedures: [gradientProcedure(lstmCellDef)],
  views: [
    ...commonViews,
    {
      kind: 'series',
      title: 'Gradient through time along the cell, ‖∂L/∂c_t‖',
      sources: [{ ref: 'proc:gradients:cNorm', label: '‖∂L/∂c_t‖' }],
      xLabel: 't',
      logY: true,
      unavailable: 'Run “Gradient through time” to compute it.',
    },
    {
      kind: 'series',
      title: 'Gates of one unit over time (choose the unit)',
      sources: [
        { ref: 'out:f', label: 'forget f' },
        { ref: 'out:i', label: 'input i' },
        { ref: 'out:o', label: 'output o' },
      ],
      xLabel: 't',
    },
    { kind: 'matrix', title: 'Forget gate f_t (rows: t, columns: units)', source: 'out:f' },
    { kind: 'matrix', title: 'Cell state c_t (rows: t, columns: units)', source: 'out:c' },
  ],
  math: {
    gates: {
      tex: '\\begin{aligned} i_t &= \\sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})\\\\ f_t &= \\sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})\\\\ g_t &= \\tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})\\\\ o_t &= \\sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\end{aligned}',
      text: 'All four gates come from one product with W_ih (4H × input) and one with W_hh (4H × H), sliced in the order i, f, g, o.',
    },
    state: {
      tex: 'c_t = f_t \\odot c_{t-1} + i_t \\odot g_t,\\qquad h_t = o_t \\odot \\tanh(c_t)',
      text: 'Along the cell path, ∂c_t/∂c_{t−1} = diag(f_t) (plus terms through the gates). With f ≈ 1 the error signal passes back nearly unchanged: the constant error carousel.',
    },
    ...commonMath,
  },
});

export default lstm;
