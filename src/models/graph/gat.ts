import {
  ParamBuilder,
  Tensor,
  T,
  add,
  concat,
  defaultOptimizer,
  dropout,
  elu,
  leakyRelu,
  linear,
  maskedFill,
  matmul,
  mean,
  p,
  scope,
  slice,
  softmax,
  stack,
  type Var,
} from '../../engine';
import { defineModel } from '../types';
import { featureDim, graphDataset, numClasses, type GraphDataConfig } from '../data/graphs';
import { constant } from '../util';
import {
  graphDataDefaults,
  graphDataFields,
  graphLimitations,
  graphViews,
  input,
  nodeClassification,
} from './common';

export type GatConfig = GraphDataConfig & {
  layers: number;
  heads: number;
  hidden: number;
  outHeads: number;
  dropout: number;
  slope: number;
};

/** [heads, width per head] of each layer; the last layer outputs the classes. */
export function gatShapes(c: GatConfig): { heads: number; width: number; concat: boolean }[] {
  const out = Array.from({ length: c.layers - 1 }, () => ({
    heads: c.heads,
    width: c.hidden,
    concat: true,
  }));
  out.push({ heads: c.outHeads, width: numClasses(c.graph), concat: false });
  return out;
}

/** 1 where j ∉ N(i) ∪ {i}: these logits are set to −∞ before the softmax. */
export function nonNeighbourMask(adj: Tensor): Tensor {
  const n = adj.shape[0];
  const m = Tensor.zeros([n, n]);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) m.data[i * n + j] = i !== j && adj.data[i * n + j] === 0 ? 1 : 0;
  return m;
}

/**
 * One attention head (dense, masked):
 *   e_ij = LeakyReLU(a₁ᵀ W h_i + a₂ᵀ W h_j) for j ∈ N(i) ∪ {i}, −∞ otherwise,
 *   α = softmax_j(e),  h'_i = Σ_j α_ij W h_j.
 */
export function gatHead(
  h: Var,
  mask: Tensor,
  name: string,
  slope: number,
  drop: number,
): { out: Var; alpha: Var } {
  const wh = linear(h, p(`${name}.weight`));
  const s = linear(wh, p(`${name}.attn`)); // [N, 2]: a₁ᵀ W h_i and a₂ᵀ W h_i
  const e = leakyRelu(add(slice(s, 1, 0, 1), T(slice(s, 1, 1, 2))), slope);
  const alpha = softmax(maskedFill(e, mask, -Infinity), -1);
  return { out: matmul(dropout(alpha, drop), wh), alpha };
}

const gat = defineModel<GatConfig>({
  id: 'gat',
  family: 'graph',
  name: 'Graph attention network',
  summary:
    'Each node attends over its neighbours: a learned score of every (node, neighbour) pair is normalised with a softmax over the neighbourhood and used to average the neighbours’ transformed features, with several heads in parallel.',
  signature:
    "\\alpha_{ij} = \\operatorname{softmax}_{j \\in \\mathcal N_i}\\big(\\mathrm{LeakyReLU}(a^{\\top}[W h_i \\Vert W h_j])\\big),\\quad h_i' = \\Vert_{k=1}^{K} \\sigma\\big(\\textstyle\\sum_{j} \\alpha^k_{ij} W^k h_j\\big)",
  fidelity: 'reduced',
  scaleNote:
    'The attention mechanism, masking to N(i) ∪ {i}, LeakyReLU slope 0.2, ELU, Glorot initialisation, multi-head concatenation in hidden layers, head averaging in the output layer, and dropout on both layer inputs and on the normalised coefficients are as published. The default has 4 heads × 4 units and 2 averaged output heads on 34- and 60-node graphs, instead of 8 heads × 8 units with a single output head on Cora and Citeseer (8 averaged output heads on Pubmed); dropout is 0.2 instead of 0.6, the learning rate 0.01 instead of 0.005, the L2 penalty 5 × 10⁻⁴ is applied to every parameter (biases included) through the optimiser, and there is no early stopping.',
  references: [
    {
      authors: 'P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio',
      title: 'Graph attention networks',
      year: 2018,
      url: 'https://arxiv.org/abs/1710.10903',
    },
    {
      authors: 'W. W. Zachary',
      title: 'An information flow model for conflict and fission in small groups',
      year: 1977,
      url: 'https://doi.org/10.1086/jar.33.4.3629752',
    },
  ],
  limitations: [
    ...graphLimitations,
    'Attention is computed densely for all N² pairs and masked with −∞ outside each neighbourhood, so non-neighbours get exactly zero weight. The paper’s sparse implementation computes the same coefficients on edges only.',
    'Dropout is applied to the layer inputs and to the normalised coefficients, as in the paper; the shown coefficients are those before dropout.',
  ],
  guide: {
    what: 'A node-labelling network in which each node decides how much to listen to each neighbour. For every edge a small learned score compares the two nodes’ transformed vectors; a softmax over the neighbourhood turns the scores into attention weights that sum to one, and the node’s new vector is the weighted average of its neighbours’ and its own. Four heads do this in parallel and are concatenated; two output heads are averaged into class scores. The transformation and scoring weights are learned from the four labelled nodes.',
    watch:
      '“Attention coefficients α, layer 1” shows a 34 × 34 matrix per head: rows sum to one and are zero off the graph’s edges. Initially each node spreads its weight almost evenly over its neighbourhood; as the loss on the four labelled nodes falls, the weights become uneven. “Predicted class” matches “True class” on about 0.87–0.93 of held-out members.',
    try: [
      {
        label: 'One head per layer',
        note: 'A single attention matrix instead of four: held-out accuracy still about 0.87–0.93 on this small graph.',
        patch: { heads: 1, outHeads: 1 },
        train: 200,
      },
      {
        label: 'Paper’s dropout 0.6',
        note: 'Heavier dropout on inputs and attention weights adds noise to the training loss; held-out accuracy stays about 0.93–0.97.',
        patch: { dropout: 0.6 },
        train: 200,
      },
      {
        label: 'Block model',
        note: 'Three blocks with noisy 8-dimensional features: held-out accuracy about 0.8–0.9.',
        patch: { graph: 'sbm' },
        train: 200,
      },
    ],
    views: ['Attention coefficients α, layer 1', 'Predicted class (arg max of softmax)'],
  },
  configSchema: [
    ...graphDataFields,
    {
      key: 'layers',
      label: 'Layers (incl. output)',
      kind: 'int',
      min: 2,
      max: 3,
      affects: 'architecture',
    },
    { key: 'heads', label: 'Hidden heads K', kind: 'int', min: 1, max: 8, affects: 'architecture' },
    {
      key: 'hidden',
      label: 'Units per head',
      kind: 'int',
      min: 1,
      max: 16,
      affects: 'architecture',
    },
    {
      key: 'outHeads',
      label: 'Output heads (averaged)',
      kind: 'int',
      min: 1,
      max: 6,
      affects: 'architecture',
    },
    {
      key: 'slope',
      label: 'LeakyReLU slope',
      kind: 'float',
      min: 0,
      max: 0.5,
      step: 0.05,
      affects: 'forward',
    },
    {
      key: 'dropout',
      label: 'Dropout rate',
      kind: 'float',
      min: 0,
      max: 0.8,
      step: 0.05,
      affects: 'forward',
    },
  ],
  defaults: {
    ...graphDataDefaults,
    layers: 2,
    heads: 4,
    hidden: 4,
    outHeads: 2,
    slope: 0.2,
    dropout: 0.2,
  },
  init(c, rng) {
    const b = new ParamBuilder(rng);
    let nIn = featureDim(c.graph);
    gatShapes(c).forEach(({ heads, width, concat: cat }, l) => {
      for (let k = 0; k < heads; k++) {
        const name = `gat${l + 1}.head${k + 1}`;
        b.add(`${name}.weight`, [width, nIn], { kind: 'xavierUniform', fanIn: nIn, fanOut: width });
        b.add(`${name}.attn`, [2, width], { kind: 'xavierUniform', fanIn: 2 * width, fanOut: 1 });
      }
      b.add(`gat${l + 1}.bias`, [cat ? heads * width : width], { kind: 'zeros' });
      nIn = cat ? heads * width : width;
    });
    return b.store;
  },
  datasets: [graphDataset<GatConfig>()],
  forward(batch, c) {
    const x = input('x', batch.inputs.x);
    const maskT = nonNeighbourMask(batch.inputs.adj);
    const shown = scope('neighbourhood', () => {
      const t = Tensor.zeros(maskT.shape);
      for (let i = 0; i < t.size; i++) t.data[i] = 1 - maskT.data[i];
      return constant(t, 'neighbourhoodMask');
    });
    const shapes = gatShapes(c);
    let h = x;
    let embedding: Var = x;
    let attention: Var | undefined;
    shapes.forEach(({ heads, concat: cat }, l) => {
      const name = `gat${l + 1}`;
      h = scope(name, () => {
        const hin = dropout(h, c.dropout);
        const outs: Var[] = [];
        const alphas: Var[] = [];
        for (let k = 0; k < heads; k++) {
          const r = scope(`head${k + 1}`, () =>
            gatHead(hin, maskT, `${name}.head${k + 1}`, c.slope, c.dropout),
          );
          outs.push(r.out);
          alphas.push(r.alpha);
        }
        if (l === 0) attention = stack(alphas, 0);
        const bias = p(`${name}.bias`);
        // Hidden layers: concatenate heads, then ELU. Output layer: average heads.
        return cat ? elu(add(concat(outs, 1), bias)) : add(mean(stack(outs, 0), 0), bias);
      });
      if (l < shapes.length - 1) embedding = h;
    });
    const emb =
      embedding.shape[1] === 2
        ? embedding
        : scope(`gat${shapes.length - 1}`, () => slice(embedding, 1, 0, 2));
    return nodeClassification(batch, h, { embedding: emb, adjNorm: shown, attention });
  },
  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 0.01, weightDecay: 5e-4 },
    batchSize: 1,
    steps: 200,
    validateEvery: 10,
  },
  views: [
    ...graphViews(
      'Last hidden layer, first two units (head 1)',
      'Attention mask: neighbourhood N(i) ∪ {i}',
    ),
    { kind: 'attention', title: 'Attention coefficients α, layer 1', source: 'out:attention' },
  ],
  math: {
    neighbourhood: {
      tex: '\\mathcal N_i = \\{ j : A_{ij} = 1 \\} \\cup \\{ i \\}',
      text: 'Masked attention: only first-order neighbours and the node itself receive non-zero weight.',
    },
    gat: {
      tex: "h_i' = \\Big\\Vert_{k=1}^{K} \\mathrm{ELU}\\Big(\\sum_{j \\in \\mathcal N_i} \\alpha^k_{ij} W^k h_j\\Big)\\ \\text{(hidden)},\\qquad z_i = \\frac{1}{K}\\sum_{k=1}^{K} \\sum_{j \\in \\mathcal N_i} \\alpha^k_{ij} W^k h_j\\ \\text{(output)}",
      text: 'Hidden layers concatenate the K heads; the output layer averages them before the softmax.',
    },
    head: {
      tex: 'e_{ij} = \\mathrm{LeakyReLU}\\big(a_1^{\\top} W h_i + a_2^{\\top} W h_j\\big),\\quad \\alpha_{ij} = \\frac{\\exp e_{ij}}{\\sum_{l \\in \\mathcal N_i} \\exp e_{il}}',
      text: 'aᵀ[Wh_i ‖ Wh_j] splits into a source and a target term, so all N² scores are one broadcast sum. Non-neighbours are set to −∞, which the softmax maps to exactly 0.',
    },
    loss: {
      tex: 'L = -\\frac{1}{|\\mathcal Y_L|} \\sum_{i \\in \\mathcal Y_L} \\log \\operatorname{softmax}(z_i)_{y_i}',
    },
  },
});

export default gat;
