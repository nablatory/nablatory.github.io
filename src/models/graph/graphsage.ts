import {
  ParamBuilder,
  Tensor,
  add,
  concat,
  declareLinear,
  defaultOptimizer,
  dense,
  dropout,
  graphRng,
  isTraining,
  matmul,
  max,
  maskedFill,
  relu,
  reshape,
  scope,
  slice,
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
  l2NormalizeRows,
  nodeClassification,
} from './common';

export type SageConfig = GraphDataConfig & {
  layers: number;
  hidden: number;
  embed2d: boolean;
  aggregator: 'mean' | 'pool';
  normalize: boolean;
  sample: number;
  dropout: number;
};

export function sageWidths(c: SageConfig): number[] {
  const hidden = Array.from({ length: c.layers - 1 }, () => c.hidden);
  if (c.embed2d && hidden.length) hidden[hidden.length - 1] = 2;
  return [...hidden, numClasses(c.graph)];
}

/**
 * Neighbourhood used by the aggregator, as an [N, N] weight matrix S with
 * S_vu = (number of times u was drawn for v) / (draws for v). Full
 * neighbourhood: S = D⁻¹A. With sampling (training only): `k` neighbours per
 * node drawn uniformly with replacement from the graph rng.
 */
export function neighbourWeights(adj: Tensor, k: number): Tensor {
  const n = adj.shape[0];
  const out = Tensor.zeros([n, n]);
  const sampled = k > 0 && isTraining();
  const rng = sampled ? graphRng() : null;
  for (let v = 0; v < n; v++) {
    const nb: number[] = [];
    for (let u = 0; u < n; u++) if (adj.data[v * n + u] !== 0) nb.push(u);
    if (!nb.length) continue;
    if (rng) {
      for (let s = 0; s < k; s++) out.data[v * n + nb[rng.int(nb.length)]] += 1 / k;
    } else {
      for (const u of nb) out.data[v * n + u] = 1 / nb.length;
    }
  }
  return out;
}

/** Mean aggregator: AGG_v = Σ_u S_vu h_u. */
export function meanAggregate(h: Var, weights: Var): Var {
  return matmul(weights, h);
}

/**
 * Max-pooling aggregator: AGG_v = max_{u ∈ N(v)} σ(W_pool h_u + b), elementwise.
 * Dense masked form: broadcast to [N, N, d], fill non-neighbours with −∞, max over u.
 */
export function poolAggregate(h: Var, weights: Tensor, name: string): Var {
  const n = h.shape[0];
  const p = relu(dense(h, name));
  const d = p.shape[1];
  const mask = Tensor.zeros([n, n, d]);
  for (let v = 0; v < n; v++)
    for (let u = 0; u < n; u++)
      if (weights.data[v * n + u] === 0) mask.data.fill(1, (v * n + u) * d, (v * n + u + 1) * d);
  const tiled = add(reshape(p, [1, n, d]), constant(Tensor.zeros([n, 1, d]), 'broadcast'));
  return max(maskedFill(tiled, mask, -Infinity), 1);
}

const graphsage = defineModel<SageConfig>({
  id: 'graphsage',
  family: 'graph',
  name: 'GraphSAGE',
  summary:
    'Each node concatenates its own representation with an aggregate of its neighbours’ (mean or max-pooling), applies a shared linear map and a nonlinearity, and normalises the result to unit length.',
  signature:
    'h_v^{k} = \\sigma\\big(W^{k} \\big[\\, h_v^{k-1} \\,\\Vert\\, \\mathrm{AGG}_k\\{h_u^{k-1} : u \\in \\mathcal N(v)\\} \\big]\\big),\\quad h_v^k \\leftarrow h_v^k / \\lVert h_v^k \\rVert_2',
  fidelity: 'reduced',
  scaleNote:
    'Algorithm 1 of the paper (aggregate, concatenate, linear map, ReLU, L2 normalisation) with K = 2 and the mean and max-pooling aggregators; here the second layer produces the class logits directly, without ReLU, normalisation or a separate classifier. Trained supervised on a few labelled nodes of a small graph (not the paper’s inductive Reddit, citation and PPI benchmarks), with 32 instead of 256 hidden units, on the full graph each step instead of minibatches of target nodes, and with the full neighbourhood by default instead of sampled neighbourhoods of 25 and 10.',
  references: [
    {
      authors: 'W. L. Hamilton, R. Ying, J. Leskovec',
      title: 'Inductive representation learning on large graphs',
      year: 2017,
      url: 'https://arxiv.org/abs/1706.02216',
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
    'Neighbour sampling: with “sampled neighbours” = 0 (default) the aggregator reads the full neighbourhood. With k > 0, during training only, k neighbours per node are drawn uniformly with replacement from the seeded graph generator; evaluation and inspection always use the full neighbourhood, whereas the paper also samples at test time.',
    'The LSTM aggregator and the unsupervised random-walk loss are not implemented.',
    'Max pooling routes the gradient to the first maximal neighbour, as torch.max does.',
    'With one-hot identity features (karate club) the separate self weights give every node its own free parameters; with two labelled members per faction GraphSAGE generalises worse than GCN here (held-out accuracy about 0.5–0.75 across seeds after 200 steps, against about 0.93–0.97 for GCN). On the block model, whose features are shared, it reaches about 0.8–0.9.',
  ],
  guide: {
    what: 'A node-labelling network in which each node keeps its own vector separate from its neighbourhood. Per layer, a node averages (or max-pools) its neighbours’ vectors, concatenates the result with its own, applies a shared linear map and a ReLU, and rescales to unit length. Two such layers (the second without ReLU or rescaling) give class scores for every node. The weights are learned from four labelled nodes. With the karate club’s one-hot identity features every node effectively gets private weights, which is why it generalises worse than GCN here.',
    watch:
      'The training loss on the four labelled nodes drops below 0.1 within 25 steps, yet “Predicted class” keeps disagreeing with “True class” on many unlabelled members: the held-out loss rises while held-out accuracy ends between about 0.5 and 0.75 depending on the seed. On the block model, whose nodes share informative features, the two pictures agree far better.',
    try: [
      {
        label: 'Block model',
        note: 'Nodes share informative features instead of one-hot identities: held-out accuracy rises to about 0.8–0.9.',
        patch: { graph: 'sbm' },
        train: 200,
      },
      {
        label: 'Max-pooling aggregator',
        note: 'Each neighbour passes through a ReLU layer before an elementwise max; on the block model it does slightly worse than the mean (about 0.76–0.82).',
        patch: { graph: 'sbm', aggregator: 'pool' },
        train: 200,
      },
      {
        label: 'Sample 3 neighbours',
        note: 'Each training step aggregates 3 random neighbours per node (with replacement); evaluation uses all of them. Accuracy stays about 0.45–0.75.',
        patch: { sample: 3 },
        train: 200,
      },
    ],
  },
  configSchema: [
    ...graphDataFields,
    {
      key: 'layers',
      label: 'Layers K (incl. output)',
      kind: 'int',
      min: 2,
      max: 3,
      affects: 'architecture',
    },
    { key: 'hidden', label: 'Hidden units', kind: 'int', min: 2, max: 32, affects: 'architecture' },
    {
      key: 'embed2d',
      label: '2-unit last hidden layer',
      kind: 'bool',
      affects: 'architecture',
      help: 'When on, the embedding scatter shows the complete last hidden layer (on the unit circle when normalised); when off, its first two units.',
    },
    {
      key: 'aggregator',
      label: 'Aggregator',
      kind: 'select',
      affects: 'architecture',
      options: [
        { value: 'mean', label: 'Mean' },
        { value: 'pool', label: 'Max-pooling' },
      ],
    },
    { key: 'normalize', label: 'L2-normalise hidden layers', kind: 'bool', affects: 'forward' },
    {
      key: 'sample',
      label: 'Sampled neighbours (0 = all)',
      kind: 'int',
      min: 0,
      max: 10,
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
    hidden: 32,
    embed2d: false,
    aggregator: 'mean',
    normalize: true,
    sample: 0,
    dropout: 0,
  },
  init(c, rng) {
    const b = new ParamBuilder(rng);
    let nIn = featureDim(c.graph);
    sageWidths(c).forEach((w, l) => {
      if (c.aggregator === 'pool')
        declareLinear(b, `sage${l + 1}.pool`, nIn, nIn, { init: 'xavier' });
      declareLinear(b, `sage${l + 1}`, 2 * nIn, w, { init: 'xavier' });
      nIn = w;
    });
    return b.store;
  },
  datasets: [graphDataset<SageConfig>()],
  forward(batch, c) {
    const x = input('x', batch.inputs.x);
    const adj = batch.inputs.adj;
    const widths = sageWidths(c);
    const wT = neighbourWeights(adj, c.sample);
    const wNode = scope('neighbourhood', () => constant(wT, 'neighbourWeights'));
    let h = x;
    let embedding: Var = x;
    widths.forEach((_, l) => {
      const last = l === widths.length - 1;
      const name = `sage${l + 1}`;
      h = scope(name, () => {
        const hin = dropout(h, c.dropout);
        const agg =
          c.aggregator === 'mean'
            ? meanAggregate(hin, wNode)
            : poolAggregate(hin, wT, `${name}.pool`);
        const z = dense(concat([hin, agg], 1), name);
        if (last) return z;
        const a = relu(z);
        return c.normalize ? l2NormalizeRows(a) : a;
      });
      if (!last) embedding = h;
    });
    const emb =
      embedding.shape[1] === 2
        ? embedding
        : scope(`sage${widths.length - 1}`, () => slice(embedding, 1, 0, 2));
    return nodeClassification(batch, h, { embedding: emb, adjNorm: wNode });
  },
  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 0.01, weightDecay: 5e-4 },
    batchSize: 1,
    steps: 200,
    validateEvery: 10,
  },
  views: graphViews(
    'Last hidden layer: first two units (all of it with the 2-unit option)',
    'Aggregation weights S (mean: D⁻¹A; the pooling aggregator uses only its support)',
  ),
  math: {
    neighbourhood: {
      tex: 'S_{vu} = \\frac{1}{|\\mathcal N(v)|}\\,[u \\in \\mathcal N(v)]',
      text: 'The neighbourhood each node aggregates over. With sampling, S_vu counts how often u was drawn for v, divided by the number of draws.',
    },
    sage: {
      tex: 'h_v^{k} = \\frac{\\sigma(W^k [h_v^{k-1} \\Vert \\mathrm{AGG}_v])}{\\lVert \\sigma(W^k [h_v^{k-1} \\Vert \\mathrm{AGG}_v]) \\rVert_2},\\quad \\mathrm{AGG}^{\\text{mean}}_v = \\sum_u S_{vu} h_u^{k-1},\\quad \\mathrm{AGG}^{\\text{pool}}_v = \\max_{u \\in \\mathcal N(v)} \\sigma(W_{\\text{pool}} h_u^{k-1} + b)',
      text: 'Unlike GCN, the node’s own representation is concatenated rather than averaged with the neighbours’, so W can weigh self and neighbourhood separately. The output layer has no σ or normalisation.',
    },
    loss: {
      tex: 'L = -\\frac{1}{|\\mathcal Y_L|} \\sum_{v \\in \\mathcal Y_L} \\log \\operatorname{softmax}(z_v)_{y_v}',
    },
  },
});

export default graphsage;
