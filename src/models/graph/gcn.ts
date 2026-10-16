import {
  ParamBuilder,
  defaultOptimizer,
  dense,
  dropout,
  matmul,
  relu,
  scope,
  slice,
  tanh,
  type Var,
} from '../../engine';
import { defineModel } from '../types';
import { featureDim, graphDataset, numClasses, type GraphDataConfig } from '../data/graphs';
import {
  graphDataDefaults,
  graphDataFields,
  graphLimitations,
  graphViews,
  input,
  nodeClassification,
  normalizeAdjacency,
  type AdjNorm,
} from './common';

export type GcnConfig = GraphDataConfig & {
  layers: number;
  hidden: number;
  embed2d: boolean;
  selfLoops: boolean;
  norm: AdjNorm;
  activation: 'relu' | 'tanh';
  dropout: number;
};

/** Output widths of the GCN layers, ending with the class count. */
export function gcnWidths(c: GcnConfig): number[] {
  const hidden = Array.from({ length: c.layers - 1 }, () => c.hidden);
  if (c.embed2d && hidden.length) hidden[hidden.length - 1] = 2;
  return [...hidden, numClasses(c.graph)];
}

/** One graph convolution H' = Â H W (+ b): the feature transform, then propagation. */
export function gcnLayer(adjNorm: Var, h: Var, name: string): Var {
  return matmul(adjNorm, dense(h, name));
}

const gcn = defineModel<GcnConfig>({
  id: 'gcn',
  family: 'graph',
  name: 'Graph convolutional network',
  summary:
    'Each layer mixes every node’s features with its neighbours’ through the symmetrically normalised adjacency matrix, then applies a shared linear map. Trained semi-supervised from a few labelled nodes.',
  signature:
    'H^{(\\ell+1)} = \\sigma\\big(\\tilde D^{-1/2} \\tilde A \\tilde D^{-1/2} H^{(\\ell)} W^{(\\ell)}\\big),\\quad \\tilde A = A + I',
  fidelity: 'reduced',
  scaleNote:
    'The propagation rule, renormalisation trick, Glorot initialisation, masked cross-entropy and Adam at learning rate 0.01 for 200 steps are as published. The graphs are small (34 and 60 nodes) instead of Citeseer, Cora and Pubmed. The default network follows the paper’s karate-club illustration (Appendix A: three graph convolutions with tanh, hidden widths 4 and 2, a softmax layer on top, one labelled node per class) rather than the two-layer 16-unit ReLU model of the citation experiments (selectable); here the three layers have 16 and 2 hidden units with tanh, the third is the class layer, and two nodes per class are labelled. Dropout is 0.2 instead of 0.5, the L2 penalty 5 × 10⁻⁴ is applied to every parameter (biases included) through the optimiser rather than to the first layer’s weights only, and there is no early stopping.',
  references: [
    {
      authors: 'T. N. Kipf, M. Welling',
      title: 'Semi-supervised classification with graph convolutional networks',
      year: 2017,
      url: 'https://arxiv.org/abs/1609.02907',
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
    'The normalised adjacency is recomputed from the adjacency input in every forward pass (it receives no gradient), so self-loops and the normalisation can be switched without re-initialising.',
    'Row normalisation D̃⁻¹Ã is offered for comparison; the paper uses the symmetric form.',
  ],
  guide: {
    what: 'A network that labels the nodes of one graph. Its input is a feature vector per node (here which node it is, as a one-hot vector) plus the edges. Each layer replaces every node’s vector by a weighted average over the node and its neighbours, then applies one linear map shared by all nodes and a nonlinearity. The output is a class probability for every node. Only the linear maps are learned, from the four nodes whose labels are given; the other 30 are scored but never see their labels.',
    watch:
      '“Predicted class” starts random and settles into the two factions within the first 25–50 steps; compare it with “True class” and the four labelled nodes. The training loss is the cross-entropy on those four nodes and falls towards zero; the held-out accuracy reported with the validation metrics levels off near 0.93–0.97 while the held-out loss stops improving after 100 steps.',
    try: [
      {
        label: 'One label per faction',
        note: 'Only Mr. Hi and the officer are labelled; the graph alone still places about 0.95 of the other members correctly.',
        patch: { labelsPerClass: 1 },
        train: 200,
      },
      {
        label: 'No self-loops',
        note: 'Each layer then averages neighbours only, discarding the node’s own vector; held-out accuracy drops to about 0.73.',
        patch: { selfLoops: false },
        train: 200,
      },
      {
        label: 'Block model',
        note: 'Three planted blocks of 20 nodes with noisy 8-dimensional features: held-out accuracy reaches about 0.98.',
        patch: { graph: 'sbm' },
        train: 200,
      },
    ],
  },
  configSchema: [
    ...graphDataFields,
    {
      key: 'layers',
      label: 'GCN layers (incl. output)',
      kind: 'int',
      min: 2,
      max: 4,
      affects: 'architecture',
    },
    { key: 'hidden', label: 'Hidden units', kind: 'int', min: 2, max: 32, affects: 'architecture' },
    {
      key: 'embed2d',
      label: '2-unit last hidden layer',
      kind: 'bool',
      affects: 'architecture',
      help: 'When on, the embedding scatter shows the complete last hidden layer; when off, its first two units.',
    },
    { key: 'selfLoops', label: 'Self-loops (Ã = A + I)', kind: 'bool', affects: 'forward' },
    {
      key: 'norm',
      label: 'Normalisation',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'symmetric', label: 'D̃^−½ Ã D̃^−½ (paper)' },
        { value: 'row', label: 'D̃^−1 Ã (mean)' },
      ],
    },
    {
      key: 'activation',
      label: 'Activation σ',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'relu', label: 'ReLU (paper)' },
        { value: 'tanh', label: 'tanh' },
      ],
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
    layers: 3,
    hidden: 16,
    embed2d: true,
    selfLoops: true,
    norm: 'symmetric',
    activation: 'tanh',
    dropout: 0.2,
  },
  init(c, rng) {
    const b = new ParamBuilder(rng);
    let nIn = featureDim(c.graph);
    gcnWidths(c).forEach((w, l) => {
      // Glorot initialisation, as in the paper's implementation.
      b.add(`gcn${l + 1}.weight`, [w, nIn], { kind: 'xavierUniform', fanIn: nIn, fanOut: w });
      b.add(`gcn${l + 1}.bias`, [w], { kind: 'zeros' });
      nIn = w;
    });
    return b.store;
  },
  datasets: [graphDataset<GcnConfig>()],
  forward(batch, c) {
    const x = input('x', batch.inputs.x);
    const adj = input('adj', batch.inputs.adj, false);
    const adjNorm = scope('normalise', () => normalizeAdjacency(adj, c.selfLoops, c.norm));
    const act = c.activation === 'relu' ? relu : tanh;
    const widths = gcnWidths(c);
    let h = x;
    let embedding: Var = x;
    widths.forEach((_, l) => {
      const last = l === widths.length - 1;
      h = scope(`gcn${l + 1}`, () => {
        const z = gcnLayer(adjNorm, dropout(h, c.dropout), `gcn${l + 1}`);
        return last ? z : act(z);
      });
      if (!last) embedding = h;
    });
    const emb =
      embedding.shape[1] === 2
        ? embedding
        : scope(`gcn${widths.length - 1}`, () => slice(embedding, 1, 0, 2));
    return nodeClassification(batch, h, { embedding: emb, adjNorm });
  },
  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 0.01, weightDecay: 5e-4 },
    batchSize: 1,
    steps: 200,
    validateEvery: 10,
  },
  views: graphViews(
    'Last hidden layer (2 units by default; else its first two)',
    'Propagation matrix Â (D̃^−½ Ã D̃^−½ with the default settings)',
  ),
  math: {
    normalise: {
      tex: '\\hat A = \\tilde D^{-1/2} \\tilde A \\tilde D^{-1/2},\\quad \\tilde A = A + I_N,\\quad \\tilde D_{ii} = \\textstyle\\sum_j \\tilde A_{ij}',
      text: 'The renormalisation trick: adding self-loops keeps each node’s own features, and the symmetric scaling keeps the spectrum of Â in (−1, 1]. Row normalisation D̃⁻¹Ã and dropping the self-loops are offered as options.',
    },
    gcn: {
      tex: 'H^{(\\ell+1)} = \\sigma\\big(\\hat A\\, H^{(\\ell)} W^{(\\ell)} + b^{(\\ell)}\\big)',
      text: 'A shared linear map W transforms every node; Â then replaces each row by a degree-weighted average over the node and its neighbours. The last layer has no σ. The bias b is not in eq. 2 of the paper (its reference layer adds none by default).',
    },
    loss: {
      tex: 'L = -\\frac{1}{|\\mathcal Y_L|} \\sum_{i \\in \\mathcal Y_L} \\log \\operatorname{softmax}(Z_i)_{y_i}',
      text: 'Cross-entropy only on the labelled nodes 𝒴_L; the unlabelled nodes still shape the prediction through Â.',
    },
  },
});

export default gcn;
