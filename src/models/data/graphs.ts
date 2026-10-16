import { Rng, Tensor } from '../../engine';
import type { Batch, Config, Dataset } from '../types';

/**
 * Zachary's karate club (W. W. Zachary, "An information flow model for
 * conflict and fission in small groups", Journal of Anthropological Research
 * 33(4), 1977). 34 members, 78 undirected friendship ties, 0-indexed as in
 * NetworkX's `karate_club_graph`. Published scientific data; the edge list is
 * reproduced from the paper's adjacency matrix.
 */
export const KARATE_EDGES: [number, number][] = [
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 4],
  [0, 5],
  [0, 6],
  [0, 7],
  [0, 8],
  [0, 10],
  [0, 11],
  [0, 12],
  [0, 13],
  [0, 17],
  [0, 19],
  [0, 21],
  [0, 31],
  [1, 2],
  [1, 3],
  [1, 7],
  [1, 13],
  [1, 17],
  [1, 19],
  [1, 21],
  [1, 30],
  [2, 3],
  [2, 7],
  [2, 8],
  [2, 9],
  [2, 13],
  [2, 27],
  [2, 28],
  [2, 32],
  [3, 7],
  [3, 12],
  [3, 13],
  [4, 6],
  [4, 10],
  [5, 6],
  [5, 10],
  [5, 16],
  [6, 16],
  [8, 30],
  [8, 32],
  [8, 33],
  [9, 33],
  [13, 33],
  [14, 32],
  [14, 33],
  [15, 32],
  [15, 33],
  [18, 32],
  [18, 33],
  [19, 33],
  [20, 32],
  [20, 33],
  [22, 32],
  [22, 33],
  [23, 25],
  [23, 27],
  [23, 29],
  [23, 32],
  [23, 33],
  [24, 25],
  [24, 27],
  [24, 31],
  [25, 31],
  [26, 29],
  [26, 33],
  [27, 33],
  [28, 31],
  [28, 33],
  [29, 32],
  [29, 33],
  [30, 32],
  [30, 33],
  [31, 32],
  [31, 33],
  [32, 33],
];

/** Members who sided with the instructor ("Mr. Hi") after the split; all others sided with the officer. */
export const KARATE_MR_HI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21];

export type GraphKind = 'karate' | 'sbm';

export const GRAPH_OPTIONS: { value: GraphKind; label: string }[] = [
  { value: 'karate', label: 'Karate club (Zachary 1977)' },
  { value: 'sbm', label: 'Stochastic block model (synthetic)' },
];

export interface GraphData {
  kind: GraphKind;
  name: string;
  n: number;
  edges: [number, number][];
  /** Row-major [n, featDim] node features. */
  features: Float32Array;
  featDim: number;
  labels: number[];
  classNames: string[];
  /** Labelled training nodes, in order of preference within each class. */
  labelOrder: number[][];
}

/** Feature dimension of each graph (determines first-layer parameter shapes). */
export function featureDim(kind: GraphKind): number {
  return kind === 'karate' ? 34 : SBM.featDim;
}

export function numClasses(kind: GraphKind): number {
  return kind === 'karate' ? 2 : SBM.blocks;
}

export function karateGraph(): GraphData {
  const n = 34;
  const features = new Float32Array(n * n);
  for (let i = 0; i < n; i++) features[i * n + i] = 1;
  const hi = new Set(KARATE_MR_HI);
  const labels = Array.from({ length: n }, (_, i) => (hi.has(i) ? 0 : 1));
  // The instructor (node 0) and the officer (node 33) first, then a fixed shuffle.
  const order = new Rng(1977).shuffle(Array.from({ length: n }, (_, i) => i));
  const labelOrder = [0, 1].map((c) => {
    const lead = c === 0 ? 0 : 33;
    return [lead, ...order.filter((i) => labels[i] === c && i !== lead)];
  });
  return {
    kind: 'karate',
    name: 'Zachary’s karate club',
    n,
    edges: KARATE_EDGES,
    features,
    featDim: n,
    labels,
    classNames: ['Mr. Hi', 'Officer'],
    labelOrder,
  };
}

export const SBM = { blocks: 3, perBlock: 20, pIn: 0.25, pOut: 0.025, featDim: 8, seed: 2024 };

/**
 * Planted-partition stochastic block model: 3 blocks of 20 nodes, edges
 * inside a block with probability 0.25 and across blocks with 0.025. Each
 * node's 8 features are its block's prototype (a fixed random unit vector,
 * scaled by 1) plus N(0, σ²) noise, so features alone are weakly informative.
 * Isolated nodes are joined to a random member of their own block.
 */
export function sbmGraph(noise: number): GraphData {
  const { blocks, perBlock, pIn, pOut, featDim: F, seed } = SBM;
  const n = blocks * perBlock;
  const rng = new Rng(seed);
  const labels = Array.from({ length: n }, (_, i) => Math.floor(i / perBlock));
  const edges: [number, number][] = [];
  const deg = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      if (rng.uniform() < (labels[i] === labels[j] ? pIn : pOut)) {
        edges.push([i, j]);
        deg[i]++;
        deg[j]++;
      }
    }
  for (let i = 0; i < n; i++) {
    if (deg[i] > 0) continue;
    const c = labels[i];
    let j = i;
    while (j === i) j = c * perBlock + rng.int(perBlock);
    edges.push(i < j ? [i, j] : [j, i]);
    deg[i]++;
    deg[j]++;
  }
  const protoRng = new Rng(seed + 1);
  const protos = Array.from({ length: blocks }, () => {
    const v = Array.from({ length: F }, () => protoRng.normal());
    const s = Math.hypot(...v);
    return v.map((x) => x / s);
  });
  const featRng = new Rng(seed + 2);
  const features = new Float32Array(n * F);
  for (let i = 0; i < n; i++)
    for (let f = 0; f < F; f++)
      features[i * F + f] = protos[labels[i]][f] + noise * featRng.normal();
  const order = new Rng(seed + 3).shuffle(Array.from({ length: n }, (_, i) => i));
  const labelOrder = Array.from({ length: blocks }, (_, c) => order.filter((i) => labels[i] === c));
  return {
    kind: 'sbm',
    name: 'Stochastic block model',
    n,
    edges,
    features,
    featDim: F,
    labels,
    classNames: Array.from({ length: blocks }, (_, c) => `block ${String.fromCharCode(65 + c)}`),
    labelOrder,
  };
}

const graphCache = new Map<string, GraphData>();

export function loadGraph(kind: GraphKind, noise = 1): GraphData {
  const key = kind === 'karate' ? 'karate' : `sbm:${noise}`;
  let g = graphCache.get(key);
  if (!g) {
    g = kind === 'karate' ? karateGraph() : sbmGraph(noise);
    graphCache.set(key, g);
  }
  return g;
}

/** Dense symmetric 0/1 adjacency without self-loops. */
export function adjacencyTensor(n: number, edges: [number, number][]): Tensor {
  const a = new Float32Array(n * n);
  for (const [i, j] of edges) {
    a[i * n + j] = 1;
    a[j * n + i] = 1;
  }
  return new Tensor([n, n], a);
}

export function labelledNodes(g: GraphData, perClass: number): number[] {
  return g.labelOrder.flatMap((list) => list.slice(0, perClass)).sort((a, b) => a - b);
}

/**
 * Full-graph batch for transductive node classification. Inputs: `x` [N, F]
 * node features, `adj` [N, N] adjacency, `mask` [N] (1 = node enters the
 * loss), `labelled` [N] (1 = node's label is given for training). `labels`
 * holds every node's class; the loss only reads nodes with mask = 1.
 */
export function graphBatch(
  g: GraphData,
  perClass: number,
  split: 'train' | 'heldout',
  caption: string,
): Batch {
  const train = new Set(labelledNodes(g, perClass));
  const labelled = new Float32Array(g.n);
  const mask = new Float32Array(g.n);
  for (let i = 0; i < g.n; i++) {
    labelled[i] = train.has(i) ? 1 : 0;
    mask[i] = split === 'train' ? labelled[i] : 1 - labelled[i];
  }
  return {
    inputs: {
      x: new Tensor([g.n, g.featDim], g.features),
      adj: adjacencyTensor(g.n, g.edges),
      mask: new Tensor([g.n], mask),
      labelled: new Tensor([g.n], labelled),
    },
    labels: g.labels,
    meta: {
      edges: g.edges,
      classNames: g.classNames,
      caption,
      labelledNodes: [...train].sort((a, b) => a - b),
    },
  };
}

export interface GraphDataConfig extends Config {
  graph: GraphKind;
  labelsPerClass: number;
  featureNoise: number;
}

export function graphDataset<C extends GraphDataConfig>(): Dataset<C> {
  return {
    id: 'graphs',
    name: 'Node classification graphs',
    description:
      'Zachary’s karate club (34 members, 78 ties, two factions after the split; identity features) or a 60-node, three-block stochastic block model with noisy 8-dimensional features. One graph, transductive: a few labelled nodes per class enter the loss; the rest are held out.',
    source: 'embedded',
    license:
      'Karate club: published data from Zachary (1977), J. Anthropological Research 33(4):452–473. Block model: generated in the browser.',
    train(_rng, _batchSize, c) {
      const g = loadGraph(c.graph, c.featureNoise);
      return graphBatch(
        g,
        c.labelsPerClass,
        'train',
        `${g.name}: loss on ${c.labelsPerClass} labelled node(s) per class`,
      );
    },
    validation(c) {
      const g = loadGraph(c.graph, c.featureNoise);
      return graphBatch(g, c.labelsPerClass, 'heldout', `${g.name}: held-out (unlabelled) nodes`);
    },
    exampleCount: 1,
    example(_index, c) {
      const g = loadGraph(c.graph, c.featureNoise);
      return graphBatch(
        g,
        c.labelsPerClass,
        'train',
        `${g.name}, ${g.n} nodes, ${g.edges.length} edges`,
      );
    },
    exampleLabel(_index, c) {
      return c.graph === 'karate' ? 'Karate club' : 'Block model';
    },
  };
}
