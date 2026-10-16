import {
  Tensor,
  add,
  crossEntropy,
  div,
  mul,
  powScalar,
  reshape,
  scope,
  softmax,
  sqrt,
  square,
  sum,
  addScalar,
  take,
  type Var,
} from '../../engine';
import type { Batch, ConfigField, ForwardResult, ViewSpec } from '../types';
import { GRAPH_OPTIONS, type GraphDataConfig } from '../data/graphs';
import { constant, input } from '../util';

export type AdjNorm = 'symmetric' | 'row';

/** Configuration fields shared by the graph models (dataset choice and labels). */
export const graphDataFields: ConfigField[] = [
  {
    key: 'graph',
    label: 'Graph',
    kind: 'select',
    options: GRAPH_OPTIONS,
    affects: 'architecture',
    help: 'The feature dimension and the number of classes differ between graphs, so switching re-initialises all parameters.',
  },
  {
    key: 'labelsPerClass',
    label: 'Labelled nodes per class',
    kind: 'int',
    min: 1,
    max: 6,
    affects: 'data',
  },
  {
    key: 'featureNoise',
    label: 'Block-model feature noise σ',
    kind: 'float',
    min: 0,
    max: 3,
    step: 0.1,
    affects: 'data',
  },
];

export const graphDataDefaults: GraphDataConfig = {
  graph: 'karate',
  labelsPerClass: 2,
  featureNoise: 1.2,
};

/** Identity matrix as a constant node. */
export function identityConst(n: number): Var {
  const t = Tensor.zeros([n, n]);
  for (let i = 0; i < n; i++) t.data[i * n + i] = 1;
  return constant(t, 'identity');
}

/**
 * Propagation matrix from a 0/1 adjacency A (no gradient flows into it):
 *   Ã = A + I (with self-loops) or A,
 *   symmetric: D̃^{−1/2} Ã D̃^{−1/2},   row: D̃^{−1} Ã,
 * where D̃_ii = Σ_j Ã_ij. Every node must have D̃_ii > 0.
 */
export function normalizeAdjacency(adj: Var, selfLoops: boolean, mode: AdjNorm): Var {
  const n = adj.shape[0];
  const at = selfLoops ? add(adj, identityConst(n)) : adj;
  const deg = sum(at, -1);
  if (mode === 'row') return div(at, reshape(deg, [n, 1]));
  const dinv = powScalar(deg, -0.5);
  return mul(mul(reshape(dinv, [n, 1]), at), reshape(dinv, [1, n]));
}

/** Row-wise L2 normalisation h / ‖h‖₂ (GraphSAGE, Algorithm 1 line 7). */
export function l2NormalizeRows(h: Var, eps = 1e-12): Var {
  const norm = sqrt(addScalar(sum(square(h), -1, true), eps));
  return div(h, norm);
}

/** Indices where a 0/1 tensor is non-zero. */
export function maskIndices(t: Tensor): number[] {
  const out: number[] = [];
  for (let i = 0; i < t.size; i++) if (t.data[i] !== 0) out.push(i);
  return out;
}

function accuracyOn(probs: Tensor, labels: number[], idx: number[]): number {
  const C = probs.shape[1];
  let k = 0;
  for (const i of idx) {
    let b = 0;
    for (let c = 1; c < C; c++) if (probs.data[i * C + c] > probs.data[i * C + b]) b = c;
    if (b === labels[i]) k++;
  }
  return idx.length ? k / idx.length : 0;
}

/**
 * Semi-supervised node-classification head: softmax over classes, cross-entropy
 * on the masked nodes only, accuracies on labelled and unlabelled nodes.
 */
export function nodeClassification(
  batch: Batch,
  logits: Var,
  extra: { embedding: Var; adjNorm?: Var; attention?: Var },
): ForwardResult {
  const probs = softmax(logits, -1);
  const labels = batch.labels ?? [];
  const mask = maskIndices(batch.inputs.mask);
  const labelledIdx = maskIndices(batch.inputs.labelled);
  const unlabelledIdx = labels.map((_, i) => i).filter((i) => batch.inputs.labelled.data[i] === 0);
  let loss: Var | undefined;
  if (labels.length && mask.length) {
    loss = scope('loss', () =>
      crossEntropy(
        take(logits, mask),
        mask.map((i) => labels[i]),
      ),
    );
  }
  const outputs: Record<string, Var> = { logits, probs, embedding: extra.embedding };
  if (labels.length)
    outputs.labels = scope('loss', () =>
      constant(new Tensor([labels.length], Float32Array.from(labels)), 'labels'),
    );
  outputs.labelled = input('labelled', batch.inputs.labelled, false);
  if (extra.adjNorm) outputs.adjNorm = extra.adjNorm;
  if (extra.attention) outputs.attention = extra.attention;
  return {
    outputs,
    loss,
    metrics: labels.length
      ? {
          accuracy: accuracyOn(probs.value, labels, mask),
          labelledAccuracy: accuracyOn(probs.value, labels, labelledIdx),
          unlabelledAccuracy: accuracyOn(probs.value, labels, unlabelledIdx),
        }
      : undefined,
  };
}

export function graphViews(embeddingTitle: string, adjTitle: string): ViewSpec[] {
  return [
    {
      kind: 'graph',
      title: 'Predicted class (arg max of softmax)',
      source: 'out:probs',
      colorBy: 'argmax',
    },
    { kind: 'graph', title: 'True class', source: 'out:probs', colorBy: 'out:labels' },
    {
      kind: 'graph',
      title: 'Labelled training nodes (1) and unlabelled nodes (0)',
      source: 'out:probs',
      colorBy: 'out:labelled',
    },
    { kind: 'scatter', title: embeddingTitle, source: 'out:embedding', colorBy: 'out:labels' },
    { kind: 'matrix', title: adjTitle, source: 'out:adjNorm' },
  ];
}

export const graphLimitations = [
  'Transductive setting on one small graph: the whole graph is processed in every step (full-batch training), and the loss reads only the few labelled nodes.',
  'Dense N × N adjacency and propagation matrices; this is exact for these graph sizes but scales as O(N²) memory, unlike the sparse implementations used for large graphs.',
  'The karate-club labels are the factions after the club split, as recorded by Zachary (1977); features are one-hot node identities, so the model sees structure only.',
];

export { input };
