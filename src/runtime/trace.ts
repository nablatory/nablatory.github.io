import type { Graph, Var, AttrValue } from '../engine';
import type { BatchMeta } from '../models/types';

export interface TensorPayload {
  shape: number[];
  data: Float32Array;
}

export interface TraceNode {
  id: number;
  key: string;
  op: string;
  scope: string;
  shape: number[];
  parents: number[];
  attrs: Record<string, AttrValue>;
  param?: string;
  /** Output was replaced by a user edit (activation patching). */
  edited: boolean;
  value: Float32Array;
  /** ∂(objective)/∂(this node), when the node lies on a differentiable path. */
  grad?: Float32Array;
  aux?: Record<string, TensorPayload>;
  /** Analytic floating-point operation count of this node's forward computation. */
  flops: number;
}

export interface Trace {
  nodes: TraceNode[];
  /** Forward output name → node id. */
  outputs: Record<string, number>;
  /** Node id of the scalar the gradients were taken of. */
  objective?: { node: number; label: string };
  loss?: number;
  metrics: Record<string, number>;
  timing: { forwardMs: number; backwardMs: number };
  training: boolean;
  /** Display metadata of the input batch (tokens, class names, graph edges). */
  meta?: BatchMeta;
}

const f32 = (d: ArrayLike<number>) =>
  d instanceof Float32Array ? d.slice() : Float32Array.from(d);

/** Analytic FLOP estimate (multiply–add counted as 2). */
export function nodeFlops(v: Var): number {
  const out = v.value.size;
  const ps = v.parents;
  switch (v.op) {
    case 'linear': {
      const nIn = Number(v.attrs.in);
      return 2 * out * nIn + (v.attrs.bias ? out : 0);
    }
    case 'matmul': {
      const a = ps[0]?.value;
      const k = a ? a.shape[a.rank - 1] : 0;
      return 2 * out * k;
    }
    case 'conv2d': {
      const w = ps[1]?.value;
      return w ? 2 * out * (w.shape[1] * w.shape[2] * w.shape[3]) : 0;
    }
    case 'convTranspose2d': {
      const x = ps[0]?.value;
      const w = ps[1]?.value;
      return x && w ? 2 * x.size * w.shape[1] * w.shape[2] * w.shape[3] : 0;
    }
    case 'param':
    case 'input':
    case 'const':
    case 'reshape':
    case 'transpose':
    case 'slice':
    case 'concat':
    case 'pad':
    case 'gather':
      return 0;
    case 'softmax':
    case 'logSoftmax':
    case 'layerNorm':
    case 'batchNorm':
      return 5 * out;
    case 'maxPool2d':
    case 'avgPool2d': {
      const k = Number(v.attrs.kernel);
      return out * k * k;
    }
    default:
      return ps.reduce((s, p) => Math.max(s, p.value.size), out);
  }
}

export function serializeGraph(
  g: Graph,
  outputs: Record<string, Var>,
  extra: Omit<Trace, 'nodes' | 'outputs'>,
): Trace {
  const nodes: TraceNode[] = g.nodes.map((v) => {
    const node: TraceNode = {
      id: v.id,
      key: v.key,
      op: v.op,
      scope: v.scope,
      shape: v.shape,
      parents: v.parentIds,
      attrs: v.attrs,
      param: v.param,
      edited: v.edited,
      value: f32(v.value.data),
      flops: nodeFlops(v),
    };
    if (v.grad) node.grad = f32(v.grad.data);
    if (v.aux) {
      node.aux = {};
      for (const [k, t] of Object.entries(v.aux))
        node.aux[k] = { shape: t.shape, data: f32(t.data) };
    }
    return node;
  });
  const outIds: Record<string, number> = {};
  for (const [k, v] of Object.entries(outputs)) if (v.id >= 0) outIds[k] = v.id;
  return { nodes, outputs: outIds, ...extra };
}

/** Transferable buffers of a trace (for postMessage). */
export function traceBuffers(t: Trace): ArrayBuffer[] {
  const out: ArrayBuffer[] = [];
  for (const n of t.nodes) {
    out.push(n.value.buffer as ArrayBuffer);
    if (n.grad) out.push(n.grad.buffer as ArrayBuffer);
    if (n.aux) for (const a of Object.values(n.aux)) out.push(a.data.buffer as ArrayBuffer);
  }
  return out;
}
