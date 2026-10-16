import { Tensor } from '../../engine';
import type { Batch, Config, ForwardContext, ForwardResult, Procedure, ViewSpec } from '../types';
import { PLANAR_EXTENT, planarGrid, planarPoints, type PlanarKind } from '../data/planar';
import { constant } from '../util';

/**
 * Decision field: the model's score evaluated on a 64×64 grid, together with
 * the training points. For two-class softmax models the score is z₁ − z₀.
 */
export function fieldProcedure<C extends Config>(
  forward: (b: Batch, c: C, ctx: ForwardContext) => ForwardResult,
  kindOf: (c: C) => PlanarKind,
  score: (r: ForwardResult) => Tensor,
  noiseOf: (c: C) => number = () => 0.08,
): Procedure<C> {
  return {
    id: 'field',
    label: 'Decision field',
    description:
      'Evaluates the network on a 64 × 64 grid of inputs (4,096 forward passes in one batch) and overlays the training points.',
    run({ config, rng }) {
      const res = 64;
      const grid = planarGrid(res);
      const r = forward({ inputs: { x: grid } }, config, { training: false, rng });
      const s = score(r);
      const pts = planarPoints(kindOf(config), 400, 11, noiseOf(config));
      return {
        tensors: {
          field: new Tensor([res, res], Float32Array.from(s.data)),
          points: new Tensor([400, 2], pts.x),
          labels: new Tensor([400], Float32Array.from(pts.y)),
        },
      };
    },
  };
}

/** z₁ − z₀ from two-class logits [N, 2]. */
export function logitMargin(logits: Tensor): Tensor {
  const N = logits.shape[0];
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) out[i] = logits.data[i * 2 + 1] - logits.data[i * 2];
  return new Tensor([N], out);
}

export const fieldViews: ViewSpec[] = [
  {
    kind: 'scatter',
    title: 'Decision field and training points',
    source: 'proc:field:points',
    colorBy: 'proc:field:labels',
    field: 'proc:field:field',
    extent: PLANAR_EXTENT,
  },
];

/** Labels of a batch as a constant node (so views and losses can use them). */
export function labelTensor(batch: Batch): Tensor {
  return new Tensor([batch.labels!.length], Float32Array.from(batch.labels!));
}

/** ±1 targets t = 2y − 1 as an [N, 1] constant. */
export function signTargets(batch: Batch) {
  const n = batch.labels!.length;
  return constant(
    new Tensor(
      [n, 1],
      Float32Array.from(batch.labels!, (y) => 2 * y - 1),
    ),
    'targets',
  );
}
