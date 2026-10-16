import { activeGraph, type Tensor, type Var } from '../engine';

/** Batch input as a graph leaf. Inputs receive gradients so ∂L/∂x can be inspected. */
export function input(name: string, t: Tensor, requiresGrad = true): Var {
  const g = activeGraph();
  if (!g) throw new Error('input() outside a graph');
  return g.input(name, t, requiresGrad);
}

export function constant(t: Tensor, name = 'const'): Var {
  const g = activeGraph();
  if (!g) throw new Error('constant() outside a graph');
  return g.constant(t, name);
}

/** Index of the maximum of each row of a [N, C] tensor. */
export function argmaxRows(t: Tensor): number[] {
  const C = t.shape[t.rank - 1];
  const N = t.size / C;
  const out: number[] = [];
  for (let n = 0; n < N; n++) {
    let best = -Infinity;
    let bi = 0;
    for (let c = 0; c < C; c++) {
      const v = t.data[n * C + c];
      if (v > best) {
        best = v;
        bi = c;
      }
    }
    out.push(bi);
  }
  return out;
}

export function accuracy(scores: Tensor, labels: number[]): number {
  const pred = argmaxRows(scores);
  let k = 0;
  for (let i = 0; i < pred.length; i++) if (pred[i] === labels[i]) k++;
  return k / pred.length;
}
