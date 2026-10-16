import { Tensor, alloc, shapeEq, fmtShape } from '../tensor';
import { op, type Var } from '../graph';
import { softmaxT } from './reduce';
import { sigmoidFn } from './elementwise';

/**
 * Mean cross-entropy between logits [N, C] and integer targets:
 * L = −(1/N) Σ_n log softmax(z_n)[y_n]. Gradient: (softmax(z) − onehot(y)) / N.
 */
export function crossEntropy(logits: Var, targets: ArrayLike<number>, labelSmoothing = 0): Var {
  const Z = logits.value;
  const C = Z.shape[Z.rank - 1];
  const N = Z.size / C;
  if (targets.length !== N)
    throw new Error(`crossEntropy: ${targets.length} targets for ${N} rows`);
  const p = softmaxT(Z, -1);
  let loss = 0;
  const eps = labelSmoothing;
  for (let n = 0; n < N; n++) {
    const t = targets[n];
    for (let c = 0; c < C; c++) {
      const q = (c === t ? 1 - eps : 0) + eps / C;
      if (q > 0) loss -= q * Math.log(Math.max(p.data[n * C + c], 1e-300));
    }
  }
  loss /= N;
  const tg = Array.from(targets);
  return op(
    'crossEntropy',
    new Tensor([], alloc(1, Z.data).fill(loss)),
    [logits],
    (g) => {
      const gz = alloc(Z.size, Z.data);
      const s = g.data[0] / N;
      for (let n = 0; n < N; n++)
        for (let c = 0; c < C; c++) {
          const q = (c === tg[n] ? 1 - eps : 0) + eps / C;
          gz[n * C + c] = s * (p.data[n * C + c] - q);
        }
      return [new Tensor(Z.shape, gz)];
    },
    { classes: C, labelSmoothing },
    { probs: p },
  );
}

/** Mean squared error, (1/n) Σ (ŷ − y)². */
export function mse(pred: Var, target: Var | Tensor): Var {
  const T = target instanceof Tensor ? target : target.value;
  if (!shapeEq(pred.shape, T.shape)) {
    throw new Error(`mse shapes ${fmtShape(pred.shape)} vs ${fmtShape(T.shape)}`);
  }
  const n = T.size;
  const pd = pred.value.data;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const d = pd[i] - T.data[i];
    s += d * d;
  }
  const parents = target instanceof Tensor ? [pred] : [pred, target];
  return op('mse', new Tensor([], alloc(1, pd).fill(s / n)), parents, (g) => {
    const k = (2 * g.data[0]) / n;
    const gp = alloc(n, pd);
    for (let i = 0; i < n; i++) gp[i] = k * (pd[i] - T.data[i]);
    const out: (Tensor | null)[] = [new Tensor(pred.shape, gp)];
    if (!(target instanceof Tensor)) {
      const gt = alloc(n, pd);
      for (let i = 0; i < n; i++) gt[i] = -gp[i];
      out.push(new Tensor(T.shape, gt));
    }
    return out;
  });
}

/** Mean binary cross-entropy on logits, numerically stable form. */
export function bceWithLogits(logits: Var, targets: Tensor): Var {
  const zd = logits.value.data;
  const n = zd.length;
  if (targets.size !== n) throw new Error('bceWithLogits: target size mismatch');
  let s = 0;
  for (let i = 0; i < n; i++) {
    const z = zd[i];
    const t = targets.data[i];
    s += Math.max(z, 0) - z * t + Math.log1p(Math.exp(-Math.abs(z)));
  }
  return op('bceWithLogits', new Tensor([], alloc(1, zd).fill(s / n)), [logits], (g) => {
    const gz = alloc(n, zd);
    const k = g.data[0] / n;
    for (let i = 0; i < n; i++) gz[i] = k * (sigmoidFn(zd[i]) - targets.data[i]);
    return [new Tensor(logits.shape, gz)];
  });
}
