import { Tensor, alloc, op, sigmoidFn, shapeEq, fmtShape, type Var } from '../../engine';

/**
 * Neumaier (compensated) summation. Reconstruction losses add up thousands of
 * per-pixel terms; plain left-to-right summation loses enough precision that
 * float64 central differences at ε = 1e-6 can no longer resolve small gradients.
 */
function compensatedSum(n: number, term: (i: number) => number): number {
  let s = 0;
  let c = 0;
  for (let i = 0; i < n; i++) {
    const x = term(i);
    const t = s + x;
    c += Math.abs(s) >= Math.abs(x) ? s - t + x : x - t + s;
    s = t;
  }
  return s + c;
}

/**
 * scale · Σᵢ [max(zᵢ, 0) − zᵢ tᵢ + log(1 + e^{−|zᵢ|})], the Bernoulli negative
 * log-likelihood of targets t ∈ [0, 1] under logits z (binary cross-entropy).
 * ∂/∂zᵢ = scale · (σ(zᵢ) − tᵢ).
 */
export function bernoulliNll(logits: Var, targets: Tensor, scale: number): Var {
  const zd = logits.value.data;
  const n = zd.length;
  if (targets.size !== n) throw new Error(`bernoulliNll: ${n} logits, ${targets.size} targets`);
  const td = targets.data;
  const s = compensatedSum(n, (i) => {
    const z = zd[i];
    return Math.max(z, 0) - z * td[i] + Math.log1p(Math.exp(-Math.abs(z)));
  });
  return op(
    'bernoulliNll',
    new Tensor([], alloc(1, zd).fill(scale * s)),
    [logits],
    (g) => {
      const k = g.data[0] * scale;
      const gz = alloc(n, zd);
      for (let i = 0; i < n; i++) gz[i] = k * (sigmoidFn(zd[i]) - td[i]);
      return [new Tensor(logits.shape, gz)];
    },
    { scale },
  );
}

/** scale · Σᵢ (pᵢ − tᵢ)². ∂/∂pᵢ = 2 · scale · (pᵢ − tᵢ). */
export function squaredError(pred: Var, targets: Tensor, scale: number): Var {
  const pd = pred.value.data;
  if (!shapeEq(pred.shape, targets.shape) && pd.length !== targets.size) {
    throw new Error(`squaredError: ${fmtShape(pred.shape)} vs ${fmtShape(targets.shape)}`);
  }
  const n = pd.length;
  const td = targets.data;
  const s = compensatedSum(n, (i) => (pd[i] - td[i]) ** 2);
  return op(
    'squaredError',
    new Tensor([], alloc(1, pd).fill(scale * s)),
    [pred],
    (g) => {
      const k = 2 * g.data[0] * scale;
      const gp = alloc(n, pd);
      for (let i = 0; i < n; i++) gp[i] = k * (pd[i] - td[i]);
      return [new Tensor(pred.shape, gp)];
    },
    { scale },
  );
}
