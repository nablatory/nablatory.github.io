import { Tensor, alloc, op, type Var } from '../../engine';

const C = Math.sqrt(2 / Math.PI);
const A = 0.044715;

/**
 * GELU in the tanh form used by the original BERT and GPT-2 code:
 * GELU(x) ≈ ½ x (1 + tanh(√(2/π) (x + 0.044715 x³))).
 * Its derivative is computed from the same expression, so forward and
 * backward are mutually consistent to rounding error:
 * d/dx = ½(1 + t) + ½ x (1 − t²) √(2/π) (1 + 3·0.044715 x²), t = tanh(u).
 */
export function geluTanh(x: Var): Var {
  const xd = x.value.data;
  const n = xd.length;
  const out = alloc(n, xd);
  const t = alloc(n, xd);
  for (let i = 0; i < n; i++) {
    const v = xd[i];
    const th = Math.tanh(C * (v + A * v * v * v));
    t[i] = th;
    out[i] = 0.5 * v * (1 + th);
  }
  return op('geluTanh', new Tensor(x.shape, out), [x], (g) => {
    const gx = alloc(n, xd);
    for (let i = 0; i < n; i++) {
      const v = xd[i];
      const th = t[i];
      const d = 0.5 * (1 + th) + 0.5 * v * (1 - th * th) * C * (1 + 3 * A * v * v);
      gx[i] = g.data[i] * d;
    }
    return [new Tensor(x.shape, gx)];
  });
}
