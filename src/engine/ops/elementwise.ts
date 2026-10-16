import { Tensor, alloc, shapeEq, sizeOf, stridesOf, fmtShape } from '../tensor';
import { op, type Var, type Attrs } from '../graph';

export function broadcastShape(a: readonly number[], b: readonly number[]): number[] {
  const n = Math.max(a.length, b.length);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const da = a[a.length - n + i] ?? 1;
    const db = b[b.length - n + i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(`cannot broadcast ${fmtShape(a)} with ${fmtShape(b)}`);
    }
    out[i] = da === 1 ? db : da;
  }
  return out;
}

/** For each element of `outShape`, the flat offset into a tensor of `srcShape` broadcast to it. */
export function broadcastMap(outShape: readonly number[], srcShape: readonly number[]): Int32Array {
  const n = sizeOf(outShape);
  const map = new Int32Array(n);
  const rank = outShape.length;
  const srcStrides = stridesOf(srcShape);
  const eff = new Array<number>(rank).fill(0);
  for (let i = 0; i < rank; i++) {
    const j = srcShape.length - rank + i;
    if (j >= 0 && srcShape[j] !== 1) eff[i] = srcStrides[j];
  }
  const idx = new Array<number>(rank).fill(0);
  let off = 0;
  for (let k = 0; k < n; k++) {
    map[k] = off;
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++;
      off += eff[d];
      if (idx[d] < outShape[d]) break;
      off -= eff[d] * idx[d];
      idx[d] = 0;
    }
  }
  return map;
}

/** Sum a gradient of a broadcast result back to the operand's shape. */
export function sumToShape(g: Tensor, shape: readonly number[]): Tensor {
  if (shapeEq(g.shape, shape)) return g;
  const out = Tensor.zeros(shape, g.data);
  const map = broadcastMap(g.shape, shape);
  const o = out.data;
  const s = g.data;
  for (let k = 0; k < s.length; k++) o[map[k]] += s[k];
  return out;
}

type BinFn = (x: number, y: number) => number;

function binaryForward(a: Tensor, b: Tensor, f: BinFn): Tensor {
  const ad = a.data;
  const bd = b.data;
  if (shapeEq(a.shape, b.shape)) {
    const out = alloc(ad.length, ad);
    for (let i = 0; i < ad.length; i++) out[i] = f(ad[i], bd[i]);
    return new Tensor(a.shape, out);
  }
  const shape = broadcastShape(a.shape, b.shape);
  const n = sizeOf(shape);
  const out = alloc(n, ad);
  if (b.size === 1) {
    const y = bd[0];
    if (a.size === n) {
      for (let i = 0; i < n; i++) out[i] = f(ad[i], y);
      return new Tensor(shape, out);
    }
  }
  if (a.size === 1 && b.size === n) {
    const x = ad[0];
    for (let i = 0; i < n; i++) out[i] = f(x, bd[i]);
    return new Tensor(shape, out);
  }
  const ma = broadcastMap(shape, a.shape);
  const mb = broadcastMap(shape, b.shape);
  for (let i = 0; i < n; i++) out[i] = f(ad[ma[i]], bd[mb[i]]);
  return new Tensor(shape, out);
}

export function add(a: Var, b: Var): Var {
  const y = binaryForward(a.value, b.value, (x, z) => x + z);
  return op('add', y, [a, b], (g) => [sumToShape(g, a.shape), sumToShape(g, b.shape)]);
}

export function sub(a: Var, b: Var): Var {
  const y = binaryForward(a.value, b.value, (x, z) => x - z);
  return op('sub', y, [a, b], (g) => [sumToShape(g, a.shape), sumToShape(scaleT(g, -1), b.shape)]);
}

export function mul(a: Var, b: Var): Var {
  const y = binaryForward(a.value, b.value, (x, z) => x * z);
  return op('mul', y, [a, b], (g) => [
    sumToShape(
      binaryForward(g, b.value, (u, v) => u * v),
      a.shape,
    ),
    sumToShape(
      binaryForward(g, a.value, (u, v) => u * v),
      b.shape,
    ),
  ]);
}

export function div(a: Var, b: Var): Var {
  const y = binaryForward(a.value, b.value, (x, z) => x / z);
  return op('div', y, [a, b], (g) => {
    const ga = binaryForward(g, b.value, (u, v) => u / v);
    // ∂(a/b)/∂b = −a/b² = −y/b
    const yb = binaryForward(y, b.value, (u, v) => u / v);
    const gb = binaryForward(g, yb, (u, v) => -u * v);
    return [sumToShape(ga, a.shape), sumToShape(gb, b.shape)];
  });
}

export function maximum(a: Var, b: Var): Var {
  const y = binaryForward(a.value, b.value, (x, z) => (x >= z ? x : z));
  return op('maximum', y, [a, b], (g) => {
    const ma = binaryForward(a.value, b.value, (x, z) => (x >= z ? 1 : 0));
    const ga = binaryForward(g, ma, (u, m) => u * m);
    const gb = binaryForward(g, ma, (u, m) => u * (1 - m));
    return [sumToShape(ga, a.shape), sumToShape(gb, b.shape)];
  });
}

export function scaleT(t: Tensor, s: number): Tensor {
  const out = alloc(t.size, t.data);
  for (let i = 0; i < t.size; i++) out[i] = t.data[i] * s;
  return new Tensor(t.shape, out);
}

export function mulScalar(a: Var, s: number): Var {
  return op('scale', scaleT(a.value, s), [a], (g) => [scaleT(g, s)], { factor: s });
}

export function addScalar(a: Var, s: number): Var {
  const out = alloc(a.value.size, a.value.data);
  for (let i = 0; i < out.length; i++) out[i] = a.value.data[i] + s;
  return op('addScalar', new Tensor(a.shape, out), [a], (g) => [g], { value: s });
}

/** a·x + b·y with scalar coefficients (common in diffusion and integrators). */
export function axpby(alpha: number, x: Var, beta: number, y: Var): Var {
  if (!shapeEq(x.shape, y.shape)) throw new Error('axpby needs equal shapes');
  const out = alloc(x.value.size, x.value.data);
  for (let i = 0; i < out.length; i++) out[i] = alpha * x.value.data[i] + beta * y.value.data[i];
  return op('axpby', new Tensor(x.shape, out), [x, y], (g) => [scaleT(g, alpha), scaleT(g, beta)], {
    alpha,
    beta,
  });
}

/**
 * Elementwise op given f(x) and f'(x) expressed through x and y = f(x).
 */
function unary(
  name: string,
  x: Var,
  f: (v: number) => number,
  df: (v: number, y: number) => number,
  attrs: Attrs = {},
): Var {
  const xd = x.value.data;
  const out = alloc(xd.length, xd);
  for (let i = 0; i < xd.length; i++) out[i] = f(xd[i]);
  const y = new Tensor(x.shape, out);
  return op(
    name,
    y,
    [x],
    (g) => {
      const gd = alloc(g.size, g.data);
      for (let i = 0; i < gd.length; i++) gd[i] = g.data[i] * df(xd[i], out[i]);
      return [new Tensor(x.shape, gd)];
    },
    attrs,
  );
}

export const neg = (x: Var) =>
  unary(
    'neg',
    x,
    (v) => -v,
    () => -1,
  );
export const exp = (x: Var) => unary('exp', x, Math.exp, (_v, y) => y);
export const log = (x: Var) => unary('log', x, Math.log, (v) => 1 / v);
export const sqrt = (x: Var) => unary('sqrt', x, Math.sqrt, (_v, y) => 0.5 / y);
export const square = (x: Var) =>
  unary(
    'square',
    x,
    (v) => v * v,
    (v) => 2 * v,
  );
export const abs = (x: Var) => unary('abs', x, Math.abs, (v) => (v > 0 ? 1 : v < 0 ? -1 : 0));
export const powScalar = (x: Var, p: number) =>
  unary(
    'pow',
    x,
    (v) => Math.pow(v, p),
    (v) => p * Math.pow(v, p - 1),
    { p },
  );
export const sin = (x: Var) => unary('sin', x, Math.sin, (v) => Math.cos(v));
export const cos = (x: Var) => unary('cos', x, Math.cos, (v) => -Math.sin(v));

export const tanh = (x: Var) => unary('tanh', x, Math.tanh, (_v, y) => 1 - y * y);
export const sigmoid = (x: Var) => unary('sigmoid', x, sigmoidFn, (_v, y) => y * (1 - y));
export const relu = (x: Var) =>
  unary(
    'relu',
    x,
    (v) => (v > 0 ? v : 0),
    (v) => (v > 0 ? 1 : 0),
  );
export const leakyRelu = (x: Var, alpha = 0.01) =>
  unary(
    'leakyRelu',
    x,
    (v) => (v > 0 ? v : alpha * v),
    (v) => (v > 0 ? 1 : alpha),
    { alpha },
  );
export const elu = (x: Var, alpha = 1) =>
  unary(
    'elu',
    x,
    (v) => (v > 0 ? v : alpha * (Math.exp(v) - 1)),
    (v, y) => (v > 0 ? 1 : y + alpha),
    { alpha },
  );
export const softplus = (x: Var) => unary('softplus', x, softplusFn, (v) => sigmoidFn(v));
export const silu = (x: Var) =>
  unary(
    'silu',
    x,
    (v) => v * sigmoidFn(v),
    (v) => {
      const s = sigmoidFn(v);
      return s * (1 + v * (1 - s));
    },
  );
/** Exact GELU, x·Φ(x), as in PyTorch's default. */
export const gelu = (x: Var) =>
  unary(
    'gelu',
    x,
    (v) => 0.5 * v * (1 + erf(v / Math.SQRT2)),
    (v) => 0.5 * (1 + erf(v / Math.SQRT2)) + (v * Math.exp(-0.5 * v * v)) / Math.sqrt(2 * Math.PI),
  );
/** LeCun's squashing function f(a) = A·tanh(S·a), A = 1.7159, S = 2/3. */
export const scaledTanh = (x: Var) =>
  unary(
    'scaledTanh',
    x,
    (v) => 1.7159 * Math.tanh((2 / 3) * v),
    (_v, y) => (2 / 3) * (1.7159 - (y * y) / 1.7159),
  );
export const identity = (x: Var) => x;

export function sigmoidFn(v: number): number {
  if (v >= 0) {
    const z = Math.exp(-v);
    return 1 / (1 + z);
  }
  const z = Math.exp(v);
  return z / (1 + z);
}

export function softplusFn(v: number): number {
  return v > 20 ? v : v < -20 ? Math.exp(v) : Math.log1p(Math.exp(v));
}

/**
 * Error function to double precision: Maclaurin series for |x| < 2, and the
 * continued fraction for erfc beyond (both converge to < 1e-15 there).
 */
export function erf(x: number): number {
  const z = Math.abs(x);
  if (z < 2) {
    // erf(z) = 2/√π Σ (−1)ⁿ z^(2n+1) / (n! (2n+1))
    let term = z;
    let sum = z;
    const z2 = z * z;
    for (let n = 1; n < 60; n++) {
      term *= -z2 / n;
      const d = term / (2 * n + 1);
      sum += d;
      if (Math.abs(d) < 1e-17 * Math.abs(sum)) break;
    }
    const r = (2 / Math.sqrt(Math.PI)) * sum;
    return x < 0 ? -r : r;
  }
  if (z > 6) return x < 0 ? -1 : 1;
  // erfc(z) = e^(−z²)/√π · 1/(z + (1/2)/(z + 1/(z + (3/2)/(z + …)))), evaluated from the tail.
  let f = z;
  for (let n = 60; n >= 1; n--) f = z + n / 2 / f;
  const erfc = Math.exp(-z * z) / Math.sqrt(Math.PI) / f;
  return x < 0 ? erfc - 1 : 1 - erfc;
}

export type ActivationName =
  'relu' | 'tanh' | 'sigmoid' | 'gelu' | 'silu' | 'leakyRelu' | 'elu' | 'scaledTanh' | 'identity';

export const activations: Record<ActivationName, (x: Var) => Var> = {
  relu,
  tanh,
  sigmoid,
  gelu,
  silu,
  leakyRelu: (x) => leakyRelu(x, 0.01),
  elu: (x) => elu(x, 1),
  scaledTanh,
  identity,
};

export const activationTex: Record<ActivationName, string> = {
  relu: '\\max(0, x)',
  tanh: '\\tanh x',
  sigmoid: '\\sigma(x) = 1/(1+e^{-x})',
  gelu: 'x\\,\\Phi(x)',
  silu: 'x\\,\\sigma(x)',
  leakyRelu: '\\max(0.01x, x)',
  elu: 'x \\text{ if } x>0 \\text{ else } e^x-1',
  scaledTanh: '1.7159\\,\\tanh(\\tfrac{2}{3}x)',
  identity: 'x',
};
