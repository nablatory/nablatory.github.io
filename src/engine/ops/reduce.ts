import { Tensor, alloc } from '../tensor';
import { op, type Var } from '../graph';
import { broadcastMap } from './elementwise';

function normAxes(axes: number | number[] | undefined, rank: number): number[] {
  if (axes === undefined) return [...Array(rank).keys()];
  const list = Array.isArray(axes) ? axes : [axes];
  return list.map((a) => (a < 0 ? a + rank : a)).sort((x, y) => x - y);
}

function keptShape(shape: number[], axes: number[]): number[] {
  return shape.map((d, i) => (axes.includes(i) ? 1 : d));
}

function finalShape(shape: number[], axes: number[], keepDims: boolean): number[] {
  return keepDims ? keptShape(shape, axes) : shape.filter((_, i) => !axes.includes(i));
}

export function sum(x: Var, axes?: number | number[], keepDims = false): Var {
  const ax = normAxes(axes, x.value.rank);
  const kept = keptShape(x.shape, ax);
  const map = broadcastMap(x.shape, kept);
  const outSize = kept.reduce((a, b) => a * b, 1);
  const out = alloc(outSize, x.value.data);
  const xd = x.value.data;
  for (let i = 0; i < xd.length; i++) out[map[i]] += xd[i];
  const y = new Tensor(finalShape(x.shape, ax, keepDims), out);
  return op(
    'sum',
    y,
    [x],
    (g) => {
      const gx = alloc(xd.length, xd);
      for (let i = 0; i < gx.length; i++) gx[i] = g.data[map[i]];
      return [new Tensor(x.shape, gx)];
    },
    { axes: ax, keepDims },
  );
}

export function mean(x: Var, axes?: number | number[], keepDims = false): Var {
  const ax = normAxes(axes, x.value.rank);
  const count = ax.reduce((a, i) => a * x.shape[i], 1);
  const kept = keptShape(x.shape, ax);
  const map = broadcastMap(x.shape, kept);
  const outSize = kept.reduce((a, b) => a * b, 1);
  const out = alloc(outSize, x.value.data);
  const xd = x.value.data;
  for (let i = 0; i < xd.length; i++) out[map[i]] += xd[i];
  for (let i = 0; i < outSize; i++) out[i] /= count;
  const y = new Tensor(finalShape(x.shape, ax, keepDims), out);
  return op(
    'mean',
    y,
    [x],
    (g) => {
      const gx = alloc(xd.length, xd);
      for (let i = 0; i < gx.length; i++) gx[i] = g.data[map[i]] / count;
      return [new Tensor(x.shape, gx)];
    },
    { axes: ax, keepDims },
  );
}

/** Maximum along one axis; the gradient flows to the first maximal element. */
export function max(x: Var, axis: number, keepDims = false): Var {
  const { outer, len, inner, axis: a } = axisView(x.shape, axis);
  const xd = x.value.data;
  const out = alloc(outer * inner, xd);
  const arg = new Int32Array(outer * inner);
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < inner; i++) {
      let best = -Infinity;
      let bi = 0;
      for (let l = 0; l < len; l++) {
        const v = xd[(o * len + l) * inner + i];
        if (v > best) {
          best = v;
          bi = l;
        }
      }
      out[o * inner + i] = best;
      arg[o * inner + i] = bi;
    }
  }
  const shape = keepDims
    ? x.shape.map((d, i) => (i === a ? 1 : d))
    : x.shape.filter((_, i) => i !== a);
  return op(
    'max',
    new Tensor(shape, out),
    [x],
    (g) => {
      const gx = alloc(xd.length, xd);
      for (let o = 0; o < outer; o++)
        for (let i = 0; i < inner; i++) {
          const j = o * inner + i;
          gx[(o * len + arg[j]) * inner + i] = g.data[j];
        }
      return [new Tensor(x.shape, gx)];
    },
    { axis: a, keepDims },
  );
}

export function axisView(shape: number[], axis: number) {
  const a = axis < 0 ? axis + shape.length : axis;
  let outer = 1;
  for (let i = 0; i < a; i++) outer *= shape[i];
  let inner = 1;
  for (let i = a + 1; i < shape.length; i++) inner *= shape[i];
  return { outer, len: shape[a], inner, axis: a };
}

export function softmaxT(t: Tensor, axis = -1): Tensor {
  const { outer, len, inner } = axisView(t.shape, axis);
  const xd = t.data;
  const out = alloc(xd.length, xd);
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < inner; i++) {
      const base = o * len * inner + i;
      let m = -Infinity;
      for (let l = 0; l < len; l++) m = Math.max(m, xd[base + l * inner]);
      if (m === -Infinity) m = 0;
      let s = 0;
      for (let l = 0; l < len; l++) {
        const e = Math.exp(xd[base + l * inner] - m);
        out[base + l * inner] = e;
        s += e;
      }
      for (let l = 0; l < len; l++) out[base + l * inner] = s > 0 ? out[base + l * inner] / s : 0;
    }
  }
  return new Tensor(t.shape, out);
}

/** softmax(x)_i = exp(x_i − max x) / Σ_j exp(x_j − max x) */
export function softmax(x: Var, axis = -1): Var {
  const y = softmaxT(x.value, axis);
  const { outer, len, inner, axis: a } = axisView(x.shape, axis);
  return op(
    'softmax',
    y,
    [x],
    (g) => {
      // ∂L/∂x = y ⊙ (g − Σ g ⊙ y)
      const gx = alloc(y.size, y.data);
      for (let o = 0; o < outer; o++)
        for (let i = 0; i < inner; i++) {
          const base = o * len * inner + i;
          let dot = 0;
          for (let l = 0; l < len; l++) dot += g.data[base + l * inner] * y.data[base + l * inner];
          for (let l = 0; l < len; l++) {
            const k = base + l * inner;
            gx[k] = y.data[k] * (g.data[k] - dot);
          }
        }
      return [new Tensor(x.shape, gx)];
    },
    { axis: a },
  );
}

export function logSoftmax(x: Var, axis = -1): Var {
  const { outer, len, inner, axis: a } = axisView(x.shape, axis);
  const xd = x.value.data;
  const out = alloc(xd.length, xd);
  for (let o = 0; o < outer; o++)
    for (let i = 0; i < inner; i++) {
      const base = o * len * inner + i;
      let m = -Infinity;
      for (let l = 0; l < len; l++) m = Math.max(m, xd[base + l * inner]);
      let s = 0;
      for (let l = 0; l < len; l++) s += Math.exp(xd[base + l * inner] - m);
      const lse = m + Math.log(s);
      for (let l = 0; l < len; l++) out[base + l * inner] = xd[base + l * inner] - lse;
    }
  const y = new Tensor(x.shape, out);
  return op(
    'logSoftmax',
    y,
    [x],
    (g) => {
      // ∂L/∂x = g − softmax(x) Σ g
      const gx = alloc(xd.length, xd);
      for (let o = 0; o < outer; o++)
        for (let i = 0; i < inner; i++) {
          const base = o * len * inner + i;
          let s = 0;
          for (let l = 0; l < len; l++) s += g.data[base + l * inner];
          for (let l = 0; l < len; l++) {
            const k = base + l * inner;
            gx[k] = g.data[k] - Math.exp(out[k]) * s;
          }
        }
      return [new Tensor(x.shape, gx)];
    },
    { axis: a },
  );
}

export function logSumExp(x: Var, axis = -1, keepDims = false): Var {
  const { outer, len, inner, axis: a } = axisView(x.shape, axis);
  const xd = x.value.data;
  const out = alloc(outer * inner, xd);
  for (let o = 0; o < outer; o++)
    for (let i = 0; i < inner; i++) {
      const base = o * len * inner + i;
      let m = -Infinity;
      for (let l = 0; l < len; l++) m = Math.max(m, xd[base + l * inner]);
      let s = 0;
      for (let l = 0; l < len; l++) s += Math.exp(xd[base + l * inner] - m);
      out[o * inner + i] = m + Math.log(s);
    }
  const shape = keepDims
    ? x.shape.map((d, i) => (i === a ? 1 : d))
    : x.shape.filter((_, i) => i !== a);
  return op(
    'logSumExp',
    new Tensor(shape, out),
    [x],
    (g) => {
      const gx = alloc(xd.length, xd);
      for (let o = 0; o < outer; o++)
        for (let i = 0; i < inner; i++) {
          const base = o * len * inner + i;
          const lse = out[o * inner + i];
          const gv = g.data[o * inner + i];
          for (let l = 0; l < len; l++) {
            const k = base + l * inner;
            gx[k] = gv * Math.exp(xd[k] - lse);
          }
        }
      return [new Tensor(x.shape, gx)];
    },
    { axis: a, keepDims },
  );
}
