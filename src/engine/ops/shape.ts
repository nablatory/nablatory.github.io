import { Tensor, alloc, resolveShape, stridesOf, sizeOf, fmtShape, shapeEq } from '../tensor';
import { op, type Var } from '../graph';
import { axisView } from './reduce';

export function reshape(x: Var, shape: number[]): Var {
  const s = resolveShape(shape, x.value.size);
  return op('reshape', new Tensor(s, x.value.data), [x], (g) => [new Tensor(x.shape, g.data)], {
    shape: s,
  });
}

export function flatten(x: Var, startAxis = 1): Var {
  const lead = x.shape.slice(0, startAxis);
  return reshape(x, [...lead, -1]);
}

function permuteT(t: Tensor, perm: number[]): Tensor {
  const shape = perm.map((p) => t.shape[p]);
  const inStrides = stridesOf(t.shape);
  const n = t.size;
  const out = alloc(n, t.data);
  const rank = shape.length;
  const st = perm.map((p) => inStrides[p]);
  const idx = new Array<number>(rank).fill(0);
  let off = 0;
  for (let k = 0; k < n; k++) {
    out[k] = t.data[off];
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++;
      off += st[d];
      if (idx[d] < shape[d]) break;
      off -= st[d] * idx[d];
      idx[d] = 0;
    }
  }
  return new Tensor(shape, out);
}

export function transpose(x: Var, perm?: number[]): Var {
  const rank = x.value.rank;
  const p = perm ?? [...Array(rank).keys()].reverse();
  if (p.length !== rank) throw new Error(`transpose perm ${p} for rank ${rank}`);
  const inv = new Array<number>(rank);
  p.forEach((v, i) => (inv[v] = i));
  return op('transpose', permuteT(x.value, p), [x], (g) => [permuteT(g, inv)], { perm: p });
}

/** Swap the last two axes. */
export function T(x: Var): Var {
  const r = x.value.rank;
  const perm = [...Array(r).keys()];
  [perm[r - 1], perm[r - 2]] = [perm[r - 2], perm[r - 1]];
  return transpose(x, perm);
}

export function slice(x: Var, axis: number, start: number, end: number): Var {
  const { outer, len, inner, axis: a } = axisView(x.shape, axis);
  if (start < 0 || end > len || start >= end) throw new Error(`slice [${start}, ${end}) of ${len}`);
  const w = end - start;
  const out = alloc(outer * w * inner, x.value.data);
  const xd = x.value.data;
  for (let o = 0; o < outer; o++) {
    const src = (o * len + start) * inner;
    out.set(xd.subarray(src, src + w * inner), o * w * inner);
  }
  const shape = x.shape.map((d, i) => (i === a ? w : d));
  return op(
    'slice',
    new Tensor(shape, out),
    [x],
    (g) => {
      const gx = alloc(xd.length, xd);
      for (let o = 0; o < outer; o++) {
        gx.set(g.data.subarray(o * w * inner, (o + 1) * w * inner), (o * len + start) * inner);
      }
      return [new Tensor(x.shape, gx)];
    },
    { axis: a, start, end },
  );
}

/** Index one position along an axis and drop that axis. */
export function select(x: Var, axis: number, index: number): Var {
  const a = axis < 0 ? axis + x.value.rank : axis;
  const s = slice(x, a, index, index + 1);
  return reshape(
    s,
    x.shape.filter((_, i) => i !== a),
  );
}

export function concat(xs: Var[], axis: number): Var {
  if (xs.length === 1) return xs[0];
  const a = axis < 0 ? axis + xs[0].value.rank : axis;
  const base = xs[0].shape;
  for (const v of xs) {
    if (v.value.rank !== base.length || v.shape.some((d, i) => i !== a && d !== base[i])) {
      throw new Error(`concat shapes ${xs.map((t) => fmtShape(t.shape)).join(' ')}`);
    }
  }
  const lens = xs.map((v) => v.shape[a]);
  const total = lens.reduce((p, q) => p + q, 0);
  const { outer, inner } = axisView(base, a);
  const out = alloc(outer * total * inner, xs[0].value.data);
  let acc = 0;
  for (let t = 0; t < xs.length; t++) {
    const d = xs[t].value.data;
    const w = lens[t];
    for (let o = 0; o < outer; o++) {
      out.set(d.subarray(o * w * inner, (o + 1) * w * inner), (o * total + acc) * inner);
    }
    acc += w;
  }
  const shape = base.map((d, i) => (i === a ? total : d));
  return op(
    'concat',
    new Tensor(shape, out),
    xs,
    (g) => {
      const grads: Tensor[] = [];
      let off = 0;
      for (let t = 0; t < xs.length; t++) {
        const w = lens[t];
        const gx = alloc(outer * w * inner, g.data);
        for (let o = 0; o < outer; o++) {
          gx.set(
            g.data.subarray((o * total + off) * inner, (o * total + off + w) * inner),
            o * w * inner,
          );
        }
        grads.push(new Tensor(xs[t].shape, gx));
        off += w;
      }
      return grads;
    },
    { axis: a, sizes: lens },
  );
}

export function stack(xs: Var[], axis = 0): Var {
  const a = axis < 0 ? axis + xs[0].value.rank + 1 : axis;
  const expanded = xs.map((v) => reshape(v, [...v.shape.slice(0, a), 1, ...v.shape.slice(a)]));
  return concat(expanded, a);
}

/** Row gather: out[..., :] = table[ids[...], :] (embedding lookup). */
export function take(table: Var, ids: ArrayLike<number>, idsShape: number[] = [ids.length]): Var {
  const [V, D] = table.shape;
  if (table.value.rank !== 2) throw new Error('take expects a [V, D] table');
  const n = ids.length;
  const out = alloc(n * D, table.value.data);
  const td = table.value.data;
  for (let i = 0; i < n; i++) {
    const r = ids[i];
    if (r < 0 || r >= V) throw new Error(`index ${r} out of range [0, ${V})`);
    out.set(td.subarray(r * D, (r + 1) * D), i * D);
  }
  const idsCopy = Array.from(ids);
  return op(
    'gather',
    new Tensor([...idsShape, D], out),
    [table],
    (g) => {
      const gt = alloc(td.length, td);
      for (let i = 0; i < n; i++) {
        const r = idsCopy[i];
        for (let j = 0; j < D; j++) gt[r * D + j] += g.data[i * D + j];
      }
      return [new Tensor(table.shape, gt)];
    },
    { rows: idsCopy.length <= 64 ? idsCopy : null },
  );
}

/** Constant padding: pads[i] = [before, after] for axis i. */
export function pad(x: Var, pads: [number, number][], value = 0): Var {
  const rank = x.value.rank;
  if (pads.length !== rank) throw new Error('pad needs one [before, after] pair per axis');
  const shape = x.shape.map((d, i) => d + pads[i][0] + pads[i][1]);
  const out = alloc(sizeOf(shape), x.value.data);
  if (value !== 0) out.fill(value);
  const os = stridesOf(shape);
  const xd = x.value.data;
  const idx = new Array<number>(rank).fill(0);
  const map = new Int32Array(xd.length);
  for (let k = 0; k < xd.length; k++) {
    let o = 0;
    for (let d = 0; d < rank; d++) o += (idx[d] + pads[d][0]) * os[d];
    map[k] = o;
    out[o] = xd[k];
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++;
      if (idx[d] < x.shape[d]) break;
      idx[d] = 0;
    }
  }
  return op(
    'pad',
    new Tensor(shape, out),
    [x],
    (g) => {
      const gx = alloc(xd.length, xd);
      for (let k = 0; k < gx.length; k++) gx[k] = g.data[map[k]];
      return [new Tensor(x.shape, gx)];
    },
    { pads: pads.flat(), value },
  );
}

/**
 * y = x where mask = 0, `value` where mask ≠ 0. The mask broadcasts over the
 * leading axes of x. Masked positions receive no gradient.
 */
export function maskedFill(x: Var, mask: Tensor, value: number): Var {
  const xd = x.value.data;
  const md = mask.data;
  const period = mask.size;
  if (xd.length % period !== 0 || !shapeEq(mask.shape, x.shape.slice(x.value.rank - mask.rank))) {
    throw new Error(
      `mask ${fmtShape(mask.shape)} does not match trailing axes of ${fmtShape(x.shape)}`,
    );
  }
  const out = alloc(xd.length, xd);
  for (let i = 0; i < xd.length; i++) out[i] = md[i % period] !== 0 ? value : xd[i];
  return op(
    'maskedFill',
    new Tensor(x.shape, out),
    [x],
    (g) => {
      const gx = alloc(xd.length, xd);
      for (let i = 0; i < xd.length; i++) gx[i] = md[i % period] !== 0 ? 0 : g.data[i];
      return [new Tensor(x.shape, gx)];
    },
    { value: Number.isFinite(value) ? value : String(value) },
    { mask },
  );
}
