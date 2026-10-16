import { Tensor, alloc, fmtShape, type Data } from '../tensor';
import { op, type Var } from '../graph';

// Kernels are register-blocked over four rows (or columns) so each value
// loaded from the shared operand is used four times. Exactness is unchanged:
// every output is the same sum of products, accumulated in the same order.

/** C[m,n] (+)= A[m,k] · B[k,n] */
export function mmNN(
  A: Data,
  aOff: number,
  B: Data,
  bOff: number,
  C: Data,
  cOff: number,
  m: number,
  k: number,
  n: number,
  accumulate = false,
): void {
  if (!accumulate) C.fill(0, cOff, cOff + m * n);
  let i = 0;
  for (; i + 3 < m; i += 4) {
    const c0 = cOff + i * n,
      c1 = c0 + n,
      c2 = c1 + n,
      c3 = c2 + n;
    const a0 = aOff + i * k,
      a1 = a0 + k,
      a2 = a1 + k,
      a3 = a2 + k;
    for (let p = 0; p < k; p++) {
      const x0 = A[a0 + p],
        x1 = A[a1 + p],
        x2 = A[a2 + p],
        x3 = A[a3 + p];
      if (x0 === 0 && x1 === 0 && x2 === 0 && x3 === 0) continue;
      const bp = bOff + p * n;
      for (let j = 0; j < n; j++) {
        const b = B[bp + j];
        C[c0 + j] += x0 * b;
        C[c1 + j] += x1 * b;
        C[c2 + j] += x2 * b;
        C[c3 + j] += x3 * b;
      }
    }
  }
  for (; i < m; i++) {
    const ci = cOff + i * n;
    const ai = aOff + i * k;
    for (let p = 0; p < k; p++) {
      const a = A[ai + p];
      if (a === 0) continue;
      const bp = bOff + p * n;
      for (let j = 0; j < n; j++) C[ci + j] += a * B[bp + j];
    }
  }
}

/** C[m,n] (+)= A[m,k] · B[n,k]ᵀ */
export function mmNT(
  A: Data,
  aOff: number,
  B: Data,
  bOff: number,
  C: Data,
  cOff: number,
  m: number,
  k: number,
  n: number,
  accumulate = false,
): void {
  for (let i = 0; i < m; i++) {
    const ai = aOff + i * k;
    const ci = cOff + i * n;
    let j = 0;
    for (; j + 3 < n; j += 4) {
      const b0 = bOff + j * k,
        b1 = b0 + k,
        b2 = b1 + k,
        b3 = b2 + k;
      let s0 = 0,
        s1 = 0,
        s2 = 0,
        s3 = 0;
      for (let p = 0; p < k; p++) {
        const a = A[ai + p];
        s0 += a * B[b0 + p];
        s1 += a * B[b1 + p];
        s2 += a * B[b2 + p];
        s3 += a * B[b3 + p];
      }
      if (accumulate) {
        C[ci + j] += s0;
        C[ci + j + 1] += s1;
        C[ci + j + 2] += s2;
        C[ci + j + 3] += s3;
      } else {
        C[ci + j] = s0;
        C[ci + j + 1] = s1;
        C[ci + j + 2] = s2;
        C[ci + j + 3] = s3;
      }
    }
    for (; j < n; j++) {
      const bj = bOff + j * k;
      let s = 0;
      for (let p = 0; p < k; p++) s += A[ai + p] * B[bj + p];
      C[ci + j] = accumulate ? C[ci + j] + s : s;
    }
  }
}

/** C[m,n] (+)= A[k,m]ᵀ · B[k,n] */
export function mmTN(
  A: Data,
  aOff: number,
  B: Data,
  bOff: number,
  C: Data,
  cOff: number,
  m: number,
  k: number,
  n: number,
  accumulate = false,
): void {
  if (!accumulate) C.fill(0, cOff, cOff + m * n);
  let i = 0;
  for (; i + 3 < m; i += 4) {
    const c0 = cOff + i * n,
      c1 = c0 + n,
      c2 = c1 + n,
      c3 = c2 + n;
    for (let p = 0; p < k; p++) {
      const ap = aOff + p * m + i;
      const x0 = A[ap],
        x1 = A[ap + 1],
        x2 = A[ap + 2],
        x3 = A[ap + 3];
      if (x0 === 0 && x1 === 0 && x2 === 0 && x3 === 0) continue;
      const bp = bOff + p * n;
      for (let j = 0; j < n; j++) {
        const b = B[bp + j];
        C[c0 + j] += x0 * b;
        C[c1 + j] += x1 * b;
        C[c2 + j] += x2 * b;
        C[c3 + j] += x3 * b;
      }
    }
  }
  for (; i < m; i++) {
    const ci = cOff + i * n;
    for (let p = 0; p < k; p++) {
      const a = A[aOff + p * m + i];
      if (a === 0) continue;
      const bp = bOff + p * n;
      for (let j = 0; j < n; j++) C[ci + j] += a * B[bp + j];
    }
  }
}

function batchOf(shape: number[]): number {
  let b = 1;
  for (let i = 0; i < shape.length - 2; i++) b *= shape[i];
  return b;
}

/**
 * Matrix product over the last two axes. Leading (batch) axes must match,
 * or one operand must be a plain matrix that is shared across the batch.
 */
export function matmul(a: Var, b: Var): Var {
  const A = a.value;
  const B = b.value;
  if (A.rank < 2 || B.rank < 2) throw new Error('matmul needs rank ≥ 2 operands');
  const m = A.shape[A.rank - 2];
  const k = A.shape[A.rank - 1];
  const k2 = B.shape[B.rank - 2];
  const n = B.shape[B.rank - 1];
  if (k !== k2) throw new Error(`matmul inner dims ${fmtShape(A.shape)} · ${fmtShape(B.shape)}`);

  if (B.rank === 2) {
    // [..., m, k] · [k, n]: fold batch into rows.
    const rows = A.size / k;
    const out = alloc(rows * n, A.data);
    mmNN(A.data, 0, B.data, 0, out, 0, rows, k, n);
    const y = new Tensor([...A.shape.slice(0, -1), n], out);
    return op('matmul', y, [a, b], (g) => {
      const ga = alloc(A.size, A.data);
      mmNT(g.data, 0, B.data, 0, ga, 0, rows, n, k);
      const gb = alloc(B.size, B.data);
      mmTN(A.data, 0, g.data, 0, gb, 0, k, rows, n);
      return [new Tensor(A.shape, ga), new Tensor(B.shape, gb)];
    });
  }

  const batchA = A.rank === 2 ? 1 : batchOf(A.shape);
  const batchB = batchOf(B.shape);
  const sharedA = A.rank === 2;
  if (!sharedA) {
    const la = A.shape.slice(0, -2).join(',');
    const lb = B.shape.slice(0, -2).join(',');
    if (la !== lb) throw new Error(`matmul batch dims ${fmtShape(A.shape)} · ${fmtShape(B.shape)}`);
  }
  const batch = Math.max(batchA, batchB);
  const out = alloc(batch * m * n, A.data);
  for (let t = 0; t < batch; t++) {
    mmNN(A.data, sharedA ? 0 : t * m * k, B.data, t * k * n, out, t * m * n, m, k, n);
  }
  const lead = B.shape.slice(0, -2);
  const y = new Tensor([...lead, m, n], out);
  return op('matmul', y, [a, b], (g) => {
    const ga = alloc(A.size, A.data);
    const gb = alloc(B.size, B.data);
    for (let t = 0; t < batch; t++) {
      mmNT(g.data, t * m * n, B.data, t * k * n, ga, sharedA ? 0 : t * m * k, m, n, k, sharedA);
      mmTN(A.data, sharedA ? 0 : t * m * k, g.data, t * m * n, gb, t * k * n, k, m, n);
    }
    return [new Tensor(A.shape, ga), new Tensor(B.shape, gb)];
  });
}

/**
 * Affine map y = x Wᵀ + b with W of shape [out, in] (PyTorch convention).
 * x may have any number of leading axes.
 */
export function linear(x: Var, w: Var, b?: Var): Var {
  const X = x.value;
  const W = w.value;
  const nIn = X.shape[X.rank - 1];
  const nOut = W.shape[0];
  if (W.shape[1] !== nIn) {
    throw new Error(`linear: input ${fmtShape(X.shape)} vs weight ${fmtShape(W.shape)}`);
  }
  const rows = X.size / nIn;
  const out = alloc(rows * nOut, X.data);
  mmNT(X.data, 0, W.data, 0, out, 0, rows, nIn, nOut);
  if (b) {
    const bd = b.value.data;
    for (let r = 0; r < rows; r++) for (let j = 0; j < nOut; j++) out[r * nOut + j] += bd[j];
  }
  const y = new Tensor([...X.shape.slice(0, -1), nOut], out);
  const parents = b ? [x, w, b] : [x, w];
  return op(
    'linear',
    y,
    parents,
    (g) => {
      const gx = alloc(X.size, X.data);
      mmNN(g.data, 0, W.data, 0, gx, 0, rows, nOut, nIn);
      const gw = alloc(W.size, W.data);
      mmTN(g.data, 0, X.data, 0, gw, 0, nOut, rows, nIn);
      const grads: (Tensor | null)[] = [new Tensor(X.shape, gx), new Tensor(W.shape, gw)];
      if (b) {
        const gb = alloc(nOut, X.data);
        for (let r = 0; r < rows; r++) for (let j = 0; j < nOut; j++) gb[j] += g.data[r * nOut + j];
        grads.push(new Tensor([nOut], gb));
      }
      return grads;
    },
    { in: nIn, out: nOut, bias: !!b },
  );
}
