import { Tensor, alloc, fmtShape, type Data } from '../tensor';
import { op, type Var } from '../graph';
import { mmNN, mmNT, mmTN } from './linalg';

export interface Conv2dOptions {
  stride?: number | [number, number];
  padding?: number | [number, number];
  dilation?: number | [number, number];
  groups?: number;
}

const pair = (v: number | [number, number] | undefined, d: number): [number, number] =>
  v === undefined ? [d, d] : typeof v === 'number' ? [v, v] : v;

export function convOutSize(n: number, k: number, s: number, p: number, d = 1): number {
  return Math.floor((n + 2 * p - d * (k - 1) - 1) / s) + 1;
}

interface Geo {
  C: number;
  H: number;
  W: number;
  kh: number;
  kw: number;
  sh: number;
  sw: number;
  ph: number;
  pw: number;
  dh: number;
  dw: number;
  OH: number;
  OW: number;
}

/** cols[(c·kh + i)·kw + j, oy·OW + ox] = x[c, oy·sh − ph + i·dh, ox·sw − pw + j·dw] */
function im2col(x: Data, xOff: number, g: Geo, cols: Data): void {
  const { C, H, W, kh, kw, sh, sw, ph, pw, dh, dw, OH, OW } = g;
  const P = OH * OW;
  for (let c = 0; c < C; c++)
    for (let i = 0; i < kh; i++)
      for (let j = 0; j < kw; j++) {
        const row = ((c * kh + i) * kw + j) * P;
        for (let oy = 0; oy < OH; oy++) {
          const iy = oy * sh - ph + i * dh;
          const r = row + oy * OW;
          if (iy < 0 || iy >= H) {
            for (let ox = 0; ox < OW; ox++) cols[r + ox] = 0;
            continue;
          }
          const xr = xOff + (c * H + iy) * W;
          for (let ox = 0; ox < OW; ox++) {
            const ix = ox * sw - pw + j * dw;
            cols[r + ox] = ix >= 0 && ix < W ? x[xr + ix] : 0;
          }
        }
      }
}

/** Adjoint of im2col: scatter-add columns back into an image. */
function col2im(cols: Data, g: Geo, x: Data, xOff: number): void {
  const { C, H, W, kh, kw, sh, sw, ph, pw, dh, dw, OH, OW } = g;
  const P = OH * OW;
  for (let c = 0; c < C; c++)
    for (let i = 0; i < kh; i++)
      for (let j = 0; j < kw; j++) {
        const row = ((c * kh + i) * kw + j) * P;
        for (let oy = 0; oy < OH; oy++) {
          const iy = oy * sh - ph + i * dh;
          if (iy < 0 || iy >= H) continue;
          const xr = xOff + (c * H + iy) * W;
          const r = row + oy * OW;
          for (let ox = 0; ox < OW; ox++) {
            const ix = ox * sw - pw + j * dw;
            if (ix >= 0 && ix < W) x[xr + ix] += cols[r + ox];
          }
        }
      }
}

/**
 * 2-D cross-correlation (the deep-learning "convolution"), NCHW layout.
 * y[n,o,oy,ox] = b[o] + Σ_{c,i,j} w[o,c,i,j] · x[n, g·C_g + c, oy·s − p + i·d, ox·s − p + j·d]
 */
export function conv2d(x: Var, w: Var, b: Var | undefined, opts: Conv2dOptions = {}): Var {
  const X = x.value;
  const Wt = w.value;
  if (X.rank !== 4 || Wt.rank !== 4) {
    throw new Error(
      `conv2d expects NCHW input and OIHW weight, got ${fmtShape(X.shape)}, ${fmtShape(Wt.shape)}`,
    );
  }
  const [N, C, H, Wd] = X.shape;
  const [O, Cg, kh, kw] = Wt.shape;
  const groups = opts.groups ?? 1;
  if (C !== Cg * groups || O % groups !== 0) {
    throw new Error(
      `conv2d channels: input C=${C}, weight ${fmtShape(Wt.shape)}, groups=${groups}`,
    );
  }
  const [sh, sw] = pair(opts.stride, 1);
  const [ph, pw] = pair(opts.padding, 0);
  const [dh, dw] = pair(opts.dilation, 1);
  const OH = convOutSize(H, kh, sh, ph, dh);
  const OW = convOutSize(Wd, kw, sw, pw, dw);
  if (OH <= 0 || OW <= 0) throw new Error(`conv2d output would be empty (${OH}×${OW})`);
  const geo: Geo = { C, H, W: Wd, kh, kw, sh, sw, ph, pw, dh, dw, OH, OW };
  const P = OH * OW;
  const K = Cg * kh * kw;
  const Og = O / groups;
  const colsAll: Data[] = [];
  const out = alloc(N * O * P, X.data);
  for (let n = 0; n < N; n++) {
    const cols = alloc(C * kh * kw * P, X.data);
    im2col(X.data, n * C * H * Wd, geo, cols);
    colsAll.push(cols);
    for (let gi = 0; gi < groups; gi++) {
      mmNN(Wt.data, gi * Og * K, cols, gi * K * P, out, (n * O + gi * Og) * P, Og, K, P);
    }
    if (b) {
      const bd = b.value.data;
      for (let o = 0; o < O; o++) {
        const base = (n * O + o) * P;
        for (let p = 0; p < P; p++) out[base + p] += bd[o];
      }
    }
  }
  const y = new Tensor([N, O, OH, OW], out);
  const parents = b ? [x, w, b] : [x, w];
  return op(
    'conv2d',
    y,
    parents,
    (g) => {
      const gx = alloc(X.size, X.data);
      const gw = alloc(Wt.size, X.data);
      const dcols = alloc(C * kh * kw * P, X.data);
      for (let n = 0; n < N; n++) {
        for (let gi = 0; gi < groups; gi++) {
          const gOff = (n * O + gi * Og) * P;
          mmNT(g.data, gOff, colsAll[n], gi * K * P, gw, gi * Og * K, Og, P, K, true);
          mmTN(Wt.data, gi * Og * K, g.data, gOff, dcols, gi * K * P, K, Og, P);
        }
        col2im(dcols, geo, gx, n * C * H * Wd);
      }
      const grads: (Tensor | null)[] = [new Tensor(X.shape, gx), new Tensor(Wt.shape, gw)];
      if (b) {
        const gb = alloc(O, X.data);
        for (let n = 0; n < N; n++)
          for (let o = 0; o < O; o++) {
            const base = (n * O + o) * P;
            let s = 0;
            for (let p = 0; p < P; p++) s += g.data[base + p];
            gb[o] += s;
          }
        grads.push(new Tensor([O], gb));
      }
      return grads;
    },
    { stride: [sh, sw], padding: [ph, pw], dilation: [dh, dw], groups, kernel: [kh, kw] },
  );
}

export interface ConvTranspose2dOptions {
  stride?: number;
  padding?: number;
  outputPadding?: number;
}

/**
 * Transposed convolution (adjoint of conv2d with respect to its input).
 * Weight layout [C_in, C_out, kh, kw], as in PyTorch.
 */
export function convTranspose2d(
  x: Var,
  w: Var,
  b: Var | undefined,
  opts: ConvTranspose2dOptions = {},
): Var {
  const X = x.value;
  const Wt = w.value;
  const [N, C, H, Wd] = X.shape;
  const [Cw, O, kh, kw] = Wt.shape;
  if (C !== Cw) throw new Error(`convTranspose2d: input C=${C}, weight ${fmtShape(Wt.shape)}`);
  const s = opts.stride ?? 1;
  const p = opts.padding ?? 0;
  const op_ = opts.outputPadding ?? 0;
  const OH = (H - 1) * s - 2 * p + (kh - 1) + op_ + 1;
  const OW = (Wd - 1) * s - 2 * p + (kw - 1) + op_ + 1;
  // Geometry of the forward conv that maps [O, OH, OW] → [C, H, W].
  const geo: Geo = {
    C: O,
    H: OH,
    W: OW,
    kh,
    kw,
    sh: s,
    sw: s,
    ph: p,
    pw: p,
    dh: 1,
    dw: 1,
    OH: H,
    OW: Wd,
  };
  const KK = O * kh * kw;
  const HW = H * Wd;
  const out = alloc(N * O * OH * OW, X.data);
  const cols = alloc(KK * HW, X.data);
  for (let n = 0; n < N; n++) {
    mmTN(Wt.data, 0, X.data, n * C * HW, cols, 0, KK, C, HW);
    col2im(cols, geo, out, n * O * OH * OW);
    if (b) {
      const bd = b.value.data;
      for (let o = 0; o < O; o++) {
        const base = (n * O + o) * OH * OW;
        for (let q = 0; q < OH * OW; q++) out[base + q] += bd[o];
      }
    }
  }
  const parents = b ? [x, w, b] : [x, w];
  return op(
    'convTranspose2d',
    new Tensor([N, O, OH, OW], out),
    parents,
    (g) => {
      const gx = alloc(X.size, X.data);
      const gw = alloc(Wt.size, X.data);
      const dcols = alloc(KK * HW, X.data);
      for (let n = 0; n < N; n++) {
        im2col(g.data, n * O * OH * OW, geo, dcols);
        mmNN(Wt.data, 0, dcols, 0, gx, n * C * HW, C, KK, HW);
        mmNT(X.data, n * C * HW, dcols, 0, gw, 0, C, HW, KK, true);
      }
      const grads: (Tensor | null)[] = [new Tensor(X.shape, gx), new Tensor(Wt.shape, gw)];
      if (b) {
        const gb = alloc(O, X.data);
        for (let n = 0; n < N; n++)
          for (let o = 0; o < O; o++) {
            const base = (n * O + o) * OH * OW;
            for (let q = 0; q < OH * OW; q++) gb[o] += g.data[base + q];
          }
        grads.push(new Tensor([O], gb));
      }
      return grads;
    },
    { stride: s, padding: p, outputPadding: op_, kernel: [kh, kw] },
  );
}

export function maxPool2d(x: Var, k: number, stride = k, padding = 0): Var {
  const [N, C, H, W] = x.shape;
  const OH = convOutSize(H, k, stride, padding);
  const OW = convOutSize(W, k, stride, padding);
  const xd = x.value.data;
  const out = alloc(N * C * OH * OW, xd);
  const arg = new Int32Array(out.length);
  for (let nc = 0; nc < N * C; nc++) {
    const base = nc * H * W;
    for (let oy = 0; oy < OH; oy++)
      for (let ox = 0; ox < OW; ox++) {
        let best = -Infinity;
        let bi = -1;
        for (let i = 0; i < k; i++) {
          const iy = oy * stride - padding + i;
          if (iy < 0 || iy >= H) continue;
          for (let j = 0; j < k; j++) {
            const ix = ox * stride - padding + j;
            if (ix < 0 || ix >= W) continue;
            const v = xd[base + iy * W + ix];
            if (v > best) {
              best = v;
              bi = base + iy * W + ix;
            }
          }
        }
        const o = (nc * OH + oy) * OW + ox;
        out[o] = best;
        arg[o] = bi;
      }
  }
  const argT = new Tensor([N, C, OH, OW], Float32Array.from(arg));
  return op(
    'maxPool2d',
    new Tensor([N, C, OH, OW], out),
    [x],
    (g) => {
      const gx = alloc(xd.length, xd);
      for (let o = 0; o < out.length; o++) if (arg[o] >= 0) gx[arg[o]] += g.data[o];
      return [new Tensor(x.shape, gx)];
    },
    { kernel: k, stride, padding },
    { argmax: argT },
  );
}

/** Average pooling; padded positions count toward the divisor (PyTorch default). */
export function avgPool2d(x: Var, k: number, stride = k, padding = 0): Var {
  const [N, C, H, W] = x.shape;
  const OH = convOutSize(H, k, stride, padding);
  const OW = convOutSize(W, k, stride, padding);
  const xd = x.value.data;
  const out = alloc(N * C * OH * OW, xd);
  const inv = 1 / (k * k);
  for (let nc = 0; nc < N * C; nc++) {
    const base = nc * H * W;
    for (let oy = 0; oy < OH; oy++)
      for (let ox = 0; ox < OW; ox++) {
        let s = 0;
        for (let i = 0; i < k; i++) {
          const iy = oy * stride - padding + i;
          if (iy < 0 || iy >= H) continue;
          for (let j = 0; j < k; j++) {
            const ix = ox * stride - padding + j;
            if (ix >= 0 && ix < W) s += xd[base + iy * W + ix];
          }
        }
        out[(nc * OH + oy) * OW + ox] = s * inv;
      }
  }
  return op(
    'avgPool2d',
    new Tensor([N, C, OH, OW], out),
    [x],
    (g) => {
      const gx = alloc(xd.length, xd);
      for (let nc = 0; nc < N * C; nc++) {
        const base = nc * H * W;
        for (let oy = 0; oy < OH; oy++)
          for (let ox = 0; ox < OW; ox++) {
            const gv = g.data[(nc * OH + oy) * OW + ox] * inv;
            for (let i = 0; i < k; i++) {
              const iy = oy * stride - padding + i;
              if (iy < 0 || iy >= H) continue;
              for (let j = 0; j < k; j++) {
                const ix = ox * stride - padding + j;
                if (ix >= 0 && ix < W) gx[base + iy * W + ix] += gv;
              }
            }
          }
      }
      return [new Tensor(x.shape, gx)];
    },
    { kernel: k, stride, padding },
  );
}

export function upsampleNearest2d(x: Var, scale: number): Var {
  const [N, C, H, W] = x.shape;
  const OH = H * scale;
  const OW = W * scale;
  const xd = x.value.data;
  const out = alloc(N * C * OH * OW, xd);
  for (let nc = 0; nc < N * C; nc++)
    for (let oy = 0; oy < OH; oy++)
      for (let ox = 0; ox < OW; ox++) {
        out[(nc * OH + oy) * OW + ox] =
          xd[(nc * H + Math.floor(oy / scale)) * W + Math.floor(ox / scale)];
      }
  return op(
    'upsampleNearest2d',
    new Tensor([N, C, OH, OW], out),
    [x],
    (g) => {
      const gx = alloc(xd.length, xd);
      for (let nc = 0; nc < N * C; nc++)
        for (let oy = 0; oy < OH; oy++)
          for (let ox = 0; ox < OW; ox++) {
            gx[(nc * H + Math.floor(oy / scale)) * W + Math.floor(ox / scale)] +=
              g.data[(nc * OH + oy) * OW + ox];
          }
      return [new Tensor(x.shape, gx)];
    },
    { scale },
  );
}
