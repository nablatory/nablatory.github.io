import { Tensor, alloc } from '../tensor';
import { op, isTraining, type Var } from '../graph';

/**
 * Layer normalisation over the last axis:
 * y = γ ⊙ (x − μ) / √(σ² + ε) + β, with μ, σ² the row mean and biased variance.
 */
export function layerNorm(x: Var, gamma?: Var, beta?: Var, eps = 1e-5): Var {
  const D = x.shape[x.value.rank - 1];
  const rows = x.value.size / D;
  const xd = x.value.data;
  const xhat = alloc(xd.length, xd);
  const mu = alloc(rows, xd);
  const rstd = alloc(rows, xd);
  const out = alloc(xd.length, xd);
  const gd = gamma?.value.data;
  const bd = beta?.value.data;
  for (let r = 0; r < rows; r++) {
    let m = 0;
    for (let j = 0; j < D; j++) m += xd[r * D + j];
    m /= D;
    let v = 0;
    for (let j = 0; j < D; j++) {
      const c = xd[r * D + j] - m;
      v += c * c;
    }
    v /= D;
    const rs = 1 / Math.sqrt(v + eps);
    mu[r] = m;
    rstd[r] = rs;
    for (let j = 0; j < D; j++) {
      const k = r * D + j;
      const h = (xd[k] - m) * rs;
      xhat[k] = h;
      out[k] = (gd ? gd[j] : 1) * h + (bd ? bd[j] : 0);
    }
  }
  const parents = [x, ...(gamma ? [gamma] : []), ...(beta ? [beta] : [])];
  return op(
    'layerNorm',
    new Tensor(x.shape, out),
    parents,
    (g) => {
      const gx = alloc(xd.length, xd);
      const gg = gamma ? alloc(D, xd) : null;
      const gb = beta ? alloc(D, xd) : null;
      for (let r = 0; r < rows; r++) {
        let s1 = 0;
        let s2 = 0;
        for (let j = 0; j < D; j++) {
          const k = r * D + j;
          const dh = g.data[k] * (gd ? gd[j] : 1);
          s1 += dh;
          s2 += dh * xhat[k];
          if (gg) gg[j] += g.data[k] * xhat[k];
          if (gb) gb[j] += g.data[k];
        }
        s1 /= D;
        s2 /= D;
        for (let j = 0; j < D; j++) {
          const k = r * D + j;
          const dh = g.data[k] * (gd ? gd[j] : 1);
          gx[k] = rstd[r] * (dh - s1 - xhat[k] * s2);
        }
      }
      const grads: (Tensor | null)[] = [new Tensor(x.shape, gx)];
      if (gg) grads.push(new Tensor([D], gg));
      if (gb) grads.push(new Tensor([D], gb));
      return grads;
    },
    { eps, axis: -1 },
    {
      mean: new Tensor([...x.shape.slice(0, -1), 1], mu),
      rstd: new Tensor([...x.shape.slice(0, -1), 1], rstd),
    },
  );
}

/** RMS normalisation over the last axis: y = γ ⊙ x / √(mean(x²) + ε). */
export function rmsNorm(x: Var, gamma?: Var, eps = 1e-5): Var {
  const D = x.shape[x.value.rank - 1];
  const rows = x.value.size / D;
  const xd = x.value.data;
  const rinv = alloc(rows, xd);
  const out = alloc(xd.length, xd);
  const gd = gamma?.value.data;
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let j = 0; j < D; j++) s += xd[r * D + j] * xd[r * D + j];
    const ri = 1 / Math.sqrt(s / D + eps);
    rinv[r] = ri;
    for (let j = 0; j < D; j++) out[r * D + j] = xd[r * D + j] * ri * (gd ? gd[j] : 1);
  }
  const parents = gamma ? [x, gamma] : [x];
  return op(
    'rmsNorm',
    new Tensor(x.shape, out),
    parents,
    (g) => {
      const gx = alloc(xd.length, xd);
      const gg = gamma ? alloc(D, xd) : null;
      for (let r = 0; r < rows; r++) {
        const ri = rinv[r];
        let dot = 0;
        for (let j = 0; j < D; j++) {
          const k = r * D + j;
          const dh = g.data[k] * (gd ? gd[j] : 1);
          dot += dh * xd[k] * ri;
          if (gg) gg[j] += g.data[k] * xd[k] * ri;
        }
        dot /= D;
        for (let j = 0; j < D; j++) {
          const k = r * D + j;
          const dh = g.data[k] * (gd ? gd[j] : 1);
          gx[k] = ri * (dh - xd[k] * ri * dot);
        }
      }
      return gg ? [new Tensor(x.shape, gx), new Tensor([D], gg)] : [new Tensor(x.shape, gx)];
    },
    { eps },
    { rinv: new Tensor([...x.shape.slice(0, -1), 1], rinv) },
  );
}

export interface RunningStats {
  mean: Tensor;
  var: Tensor;
  momentum: number;
}

/**
 * Batch normalisation for [N, C] or [N, C, H, W]. In training mode the batch
 * statistics are used and the running statistics are updated in place
 * (unbiased variance, PyTorch convention); in evaluation mode the running
 * statistics are used.
 */
export function batchNorm(x: Var, gamma: Var, beta: Var, running: RunningStats, eps = 1e-5): Var {
  const training = isTraining();
  const [N, C] = x.shape;
  const S = x.value.rank === 4 ? x.shape[2] * x.shape[3] : 1;
  const M = N * S;
  const xd = x.value.data;
  const mu = alloc(C, xd);
  const rstd = alloc(C, xd);
  if (training) {
    for (let c = 0; c < C; c++) {
      let m = 0;
      for (let n = 0; n < N; n++) for (let s = 0; s < S; s++) m += xd[(n * C + c) * S + s];
      m /= M;
      let v = 0;
      for (let n = 0; n < N; n++)
        for (let s = 0; s < S; s++) {
          const d = xd[(n * C + c) * S + s] - m;
          v += d * d;
        }
      const vb = v / M;
      mu[c] = m;
      rstd[c] = 1 / Math.sqrt(vb + eps);
      const mom = running.momentum;
      running.mean.data[c] = (1 - mom) * running.mean.data[c] + mom * m;
      running.var.data[c] = (1 - mom) * running.var.data[c] + mom * (M > 1 ? v / (M - 1) : vb);
    }
  } else {
    for (let c = 0; c < C; c++) {
      mu[c] = running.mean.data[c];
      rstd[c] = 1 / Math.sqrt(running.var.data[c] + eps);
    }
  }
  const gd = gamma.value.data;
  const bd = beta.value.data;
  const xhat = alloc(xd.length, xd);
  const out = alloc(xd.length, xd);
  for (let n = 0; n < N; n++)
    for (let c = 0; c < C; c++)
      for (let s = 0; s < S; s++) {
        const k = (n * C + c) * S + s;
        xhat[k] = (xd[k] - mu[c]) * rstd[c];
        out[k] = gd[c] * xhat[k] + bd[c];
      }
  return op(
    'batchNorm',
    new Tensor(x.shape, out),
    [x, gamma, beta],
    (g) => {
      const gx = alloc(xd.length, xd);
      const gg = alloc(C, xd);
      const gb = alloc(C, xd);
      for (let c = 0; c < C; c++) {
        let s1 = 0;
        let s2 = 0;
        for (let n = 0; n < N; n++)
          for (let s = 0; s < S; s++) {
            const k = (n * C + c) * S + s;
            s1 += g.data[k];
            s2 += g.data[k] * xhat[k];
          }
        gb[c] = s1;
        gg[c] = s2;
        for (let n = 0; n < N; n++)
          for (let s = 0; s < S; s++) {
            const k = (n * C + c) * S + s;
            gx[k] = training
              ? gd[c] * rstd[c] * (g.data[k] - s1 / M - (xhat[k] * s2) / M)
              : gd[c] * rstd[c] * g.data[k];
          }
      }
      return [new Tensor(x.shape, gx), new Tensor([C], gg), new Tensor([C], gb)];
    },
    { eps, mode: training ? 'batch statistics' : 'running statistics' },
    { mean: new Tensor([C], mu), rstd: new Tensor([C], rstd) },
  );
}
