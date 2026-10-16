import { Tensor, alloc, fmtShape } from '../../engine';
import { op, type Var } from '../../engine/graph';

/**
 * Selective scan (Gu & Dao 2023, Algorithm 2) with zero-order-hold Ā and the
 * Euler B̄ used by the reference implementation:
 *
 *   h_t[e, n] = exp(Δ_t[e] A[e, n]) · h_{t−1}[e, n] + Δ_t[e] B_t[n] x_t[e],   h_{−1} = 0
 *
 * Shapes: delta, x [Bt, L, E]; A [E, N]; B [Bt, L, N]. Returns every state
 * h [Bt, L, E, N] so that y_t = C_t · h_t + D x_t can be formed with ordinary
 * ops (and the states can be displayed). The scan runs sequentially over t.
 *
 * Backward pass, with g_t = ∂L/∂h_t accumulated from later steps:
 *   g_t ← ĝ_t + exp(Δ_{t+1} A) ⊙ g_{t+1}
 *   ∂Δ_t[e] = Σ_n g_t (A exp(Δ_t A) h_{t−1} + B_t x_t)
 *   ∂A[e,n] = Σ_t g_t Δ_t exp(Δ_t A) h_{t−1}
 *   ∂B_t[n] = Σ_e g_t Δ_t x_t,   ∂x_t[e] = Σ_n g_t Δ_t B_t
 */
export function selectiveScan(delta: Var, A: Var, B: Var, x: Var): Var {
  const [Bt, L, E] = delta.shape;
  const N = A.shape[1];
  if (
    x.shape.join() !== delta.shape.join() ||
    A.shape[0] !== E ||
    B.shape.join() !== [Bt, L, N].join()
  ) {
    throw new Error(
      `selectiveScan shapes: delta ${fmtShape(delta.shape)}, A ${fmtShape(A.shape)}, B ${fmtShape(B.shape)}, x ${fmtShape(x.shape)}`,
    );
  }
  const dd = delta.value.data;
  const ad = A.value.data;
  const bd = B.value.data;
  const xd = x.value.data;
  const h = alloc(Bt * L * E * N, dd);
  for (let b = 0; b < Bt; b++)
    for (let t = 0; t < L; t++) {
      const row = (b * L + t) * E;
      const bRow = (b * L + t) * N;
      for (let e = 0; e < E; e++) {
        const dt = dd[row + e];
        const u = dt * xd[row + e];
        const o = (row + e) * N;
        const prev = o - E * N;
        for (let n = 0; n < N; n++) {
          const carry = t > 0 ? Math.exp(dt * ad[e * N + n]) * h[prev + n] : 0;
          h[o + n] = carry + u * bd[bRow + n];
        }
      }
    }
  return op(
    'selectiveScan',
    new Tensor([Bt, L, E, N], h),
    [delta, A, B, x],
    (gOut) => {
      const go = gOut.data;
      const gDelta = alloc(dd.length, dd);
      const gA = alloc(ad.length, dd);
      const gB = alloc(bd.length, dd);
      const gx = alloc(xd.length, dd);
      const g = alloc(E * N, dd);
      for (let b = 0; b < Bt; b++) {
        g.fill(0);
        for (let t = L - 1; t >= 0; t--) {
          const row = (b * L + t) * E;
          const bRow = (b * L + t) * N;
          for (let e = 0; e < E; e++) {
            const dt = dd[row + e];
            const xv = xd[row + e];
            const o = (row + e) * N;
            let sDelta = 0;
            let sx = 0;
            for (let n = 0; n < N; n++) {
              const k = e * N + n;
              const gt = g[k] + go[o + n];
              const bn = bd[bRow + n];
              sDelta += gt * bn * xv;
              sx += gt * dt * bn;
              gB[bRow + n] += gt * dt * xv;
              if (t > 0) {
                const a = ad[k];
                const dA = Math.exp(dt * a);
                const hp = h[o - E * N + n];
                sDelta += gt * a * dA * hp;
                gA[k] += gt * dt * dA * hp;
                g[k] = gt * dA;
              } else {
                g[k] = 0;
              }
            }
            gDelta[row + e] += sDelta;
            gx[row + e] += sx;
          }
        }
      }
      return [
        new Tensor(delta.shape, gDelta),
        new Tensor(A.shape, gA),
        new Tensor(B.shape, gB),
        new Tensor(x.shape, gx),
      ];
    },
    { L, E, N },
  );
}
