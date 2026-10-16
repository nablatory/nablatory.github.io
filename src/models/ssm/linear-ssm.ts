import {
  ParamBuilder,
  Tensor,
  add,
  addScalar,
  concat,
  cos,
  defaultOptimizer,
  div,
  exp,
  linear,
  mse,
  mul,
  mulScalar,
  neg,
  p,
  reshape,
  scope,
  select,
  sin,
  square,
  stack,
  sub,
  sum,
  take,
  type Var,
} from '../../engine';
import { defineModel, type Batch } from '../types';
import {
  FILTER_OPTIONS,
  filterDataset,
  targetKernel,
  type FilterKind,
} from '../data/ssm-sequences';
import { constant, input } from '../util';

export type LinearSsmConfig = {
  filter: FilterKind;
  lag: number;
  length: number;
  modes: number;
  mode: 'recurrent' | 'convolution';
  dtInit: number;
};

interface Discrete {
  /** Ā = exp(ΔA), real and imaginary parts, [N]. */
  abRe: Var;
  abIm: Var;
  /** B̄ = (ΔA)⁻¹(exp(ΔA) − 1) ΔB, [N]. */
  bbRe: Var;
  bbIm: Var;
  /** Δ·A, [N]. */
  dtARe: Var;
  dtAIm: Var;
}

/** Zero-order hold for a diagonal complex A, elementwise and exact. */
export function discretize(): Discrete {
  return scope('discretize', () => {
    const dt = exp(p('ssm.log_dt'));
    const aRe = neg(exp(p('ssm.log_A_real')));
    const aIm = p('ssm.A_imag');
    const dtARe = mul(dt, aRe);
    const dtAIm = mul(dt, aIm);
    const mag = exp(dtARe);
    const abRe = mul(mag, cos(dtAIm));
    const abIm = mul(mag, sin(dtAIm));
    // (Ā − 1) / A  =  (Ā − 1) · conj(A) / |A|²
    const nRe = addScalar(abRe, -1);
    const den = add(square(aRe), square(aIm));
    const fRe = div(add(mul(nRe, aRe), mul(abIm, aIm)), den);
    const fIm = div(sub(mul(abIm, aRe), mul(nRe, aIm)), den);
    const bRe = p('ssm.B_re');
    const bIm = p('ssm.B_im');
    const bbRe = sub(mul(fRe, bRe), mul(fIm, bIm));
    const bbIm = add(mul(fRe, bIm), mul(fIm, bRe));
    return { abRe, abIm, bbRe, bbIm, dtARe, dtAIm };
  });
}

/** K_k = 2 Re(Σ_n C_n Ā_n^k B̄_n), k = 0 … L−1, with Ā^k = exp(kΔA) in closed form. */
export function ssmKernel(d: Discrete, L: number): Var {
  return scope('kernel', () => {
    const N = d.abRe.shape[0];
    const k = constant(
      new Tensor(
        [L, 1],
        Array.from({ length: L }, (_, i) => i),
      ),
    );
    const mag = exp(mul(k, reshape(d.dtARe, [1, N])));
    const ang = mul(k, reshape(d.dtAIm, [1, N]));
    const cRe = p('ssm.C_re');
    const cIm = p('ssm.C_im');
    const wRe = sub(mul(cRe, d.bbRe), mul(cIm, d.bbIm));
    const wIm = add(mul(cRe, d.bbIm), mul(cIm, d.bbRe));
    const re = sub(mul(cos(ang), reshape(wRe, [1, N])), mul(sin(ang), reshape(wIm, [1, N])));
    return mulScalar(sum(mul(mag, re), -1), 2);
  });
}

/** y = K ∗ u as one product with the causal Toeplitz matrix T[t, s] = K_{t−s} (0 for s > t). */
export function causalConv(K: Var, u: Var): Var {
  return scope('conv', () => {
    const L = K.shape[0];
    const table = concat([reshape(K, [L, 1]), constant(Tensor.zeros([1, 1]))], 0);
    const ids = new Array<number>(L * L);
    for (let t = 0; t < L; t++) for (let s = 0; s < L; s++) ids[t * L + s] = s <= t ? t - s : L;
    const toeplitz = reshape(take(table, ids, [L, L]), [L, L]);
    return linear(reshape(u, [u.shape[0], L]), toeplitz);
  });
}

/** x_t = Ā x_{t−1} + B̄ u_t,  y_t = 2 Re(C x_t), one step at a time. u: [B, L, 1]. */
export function recurrentScan(d: Discrete, u: Var): { y: Var; states: Var } {
  const [B, L] = u.shape;
  const N = d.abRe.shape[0];
  return scope('scan', () => {
    let sRe: Var = constant(Tensor.zeros([B, N]));
    let sIm: Var = constant(Tensor.zeros([B, N]));
    const cRe = p('ssm.C_re');
    const cIm = p('ssm.C_im');
    const ys: Var[] = [];
    const re: Var[] = [];
    for (let t = 0; t < L; t++) {
      scope(`t${t}`, () => {
        const ut = select(u, 1, t);
        const nRe = add(sub(mul(d.abRe, sRe), mul(d.abIm, sIm)), mul(ut, d.bbRe));
        const nIm = add(add(mul(d.abRe, sIm), mul(d.abIm, sRe)), mul(ut, d.bbIm));
        sRe = nRe;
        sIm = nIm;
        ys.push(mulScalar(sum(sub(mul(cRe, sRe), mul(cIm, sIm)), -1), 2));
        re.push(sRe);
      });
    }
    const states = stack(re, 1);
    return { y: stack(ys, 1), states };
  });
}

export function forwardLinearSsm(batch: Batch, c: LinearSsmConfig, mode = c.mode) {
  const u = input('u', batch.inputs.u);
  const [B, L] = u.shape;
  const d = discretize();
  const K = ssmKernel(d, L);
  const outputs: Record<string, Var> = { kernel: K };
  let yK: Var;
  if (mode === 'convolution') {
    yK = causalConv(K, u);
  } else {
    const r = recurrentScan(d, u);
    yK = r.y;
    outputs.states = r.states;
  }
  const y = scope('output', () => add(reshape(yK, [B, L, 1]), mul(u, p('ssm.D'))));
  outputs.prediction = y;
  // The kernel the model applies, including the D term at lag 0, beside the true one.
  outputs.kernelEffective = scope('kernel', () => {
    const e0 = Tensor.zeros([L]);
    e0.data[0] = 1;
    return add(K, mul(constant(e0), p('ssm.D')));
  });
  const h = targetKernel({ ...c, length: L });
  outputs.targetKernel = constant(new Tensor([L], h));
  const target = batch.targets?.y;
  let loss: Var | undefined;
  let metrics: Record<string, number> | undefined;
  if (target) {
    outputs.target = constant(target);
    loss = scope('loss', () => mse(y, target));
    let num = 0;
    let den = 0;
    for (let k = 0; k < L; k++) {
      num += (outputs.kernelEffective.value.data[k] - h[k]) ** 2;
      den += h[k] ** 2;
    }
    metrics = { 'relative kernel error': Math.sqrt(num / den) };
  }
  return { outputs, loss, metrics };
}

const linearSsm = defineModel<LinearSsmConfig>({
  id: 'linear-ssm',
  family: 'ssm',
  name: 'Linear state space layer (S4D)',
  summary:
    'A continuous-time linear system with diagonal state matrix, discretised by zero-order hold. The same layer runs as a recurrence or as one long convolution with kernel K.',
  signature:
    'x_t = \\bar A x_{t-1} + \\bar B u_t,\\; y_t = C x_t + D u_t \\;\\Longleftrightarrow\\; y = K \\ast u + D u,\\; K_k = C \\bar A^{k} \\bar B',
  fidelity: 'reduced',
  scaleNote:
    'One single-input single-output S4D layer with N complex modes (S4D-Lin initialisation, zero-order hold), trained alone on a filtering task. The published models stack many such layers with H channels each, nonlinearities, normalisation and position-wise mixing, and compute the kernel with FFT convolution.',
  references: [
    {
      authors: 'A. Gu, K. Goel, C. Ré',
      title: 'Efficiently modeling long sequences with structured state spaces',
      year: 2022,
      url: 'https://arxiv.org/abs/2111.00396',
    },
    {
      authors: 'A. Gu, A. Gupta, K. Goel, C. Ré',
      title: 'On the parameterization and initialization of diagonal state space models',
      year: 2022,
      url: 'https://arxiv.org/abs/2206.11893',
    },
    {
      authors: 'A. Gu, I. Johnson, K. Goel, K. Saab, T. Dao, A. Rudra, C. Ré',
      title:
        'Combining recurrent, convolutional, and continuous-time models with linear state-space layers',
      year: 2021,
      url: 'https://arxiv.org/abs/2110.13985',
    },
  ],
  limitations: [
    'Diagonal A only (S4D). The DPLR (normal plus low-rank) HiPPO matrix of S4 is not implemented.',
    'The convolution is a dense causal Toeplitz product, O(L²), not the FFT convolution used in practice; the recurrence is a sequential loop.',
    'A single linear layer without nonlinearity, so it can only represent linear time-invariant filters. Its kernel is a sum of N damped complex exponentials, so a pure delay is approximated, not matched.',
    'B is trainable here; the minimal S4D code folds it into C.',
  ],
  guide: {
    what: 'A single linear layer from an input sequence to an output sequence. Its hidden state is N = 8 complex numbers; at every step each one is multiplied by a fixed factor, so it decays and rotates, and receives the input. The output is a weighted sum of the state. Unrolled, this is a convolution with a kernel K: the layer is a learned filter. The task is to match a known filter by learning the decay and rotation rates, the step size Δ and the input and output weights.',
    watch:
      'The kernel view starts as a random wiggle and settles onto the damped sine within about 100 steps. The loss, the mean squared error of the output, falls from a few units to about 0.01 in 100 steps, and the “relative kernel error” metric from above 1 to below 0.1.',
    try: [
      {
        label: 'One mode only',
        note: 'The resonator is itself one damped sinusoid, so a single complex mode can match it exactly: kernel error about 0.01 by 150 steps.',
        patch: { modes: 1 },
        train: 300,
      },
      {
        label: 'Pure delay',
        note: 'A spike at lag 12 is not a sum of damped exponentials: the kernel error stalls near 0.4 and ripples remain around the spike.',
        patch: { filter: 'delay' },
        train: 300,
      },
      {
        label: 'Recurrence (scan)',
        note: 'Same output computed step by step; the state view now shows the real part of each mode as it decays and oscillates.',
        patch: { mode: 'recurrent' },
      },
    ],
  },
  configSchema: [
    {
      key: 'filter',
      label: 'Target filter',
      kind: 'select',
      options: FILTER_OPTIONS,
      affects: 'data',
    },
    {
      key: 'lag',
      label: 'Filter scale (steps)',
      kind: 'int',
      min: 1,
      max: 60,
      affects: 'data',
      help: 'Resonator: period and decay time. Delay: the lag d. Moving average: the window length.',
    },
    { key: 'length', label: 'Sequence length L', kind: 'int', min: 4, max: 128, affects: 'data' },
    {
      key: 'modes',
      label: 'Complex modes N',
      kind: 'int',
      min: 1,
      max: 32,
      affects: 'architecture',
    },
    {
      key: 'dtInit',
      label: 'Initial step Δ',
      kind: 'float',
      min: 0.001,
      max: 1,
      log: true,
      affects: 'architecture',
      help: 'S4D samples Δ log-uniformly in [0.001, 0.1] per channel; with one channel a single value is used.',
    },
    {
      key: 'mode',
      label: 'Computation',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'recurrent', label: 'Recurrence (scan)' },
        { value: 'convolution', label: 'Convolution with K' },
      ],
      help: 'Both give the same output to rounding error; only the computation graph differs.',
    },
  ],
  defaults: {
    filter: 'resonator',
    lag: 12,
    length: 64,
    modes: 8,
    mode: 'convolution',
    dtInit: 0.1,
  },
  validate(c) {
    if (c.filter === 'delay' && c.lag >= c.length)
      return `A delay of ${c.lag} steps does not fit in a sequence of length ${c.length}.`;
    return null;
  },
  init(c, rng) {
    const b = new ParamBuilder(rng);
    const N = c.modes;
    b.add('ssm.log_dt', [1], { kind: 'const', value: Math.log(c.dtInit) });
    b.add('ssm.log_A_real', [N], { kind: 'const', value: Math.log(0.5) });
    b.store.add(
      'ssm.A_imag',
      new Tensor(
        [N],
        Array.from({ length: N }, (_, n) => Math.PI * n),
      ),
      'S4D-Lin: Im A_n = πn',
    );
    b.add('ssm.B_re', [N], { kind: 'ones' });
    b.add('ssm.B_im', [N], { kind: 'zeros' });
    b.add('ssm.C_re', [N], { kind: 'normal', std: Math.SQRT1_2 });
    b.add('ssm.C_im', [N], { kind: 'normal', std: Math.SQRT1_2 });
    b.add('ssm.D', [1], { kind: 'zeros' });
    return b.store;
  },
  datasets: [filterDataset<LinearSsmConfig>()],
  forward: (batch, c) => forwardLinearSsm(batch, c),
  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 3e-2 },
    batchSize: 16,
    steps: 300,
    validateEvery: 25,
  },
  views: [
    {
      kind: 'series',
      title: 'Kernel K_k (with D at lag 0) and the target filter h_k',
      sources: [
        { ref: 'out:kernelEffective', label: 'learned K' },
        { ref: 'out:targetKernel', label: 'target h' },
      ],
      xLabel: 'lag k',
    },
    {
      kind: 'series',
      title: 'Output y_t and target',
      sources: [
        { ref: 'out:prediction', label: 'y' },
        { ref: 'out:target', label: 'target' },
      ],
      xLabel: 't',
    },
    { kind: 'series', title: 'Input u_t', sources: [{ ref: 'in:u', label: 'u' }], xLabel: 't' },
    {
      kind: 'matrix',
      title: 'State Re x_t (rows: t, columns: modes)',
      source: 'out:states',
      unavailable:
        'Only when the computation is set to “Recurrence (scan)”; the convolution never forms the states.',
    },
  ],
  math: {
    discretize: {
      tex: '\\bar A = e^{\\Delta A},\\qquad \\bar B = (\\Delta A)^{-1}(e^{\\Delta A} - I)\\,\\Delta B = \\frac{e^{\\Delta A} - 1}{A}\\, B',
      text: 'Zero-order hold: the input is held constant over each step of length Δ. With diagonal A = −exp(a) + iθ, both maps act elementwise and are exact. Δ, Re A, Im A, B and C are all trained.',
    },
    kernel: {
      tex: 'K_k = C \\bar A^{k} \\bar B = 2\\,\\operatorname{Re}\\sum_{n=1}^{N} C_n\\, e^{k \\Delta A_n}\\, \\bar B_n',
      text: 'Each mode contributes a damped complex exponential. The factor 2 Re accounts for the conjugate partner of each mode, so the state is effectively 2N real numbers.',
    },
    conv: {
      tex: 'y_t = \\sum_{k=0}^{t} K_k\\, u_{t-k}',
      text: 'Implemented as one product with the lower-triangular Toeplitz matrix of K.',
    },
    scan: {
      tex: 'x_t = \\bar A \\odot x_{t-1} + \\bar B\\, u_t,\\qquad y_t = 2\\operatorname{Re}(C x_t)',
      text: 'The same map computed step by step; unrolling the recurrence gives the convolution with K.',
    },
    output: { tex: 'y_t \\leftarrow y_t + D\\, u_t' },
    loss: { tex: 'L = \\tfrac{1}{NL}\\sum_{n,t} (y_{n,t} - (h \\ast u_n)_t)^2' },
  },
});

export default linearSsm;
