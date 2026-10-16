import {
  ParamBuilder,
  Tensor,
  add,
  crossEntropy,
  defaultOptimizer,
  exp,
  linear,
  mean,
  mul,
  neg,
  p,
  pad,
  reshape,
  rmsNorm,
  scope,
  select,
  silu,
  slice,
  softmax,
  softplus,
  sum,
  take,
  type Var,
} from '../../engine';
import { defineModel, type Batch } from '../types';
import { COPY_VOCAB, copyDataset } from '../data/ssm-sequences';
import { constant } from '../util';
import { selectiveScan } from './ops';

export type MambaConfig = {
  span: number;
  memorize: number;
  dModel: number;
  layers: number;
  dState: number;
  selective: boolean;
};

const EXPAND = 2;
const CONV = 4;
const dtRank = (c: MambaConfig) => Math.ceil(c.dModel / 16);

/** Causal depthwise convolution along time: y_t[e] = b[e] + Σ_j w[e, j] x_{t−K+1+j}[e]. */
function causalDepthwiseConv(x: Var, name: string): Var {
  const [, L, E] = x.shape;
  const xp = pad(x, [
    [0, 0],
    [CONV - 1, 0],
    [0, 0],
  ]);
  const w = p(`${name}.weight`);
  let acc: Var = reshape(p(`${name}.bias`), [1, 1, E]);
  for (let j = 0; j < CONV; j++) acc = add(acc, mul(slice(xp, 1, j, j + L), select(w, 1, j)));
  return acc;
}

interface BlockTrace {
  delta: Var;
  deltaMean: Var;
  /** States of channel 0, [B, L, N]. */
  state0: Var;
}

function mambaBlock(x: Var, l: number, c: MambaConfig): { out: Var; trace: BlockTrace } {
  const name = `block${l}`;
  const [B, L] = x.shape;
  const E = EXPAND * c.dModel;
  const N = c.dState;
  const R = dtRank(c);
  return scope(name, () => {
    const h = scope('norm', () => rmsNorm(x, p(`${name}.norm.weight`)));
    const xz = scope('in_proj', () => linear(h, p(`${name}.in_proj.weight`)));
    const xs = slice(xz, 2, 0, E);
    const z = slice(xz, 2, E, 2 * E);
    const xc = scope('conv', () => silu(causalDepthwiseConv(xs, `${name}.conv`)));
    let delta: Var;
    let Bm: Var;
    let Cm: Var;
    if (c.selective) {
      const dbl = scope('x_proj', () => linear(xc, p(`${name}.x_proj.weight`)));
      Bm = slice(dbl, 2, R, R + N);
      Cm = slice(dbl, 2, R + N, R + 2 * N);
      delta = scope('dt_proj', () =>
        softplus(
          linear(slice(dbl, 2, 0, R), p(`${name}.dt_proj.weight`), p(`${name}.dt_proj.bias`)),
        ),
      );
    } else {
      // Selection removed: Δ, B and C are the same for every token (a time-invariant SSM).
      delta = scope('dt_proj', () =>
        add(constant(Tensor.zeros([B, L, E])), softplus(p(`${name}.dt_proj.bias`))),
      );
      Bm = add(constant(Tensor.zeros([B, L, N])), p(`${name}.B`));
      Cm = add(constant(Tensor.zeros([B, L, N])), p(`${name}.C`));
    }
    const { y, states } = scope('scan', () => {
      const A = neg(exp(p(`${name}.A_log`)));
      const states = selectiveScan(delta, A, Bm, xc);
      const y = add(sum(mul(states, reshape(Cm, [B, L, 1, N])), -1), mul(xc, p(`${name}.D`)));
      return { y, states };
    });
    const gated = scope('gate', () => mul(y, silu(z)));
    const out = scope('out_proj', () => linear(gated, p(`${name}.out_proj.weight`)));
    // Display tensors (not used downstream), taken before the residual sum so it stays the block output.
    const deltaMean = mean(delta, -1, true);
    const state0 = reshape(slice(states, 2, 0, 1), [B, L, N]);
    return { out: add(x, out), trace: { delta, deltaMean, state0 } };
  });
}

export function forwardMamba(batch: Batch, c: MambaConfig) {
  const ids = batch.inputs.ids;
  const [B, L] = ids.shape;
  let x = scope('embed', () => take(p('embed.weight'), ids.data, [B, L]));
  let first: BlockTrace | undefined;
  for (let l = 1; l <= c.layers; l++) {
    const r = mambaBlock(x, l, c);
    x = r.out;
    first ??= r.trace;
  }
  const hf = scope('norm_f', () => rmsNorm(x, p('norm_f.weight')));
  const logits = scope('lm_head', () => linear(hf, p('embed.weight')));
  const probs = scope('lm_head', () => softmax(logits, -1));
  const outputs: Record<string, Var> = {
    logits,
    probs,
    delta: first!.delta,
    deltaMean: first!.deltaMean,
    state: first!.state0,
  };
  let loss: Var | undefined;
  let metrics: Record<string, number> | undefined;
  if (batch.labels) {
    const m = c.memorize;
    const V = COPY_VOCAB.length;
    const q = scope('loss', () => reshape(slice(logits, 1, L - m, L), [B * m, V]));
    loss = scope('loss', () => crossEntropy(q, batch.labels!));
    let hit = 0;
    let all = 0;
    for (let b = 0; b < B; b++) {
      let ok = true;
      for (let j = 0; j < m; j++) {
        const row = q.value.data.subarray((b * m + j) * V, (b * m + j + 1) * V);
        let arg = 0;
        for (let v = 1; v < V; v++) if (row[v] > row[arg]) arg = v;
        if (arg === batch.labels[b * m + j]) hit++;
        else ok = false;
      }
      if (ok) all++;
    }
    metrics = { 'token accuracy': hit / (B * m), 'sequence accuracy': all / B };
  }
  return { outputs, loss, metrics };
}

const mamba = defineModel<MambaConfig>({
  id: 'mamba',
  family: 'ssm',
  name: 'Mamba (selective state space model)',
  summary:
    'A diagonal state space model whose step size Δ and matrices B and C are computed from each token, so the recurrence can choose what to store and what to ignore.',
  signature:
    'h_t = e^{\\Delta_t A} h_{t-1} + \\Delta_t B_t x_t,\\quad y_t = C_t h_t + D x_t,\\quad (\\Delta_t, B_t, C_t) = f(x_t)',
  fidelity: 'reduced',
  scaleNote:
    'Mamba blocks as in the reference implementation (RMSNorm, in-projection to 2·d, causal depthwise convolution of width 4, SiLU, input-dependent Δ, B, C, A = −exp(A_log), D skip, SiLU gate, out-projection, residual, tied embedding) at d_model = 16, expansion 2, state size N = 8, Δ rank ⌈d/16⌉ = 1 and 2 layers. The paper’s language models use d_model 768–2560, N = 16 and 24–64 layers.',
  references: [
    {
      authors: 'A. Gu, T. Dao',
      title: 'Mamba: linear-time sequence modeling with selective state spaces',
      year: 2023,
      url: 'https://arxiv.org/abs/2312.00752',
    },
    {
      authors: 'A. Gu, A. Gupta, K. Goel, C. Ré',
      title: 'On the parameterization and initialization of diagonal state space models',
      year: 2022,
      url: 'https://arxiv.org/abs/2206.11893',
    },
  ],
  limitations: [
    'The selective scan is a sequential loop in JavaScript. The hardware-aware parallel scan with kernel fusion and recomputation, which is what makes Mamba fast on GPUs, is not reproduced; this changes speed, not the result.',
    'One synthetic task (selective copying) with sequences of tens of tokens instead of thousands.',
    'B̄ = Δ B (the Euler simplification used in the reference code) rather than the full zero-order hold for B; Ā uses zero-order hold.',
    'The rescaling of out-projection weights by 1/√layers at initialisation follows the reference code; weight decay and other training details of the paper are not used.',
  ],
  guide: {
    what: 'A sequence model whose recurrence is adjusted by its input. Tokens enter as learned embedding vectors. In each block every channel runs its own small linear recurrence; the step size Δ and the read-in and read-out vectors B and C are computed from the current token, so a token can be stored (large Δ) or skipped (small Δ). A final layer predicts a token at every position. The task shows a span of blanks hiding a few letters, then one “?” per letter; the letters must be output in order.',
    watch:
      '“Output distribution at each position” is nearly flat at first; within 200–300 steps the “?” positions commit to the right letters while the others stay unconstrained. The loss, cross-entropy at the “?” positions, falls from about 1.8 to below 0.1 by 400 steps, sometimes after a plateau; token accuracy reaches about 0.98–0.99.',
    try: [
      {
        label: 'Selection off',
        note: 'Δ, B and C become constants, so the recurrence cannot favour data tokens: token accuracy after 400 steps about 0.8 instead of 0.99.',
        patch: { selective: false },
        train: 400,
      },
      {
        label: 'Double the span',
        note: 'The letters hide among 24 positions: learning is slower, with token accuracy about 0.75 after 400 steps.',
        patch: { span: 24 },
        train: 400,
      },
      {
        label: 'One layer',
        note: 'A single Mamba block is enough here: token accuracy about 0.98 after 400 steps.',
        patch: { layers: 1 },
        train: 400,
      },
    ],
  },
  configSchema: [
    {
      key: 'span',
      label: 'Span length (blanks and data tokens)',
      kind: 'int',
      min: 2,
      max: 48,
      affects: 'data',
    },
    { key: 'memorize', label: 'Tokens to copy', kind: 'int', min: 1, max: 8, affects: 'data' },
    {
      key: 'dModel',
      label: 'Model width d',
      kind: 'int',
      min: 4,
      max: 64,
      affects: 'architecture',
    },
    { key: 'layers', label: 'Layers', kind: 'int', min: 1, max: 4, affects: 'architecture' },
    { key: 'dState', label: 'State size N', kind: 'int', min: 1, max: 32, affects: 'architecture' },
    {
      key: 'selective',
      label: 'Selection (input-dependent Δ, B, C)',
      kind: 'bool',
      affects: 'architecture',
      help: 'Off: Δ, B and C become learned constants, giving a time-invariant SSM inside the same block (the S4-style ablation of the paper). At this small scale the ablation still learns the task partly, but more slowly.',
    },
  ],
  defaults: { span: 12, memorize: 3, dModel: 16, layers: 2, dState: 8, selective: true },
  validate: (c) =>
    c.memorize > c.span ? `${c.memorize} data tokens do not fit in a span of ${c.span}.` : null,
  init(c, rng) {
    const b = new ParamBuilder(rng);
    const D = c.dModel;
    const E = EXPAND * D;
    const N = c.dState;
    const R = dtRank(c);
    b.add('embed.weight', [COPY_VOCAB.length, D], { kind: 'normal', std: 0.02 });
    for (let l = 1; l <= c.layers; l++) {
      const n = `block${l}`;
      b.add(`${n}.norm.weight`, [D], { kind: 'ones' });
      b.add(`${n}.in_proj.weight`, [2 * E, D], { kind: 'lecunUniform', fanIn: D });
      b.add(`${n}.conv.weight`, [E, CONV], { kind: 'lecunUniform', fanIn: CONV });
      b.add(`${n}.conv.bias`, [E], { kind: 'lecunUniform', fanIn: CONV });
      if (c.selective) {
        b.add(`${n}.x_proj.weight`, [R + 2 * N, E], { kind: 'lecunUniform', fanIn: E });
        b.add(`${n}.dt_proj.weight`, [E, R], { kind: 'uniform', bound: 1 / Math.sqrt(R) });
      } else {
        b.add(`${n}.B`, [N], { kind: 'ones' });
        b.add(`${n}.C`, [N], { kind: 'normal', std: 1 / Math.sqrt(N) });
      }
      // Δ = softplus(bias) starts log-uniform in [0.001, 0.1]: bias = Δ + log(1 − e^{−Δ}).
      const r = rng.fork(`${n}.dt_proj.bias`);
      const dtBias = Tensor.zeros([E]);
      for (let e = 0; e < E; e++) {
        const dt = Math.exp(r.range(Math.log(1e-3), Math.log(1e-1)));
        dtBias.data[e] = dt + Math.log(-Math.expm1(-dt));
      }
      b.store.add(`${n}.dt_proj.bias`, dtBias, 'softplus⁻¹(Δ), Δ log-uniform in [0.001, 0.1]');
      const aLog = Tensor.zeros([E, N]);
      for (let e = 0; e < E; e++)
        for (let k = 0; k < N; k++) aLog.data[e * N + k] = Math.log(k + 1);
      b.store.add(`${n}.A_log`, aLog, 'S4D-Real: A_n = −(n + 1)');
      b.add(`${n}.D`, [E], { kind: 'ones' });
      b.add(`${n}.out_proj.weight`, [D, E], {
        kind: 'uniform',
        bound: 1 / Math.sqrt(E) / Math.sqrt(c.layers),
      });
    }
    b.add('norm_f.weight', [D], { kind: 'ones' });
    return b.store;
  },
  datasets: [copyDataset<MambaConfig>()],
  forward: (batch, c) => forwardMamba(batch, c),
  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 1e-2 },
    batchSize: 32,
    steps: 400,
    validateEvery: 25,
  },
  views: [
    {
      kind: 'tokens',
      title: 'Output distribution at each position',
      source: 'out:probs',
      note: 'At the j-th “?” the distribution is the model’s answer for the j-th data token of the span. Only those positions are trained; outputs at other positions are unconstrained.',
    },
    {
      kind: 'series',
      title: 'Mean step size Δ_t over channels, block 1',
      sources: [{ ref: 'out:deltaMean', label: 'mean Δ_t' }],
      xLabel: 'token position t',
    },
    {
      kind: 'matrix',
      title: 'Step size Δ_t, block 1 (rows: tokens, columns: channels)',
      source: 'out:delta',
    },
    {
      kind: 'series',
      title:
        'State h_t[0, n] of channel 0, block 1 (the dimension picker selects the state index n)',
      sources: [{ ref: 'out:state', label: 'h_t[0, n]' }],
      xLabel: 'token position t',
    },
  ],
  math: {
    embed: {
      tex: 'x_t = E[\\text{token}_t]',
      text: 'The output layer reuses the embedding matrix (tied weights).',
    },
    block: {
      tex: '\\begin{aligned} (u, z) &= W_{\\text{in}}\\, \\operatorname{RMSNorm}(x) \\\\ u &\\leftarrow \\operatorname{SiLU}(\\operatorname{conv}_{4}(u)) \\\\ \\Delta_t &= \\operatorname{softplus}(W_\\Delta W_{x\\Delta} u_t + b_\\Delta),\\; B_t = W_B u_t,\\; C_t = W_C u_t \\\\ h_t &= e^{\\Delta_t A} \\odot h_{t-1} + \\Delta_t B_t u_t,\\quad A = -e^{A_{\\log}} \\\\ y_t &= C_t h_t + D \\odot u_t \\\\ x &\\leftarrow x + W_{\\text{out}} (y \\odot \\operatorname{SiLU}(z)) \\end{aligned}',
      text: 'Each of the 2d channels runs its own N-dimensional diagonal SSM. A large Δ_t resets the state towards the current input; a small Δ_t keeps the state and ignores the token. Because Δ, B and C depend on the token, the model can skip blanks and store data tokens.',
    },
    norm_f: { tex: '\\hat x = \\operatorname{RMSNorm}(x)' },
    lm_head: { tex: 'p(\\cdot \\mid x_{\\le t}) = \\operatorname{softmax}(E\\, \\hat x_t)' },
    loss: {
      tex: 'L = -\\frac{1}{Nm} \\sum_{n} \\sum_{j=1}^{m} \\log p(s_{n,j} \\mid \\text{query } j)',
      text: 'Only the m query positions are scored.',
    },
  },
});

export default mamba;
