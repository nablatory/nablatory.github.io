import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  Graph,
  Tensor,
  backward,
  withDType,
  type Var,
  linear,
  matmul,
  conv2d,
  convTranspose2d,
  maxPool2d,
  avgPool2d,
  upsampleNearest2d,
  layerNorm,
  rmsNorm,
  batchNorm,
  softmax,
  logSoftmax,
  logSumExp,
  sum,
  mean,
  max,
  transpose,
  concat,
  pad,
  mul,
  div,
  sub,
  activations,
  exp,
  log,
  sqrt,
  take,
  crossEntropy,
  mse,
  bceWithLogits,
  maskedFill,
  mulScalar,
  softplus,
  T,
  gruCell,
  lstmCell,
  Optimizer,
  defaultOptimizer,
  ParamStore,
  type ActivationName,
  type OptimizerKind,
} from '../src/engine';

interface Enc {
  shape: number[];
  data: number[];
}
interface Case {
  name: string;
  attrs: Record<string, unknown>;
  inputs: Record<string, Enc>;
  output: Enc;
  upstream: Enc;
  grads: Record<string, Enc>;
}
const fixtures = JSON.parse(
  readFileSync(new URL('./fixtures/torch.json', import.meta.url), 'utf8'),
) as {
  torch: string;
  cases: Case[];
  optimizers: {
    kind: OptimizerKind;
    theta0: number[];
    grads: number[][];
    trajectory: number[][];
  }[];
  clip: { grads: number[][]; max: number; clipped: number[][] };
};

const dec = (e: Enc) => new Tensor(e.shape, Float64Array.from(e.data));

type Builder = (x: Record<string, Var>, attrs: Record<string, unknown>) => Var;

const builders: Record<string, Builder> = {
  linear: (x) => linear(x.x, x.w, x.b),
  matmul_batched: (x) => matmul(x.a, x.b),
  matmul_shared: (x) => matmul(x.a, x.b),
  matmul_shared_left: (x) => matmul(x.a, x.b),
  conv2d: (x) => conv2d(x.x, x.w, x.b, { stride: 2, padding: 1 }),
  conv2d_dilated_grouped: (x) =>
    conv2d(x.x, x.w, x.b, { stride: 1, padding: 2, dilation: 2, groups: 2 }),
  conv_transpose2d: (x) =>
    convTranspose2d(x.x, x.w, x.b, { stride: 2, padding: 1, outputPadding: 1 }),
  max_pool2d: (x) => maxPool2d(x.x, 3, 2, 1),
  avg_pool2d: (x) => avgPool2d(x.x, 2),
  upsample_nearest: (x) => upsampleNearest2d(x.x, 2),
  layer_norm: (x) => layerNorm(x.x, x.w, x.b),
  rms_norm: (x) => rmsNorm(x.x, x.w),
  batch_norm_train: (x) =>
    batchNorm(x.x, x.w, x.b, { mean: Tensor.zeros([3]), var: Tensor.full([3], 1), momentum: 0.1 }),
  batch_norm_eval: (x, a) =>
    batchNorm(x.x, x.w, x.b, {
      mean: new Tensor([3], a.runningMean as number[]),
      var: new Tensor([3], a.runningVar as number[]),
      momentum: 0.1,
    }),
  softmax_axis1: (x) => softmax(x.x, 1),
  log_softmax: (x) => logSoftmax(x.x, -1),
  logsumexp: (x) => logSumExp(x.x, -1),
  sum_axes: (x) => sum(x.x, [0, 2]),
  mean_keepdim: (x) => mean(x.x, 1, true),
  max_axis: (x) => max(x.x, 1),
  transpose_perm: (x) => transpose(x.x, [2, 0, 1]),
  concat_axis1: (x) => concat([x.a, x.b], 1),
  pad: (x) =>
    pad(x.x, [
      [0, 0],
      [0, 1],
      [1, 2],
    ]),
  broadcast_mul: (x) => mul(x.a, x.b),
  broadcast_div: (x) => div(x.a, x.b),
  broadcast_sub: (x) => sub(x.a, x.b),
  log: (x) => log(x.x),
  sqrt: (x) => sqrt(x.x),
  gather: (x, a) => take(x.w, a.ids as number[]),
  cross_entropy: (x, a) => crossEntropy(x.z, a.targets as number[]),
  cross_entropy_smooth: (x, a) =>
    crossEntropy(x.z, a.targets as number[], a.labelSmoothing as number),
  mse: (x, a) => mse(x.p, dec(a.target as Enc)),
  bce_logits: (x, a) => bceWithLogits(x.z, dec(a.target as Enc)),
  causal_attention: (x) => {
    const mask = new Tensor(
      [4, 4],
      Float64Array.from({ length: 16 }, (_, k) => (k % 4 > Math.floor(k / 4) ? 1 : 0)),
    );
    const s = mulScalar(matmul(x.q, T(x.k)), 1 / Math.sqrt(3));
    return matmul(softmax(maskedFill(s, mask, -Infinity), -1), x.v);
  },
  gru_cell: (x) => gruCell(x.x, x.h, x.wih, x.whh, x.bih, x.bhh).h,
  lstm_cell: (x) => {
    const r = lstmCell(x.x, x.h, x.c, x.wih, x.whh, x.bih, x.bhh);
    return concat([r.h, r.c], -1);
  },
};
for (const act of ['tanh', 'sigmoid', 'gelu', 'silu', 'elu', 'leakyRelu', 'relu'] as const) {
  builders[`act_${act}`] = (x) => activations[act as ActivationName](x.x);
}
builders.act_exp = (x) => exp(x.x);
builders.act_softplus = (x) => softplus(x.x);

function expectClose(actual: ArrayLike<number>, expected: number[], tol: number, label: string) {
  expect(actual.length, `${label} length`).toBe(expected.length);
  let worst = 0;
  for (let i = 0; i < expected.length; i++) {
    const err = Math.abs(actual[i] - expected[i]) / (1 + Math.abs(expected[i]));
    worst = Math.max(worst, err);
  }
  expect(worst, `${label} max scaled error`).toBeLessThan(tol);
}

describe(`engine vs PyTorch ${fixtures.torch} (float64)`, () => {
  for (const c of fixtures.cases) {
    it(c.name, () => {
      const build = builders[c.name];
      expect(build, `builder for ${c.name}`).toBeDefined();
      withDType('float64', () => {
        const g = new Graph({ grad: true, record: true, training: c.name.endsWith('_train') });
        g.run(() => {
          const vars: Record<string, Var> = {};
          for (const [k, e] of Object.entries(c.inputs)) vars[k] = g.input(k, dec(e), true);
          const y = build(vars, c.attrs);
          expect(y.shape).toEqual(c.output.shape);
          const tol = 1e-10;
          expectClose(y.value.data, c.output.data, tol, `${c.name} forward`);
          backward(y, dec(c.upstream));
          for (const [k, e] of Object.entries(c.grads)) {
            expectClose(vars[k].grad!.data, e.data, tol, `${c.name} ∂/∂${k}`);
          }
        });
      });
    });
  }
});

describe('optimizers vs PyTorch', () => {
  const settings: Record<OptimizerKind, Partial<typeof defaultOptimizer>> = {
    sgd: { lr: 0.1, weightDecay: 0.01 },
    momentum: { lr: 0.1, momentum: 0.9, weightDecay: 0.01 },
    nesterov: { lr: 0.1, momentum: 0.9 },
    rmsprop: { lr: 0.01, beta2: 0.99, eps: 1e-8 },
    adam: { lr: 0.01, weightDecay: 0.01 },
    adamw: { lr: 0.01, weightDecay: 0.1 },
  };
  for (const o of fixtures.optimizers) {
    it(o.kind, () => {
      withDType('float64', () => {
        const store = new ParamStore();
        store.add('theta', new Tensor([5], Float64Array.from(o.theta0)), 'fixture');
        const opt = new Optimizer({
          ...defaultOptimizer,
          weightDecay: 0,
          ...settings[o.kind],
          kind: o.kind,
        });
        o.grads.forEach((g, i) => {
          opt.step(store, new Map([['theta', new Tensor([5], Float64Array.from(g))]]));
          expectClose(store.get('theta').data, o.trajectory[i], 1e-12, `${o.kind} step ${i + 1}`);
        });
      });
    });
  }

  it('global-norm clipping', () => {
    withDType('float64', () => {
      const store = new ParamStore();
      store.add('a', Tensor.zeros([3]), 'fixture');
      store.add('b', Tensor.zeros([2]), 'fixture');
      const opt = new Optimizer({
        ...defaultOptimizer,
        kind: 'sgd',
        lr: 1,
        clip: 'norm',
        clipValue: fixtures.clip.max,
      });
      opt.step(
        store,
        new Map([
          ['a', new Tensor([3], Float64Array.from(fixtures.clip.grads[0]))],
          ['b', new Tensor([2], Float64Array.from(fixtures.clip.grads[1]))],
        ]),
      );
      // θ = 0 − 1·clipped(g)
      expectClose(
        store.get('a').data,
        fixtures.clip.clipped[0].map((v) => -v),
        1e-12,
        'a',
      );
      expectClose(
        store.get('b').data,
        fixtures.clip.clipped[1].map((v) => -v),
        1e-12,
        'b',
      );
    });
  });
});
