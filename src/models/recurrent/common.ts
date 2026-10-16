import {
  Graph,
  ParamBuilder,
  Tensor,
  backward,
  crossEntropy,
  linear,
  mse,
  p,
  scope,
  select,
  slice,
  softmax,
  stack,
  type Rng,
  type Var,
} from '../../engine';
import type {
  Batch,
  Config,
  ConfigField,
  ForwardResult,
  MathEntry,
  Procedure,
  ViewSpec,
} from '../types';
import {
  RECALL_SYMBOLS,
  RECURRENT_TASKS,
  taskInputSize,
  taskOutputSize,
  type RecurrentTask,
} from '../data/recurrent-sequences';
import { accuracy, constant, input } from '../util';

export type RecurrentBase = Config & { task: RecurrentTask; length: number; hidden: number };

/**
 * One recurrent cell, split into the two scopes that make up the diagram:
 *   gates – everything computed from x_t and the previous state,
 *   state – the state update itself.
 * Splitting the step this way makes the state module feed back into the
 * gates module, which the architecture canvas draws as a recurrence edge.
 */
export interface CellDef<C extends RecurrentBase> {
  /** Number of gate blocks stacked in W_ih / W_hh (PyTorch layout). */
  blocks: number;
  /** Names of the per-step gate tensors returned by `gates`. */
  gateNames: string[];
  /** Names of the state tensors; the first is the hidden state h. */
  stateNames: string[];
  /** xp = x_t W_ihᵀ + b_ih for this step; prev = previous state tensors. */
  gates(xp: Var, prev: Record<string, Var>, c: C): Record<string, Var>;
  update(g: Record<string, Var>, prev: Record<string, Var>, c: C): Record<string, Var>;
  /** Adjust the PyTorch-default biases (e.g. a forget-gate offset). */
  adjustBias?(bias: Tensor, H: number, c: C): string | null;
}

export const taskField: ConfigField = {
  key: 'task',
  label: 'Task',
  kind: 'select',
  affects: 'architecture',
  options: RECURRENT_TASKS,
  help: 'The two tasks need different input and output sizes, so switching re-initialises the parameters.',
};
export const lengthField: ConfigField = {
  key: 'length',
  label: 'Sequence length T',
  kind: 'int',
  min: 3,
  max: 100,
  affects: 'data',
  help: 'Delayed recall: the symbol at t = 0 must be reported at t = T − 1, after T − 2 distractors.',
};
export const hiddenField: ConfigField = {
  key: 'hidden',
  label: 'Hidden size H',
  kind: 'int',
  min: 1,
  max: 64,
  affects: 'architecture',
};

/** PyTorch nn.RNN/LSTM/GRU initialisation: every weight and bias ~ U(±1/√H). */
export function initRecurrent<C extends RecurrentBase>(
  cell: CellDef<C>,
  c: C,
  rng: Rng,
): ParamBuilder['store'] {
  const b = new ParamBuilder(rng);
  const H = c.hidden;
  const G = cell.blocks * H;
  const nIn = taskInputSize(c.task);
  b.add('cell.weight_ih', [G, nIn], { kind: 'lecunUniform', fanIn: H });
  b.add('cell.weight_hh', [G, H], { kind: 'lecunUniform', fanIn: H });
  b.add('cell.bias_hh', [G], { kind: 'lecunUniform', fanIn: H });
  if (cell.adjustBias) {
    const bound = 1 / Math.sqrt(H);
    const r = rng.fork('cell.bias_ih');
    const t = Tensor.zeros([G]);
    for (let i = 0; i < G; i++) t.data[i] = r.range(-bound, bound);
    const note = cell.adjustBias(t, H, c);
    b.store.add('cell.bias_ih', t, `PyTorch default, U(±1/√${H})${note ? `; ${note}` : ''}`);
  } else {
    b.add('cell.bias_ih', [G], { kind: 'lecunUniform', fanIn: H });
  }
  b.add('readout.weight', [taskOutputSize(c.task), H], { kind: 'lecunUniform', fanIn: H });
  b.add('readout.bias', [taskOutputSize(c.task)], { kind: 'lecunUniform', fanIn: H });
  return b.store;
}

export interface Unrolled {
  result: ForwardResult;
  /** Per-step state nodes (h_t, and c_t for the LSTM), for gradient inspection. */
  states: Record<string, Var[]>;
}

export function unroll<C extends RecurrentBase>(
  cell: CellDef<C>,
  batch: Batch,
  c: C,
  gradInput = true,
): Unrolled {
  const x = input('x', batch.inputs.x, gradInput);
  const [B, T] = x.shape;
  const H = c.hidden;
  // Input projections for every step at once: x_t W_ihᵀ + b_ih.
  const xp = scope('gates', () => linear(x, p('cell.weight_ih'), p('cell.bias_ih')));
  let prev: Record<string, Var> = {};
  for (const name of cell.stateNames) prev[name] = constant(Tensor.zeros([B, H]));
  const gateSeq: Record<string, Var[]> = Object.fromEntries(cell.gateNames.map((n) => [n, []]));
  const states: Record<string, Var[]> = Object.fromEntries(cell.stateNames.map((n) => [n, []]));
  for (let t = 0; t < T; t++) {
    const g = scope('gates', () => scope(`t${t}`, () => cell.gates(select(xp, 1, t), prev, c)));
    const next = scope('state', () => scope(`t${t}`, () => cell.update(g, prev, c)));
    for (const n of cell.gateNames) gateSeq[n].push(g[n]);
    for (const n of cell.stateNames) states[n].push(next[n]);
    prev = next;
  }
  const outputs: Record<string, Var> = {};
  scope('gates', () => {
    for (const n of cell.gateNames) outputs[n] = stack(gateSeq[n], 1);
  });
  scope('state', () => {
    for (const n of cell.stateNames) outputs[n] = stack(states[n], 1);
  });
  const y = scope('readout', () => linear(outputs.h, p('readout.weight'), p('readout.bias')));
  let loss: Var | undefined;
  let metrics: Record<string, number> | undefined;
  if (c.task === 'recall') {
    const K = RECALL_SYMBOLS.length;
    const last = scope('readout', () => select(y, 1, T - 1));
    outputs.logits = last;
    outputs.probs = scope('readout', () => softmax(last, -1));
    outputs.prediction = scope('readout', () => softmax(y, -1));
    if (batch.labels) {
      const onehot = Tensor.zeros([B, T, K]);
      for (let b = 0; b < B; b++)
        for (let t = 0; t < T; t++) onehot.data[(b * T + t) * K + batch.labels[b]] = 1;
      outputs.target = constant(onehot);
      loss = scope('loss', () => crossEntropy(last, batch.labels!));
      metrics = { accuracy: accuracy(outputs.probs.value, batch.labels) };
    }
  } else {
    outputs.prediction = y;
    const target = batch.targets?.y;
    if (target) {
      outputs.target = constant(target);
      loss = scope('loss', () => mse(y, target));
      // Error of the trivial predictor ŝ_{t+1} = s_t, for comparison.
      let s = 0;
      for (let i = 0; i < target.size; i++) s += (target.data[i] - batch.inputs.x.data[i]) ** 2;
      metrics = { 'copy-last MSE': s / target.size };
    }
  }
  return { result: { outputs, loss, metrics }, states };
}

/** Real ‖∂L/∂h_t‖ per step from one backward pass of the engine. */
export function gradientProcedure<C extends RecurrentBase>(cell: CellDef<C>): Procedure<C> {
  return {
    id: 'gradients',
    label: 'Gradient through time',
    description:
      'Runs a forward and backward pass on 64 fresh sequences and reports the mean per-sequence norm of ∂L/∂h_t at every step t. With the recall task the loss sits at the last step, so this shows how much of the error signal reaches early steps.',
    args: [{ key: 'batch', label: 'Sequences', kind: 'int', min: 1, max: 256, affects: 'data' }],
    defaults: { batch: 64 },
    run({ params, config, rng, args, dataset }) {
      const B = Number(args.batch) || 64;
      const batch = dataset.train(rng.fork('batch'), B, config);
      const g = new Graph({ params, record: false, grad: true, training: false, rng });
      const { result, states } = g.run(() => unroll(cell, batch, config, false));
      if (!result.loss) throw new Error('no loss for this batch');
      backward(result.loss);
      const tensors: Record<string, Tensor> = {};
      const scalars: Record<string, number> = { loss: result.loss.value.data[0] };
      for (const name of cell.stateNames) {
        const seq = states[name];
        const T = seq.length;
        const norm = new Float32Array(T);
        for (let t = 0; t < T; t++) {
          const gr = seq[t].grad;
          const H = seq[t].shape[1];
          let acc = 0;
          for (let b = 0; b < B; b++) {
            let s = 0;
            for (let j = 0; j < H; j++) s += (gr ? gr.data[b * H + j] : 0) ** 2;
            // L is a batch mean, so the per-sequence gradient is B · ∂L/∂h_t.
            acc += B * Math.sqrt(s);
          }
          norm[t] = acc / B;
        }
        tensors[`${name}Norm`] = new Tensor([T], norm);
        scalars[`${name}: first / last`] = norm[0] / Math.max(norm[T - 1], 1e-30);
      }
      return { tensors, scalars };
    },
  };
}

export const commonViews: ViewSpec[] = [
  {
    kind: 'series',
    title: 'Gradient through time, ‖∂L/∂h_t‖',
    sources: [{ ref: 'proc:gradients:hNorm', label: '‖∂L/∂h_t‖' }],
    xLabel: 't',
    logY: true,
    note: 'Mean over sequences of the norm of the per-sequence loss gradient with respect to h_t, from one backward pass. Logarithmic axis: a straight downward slope towards t = 0 means geometric decay.',
    unavailable: 'Run “Gradient through time” to compute it.',
  },
  { kind: 'matrix', title: 'Hidden states h_t (rows: t, columns: units)', source: 'out:h' },
  {
    kind: 'series',
    title: 'Readout at every step (dimension = class for recall)',
    sources: [
      { ref: 'out:prediction', label: 'prediction' },
      { ref: 'out:target', label: 'target' },
    ],
    xLabel: 't',
    note: 'Recall: softmax probability of the selected class at each step, against the one-hot answer; only the last step enters the loss. Sine: predicted and true s_{t+1}.',
  },
  {
    kind: 'probs',
    title: 'Answer at the query step',
    source: 'out:probs',
    classNames: RECALL_SYMBOLS,
    unavailable: 'Only for the delayed-recall task.',
  },
];

export const commonMath: Record<string, MathEntry> = {
  readout: {
    tex: 'y_t = W_{\\text{out}} h_t + b_{\\text{out}}',
    text: 'The readout is applied at every step. The recall loss uses only the last step (the query); the sine loss uses every step.',
  },
  loss: {
    tex: 'L_{\\text{recall}} = -\\tfrac1N \\textstyle\\sum_n \\log \\operatorname{softmax}(y_{T-1})_{s_n},\\quad L_{\\text{sine}} = \\tfrac{1}{NT}\\sum_{n,t} (y_t - s_{t+1})^2',
  },
};

/** Slice gate block k of size H from a [B, blocks·H] tensor. */
export function block(v: Var, k: number, H: number): Var {
  return slice(v, 1, k * H, (k + 1) * H);
}
