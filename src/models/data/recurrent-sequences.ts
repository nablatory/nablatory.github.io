import { Rng, Tensor } from '../../engine';
import type { Batch, Config, Dataset } from '../types';

/**
 * Synthetic sequence tasks for the recurrent family.
 *
 * recall – delayed recall (after Hochreiter & Schmidhuber 1997, §5): the first
 *          token is one of four symbols, followed by T − 2 random distractor
 *          tokens and a query token. The answer is the first symbol, read out
 *          at the query step. Gradient information has to travel T − 1 steps.
 * sine   – next-step prediction of a sine wave with random frequency and
 *          phase: input s_t, target s_{t+1}. Short-range and smooth.
 */
export type RecurrentTask = 'recall' | 'sine';

export const RECALL_SYMBOLS = ['A', 'B', 'C', 'D'];
const DISTRACTORS = ['w', 'x', 'y', 'z'];
export const RECALL_VOCAB = [...RECALL_SYMBOLS, ...DISTRACTORS, '?'];
const QUERY = RECALL_VOCAB.length - 1;

export const RECURRENT_TASKS: { value: RecurrentTask; label: string }[] = [
  { value: 'recall', label: 'Delayed recall' },
  { value: 'sine', label: 'Sine continuation' },
];

export function taskInputSize(task: RecurrentTask): number {
  return task === 'recall' ? RECALL_VOCAB.length : 1;
}

export function taskOutputSize(task: RecurrentTask): number {
  return task === 'recall' ? RECALL_SYMBOLS.length : 1;
}

type TaskConfig = Config & { task: RecurrentTask; length: number };

function recallBatch(rng: Rng, B: number, T: number): Batch {
  const V = RECALL_VOCAB.length;
  const x = Tensor.zeros([B, T, V]);
  const labels: number[] = [];
  const first: number[] = [];
  for (let b = 0; b < B; b++) {
    const s = rng.int(RECALL_SYMBOLS.length);
    labels.push(s);
    for (let t = 0; t < T; t++) {
      const tok =
        t === 0 ? s : t === T - 1 ? QUERY : RECALL_SYMBOLS.length + rng.int(DISTRACTORS.length);
      x.data[(b * T + t) * V + tok] = 1;
      if (b === 0) first.push(tok);
    }
  }
  const tokens = first.map((k) => RECALL_VOCAB[k]);
  return {
    inputs: { x },
    labels,
    meta: {
      tokens,
      vocab: RECALL_VOCAB,
      classNames: RECALL_SYMBOLS,
      caption: `Remember “${RECALL_SYMBOLS[labels[0]]}” across ${T - 2} distractors, answer at “?”.`,
    },
  };
}

function sineBatch(rng: Rng, B: number, T: number): Batch {
  const x = Tensor.zeros([B, T, 1]);
  const y = Tensor.zeros([B, T, 1]);
  let caption = '';
  for (let b = 0; b < B; b++) {
    const w = rng.range(0.15, 0.6);
    const phi = rng.range(0, 2 * Math.PI);
    for (let t = 0; t < T; t++) {
      x.data[b * T + t] = Math.sin(w * t + phi);
      y.data[b * T + t] = Math.sin(w * (t + 1) + phi);
    }
    if (b === 0)
      caption = `s_t = sin(${w.toFixed(3)} t + ${phi.toFixed(2)}); predict s_{t+1} from s_0 … s_t.`;
  }
  return { inputs: { x }, targets: { y }, meta: { caption } };
}

export function recurrentBatch(task: RecurrentTask, rng: Rng, B: number, T: number): Batch {
  return task === 'recall' ? recallBatch(rng, B, T) : sineBatch(rng, B, T);
}

export function recurrentDataset<C extends TaskConfig>(): Dataset<C> {
  return {
    id: 'recurrent-sequences',
    name: 'Synthetic sequences',
    description:
      'Delayed recall: remember the first of T tokens until a query at the last step. Sine continuation: predict the next sample of a sine wave of random frequency and phase.',
    source: 'synthetic',
    train: (rng, B, c) => recurrentBatch(c.task, rng, B, c.length),
    validation: (c) => recurrentBatch(c.task, new Rng(9001), 256, c.length),
    exampleCount: 8,
    example: (i, c) => recurrentBatch(c.task, new Rng(500 + i), 1, c.length),
    exampleLabel: (i, c) => {
      const b = recurrentBatch(c.task, new Rng(500 + i), 1, c.length);
      return c.task === 'recall' ? `Recall “${RECALL_SYMBOLS[b.labels![0]]}”` : `Sine ${i + 1}`;
    },
  };
}
