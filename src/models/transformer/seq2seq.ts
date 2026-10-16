import { Rng, Tensor } from '../../engine';
import type { Batch, Config, Dataset } from '../types';

/**
 * Deterministic sequence-to-sequence tasks over the digits 0–9.
 * reverse – y = (x_L, …, x_1)
 * sort    – y = x sorted in ascending order
 * The decoder reads the target shifted right by one with a start symbol
 * <s> (teacher forcing): (<s>, y_1, …, y_{L−1}) → (y_1, …, y_L).
 */
export type Seq2SeqTask = 'reverse' | 'sort';
export const BOS = 10;
export const SEQ_VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<s>'];
export const MAX_SEQ = 16;

export function targetOf(task: Seq2SeqTask, x: number[]): number[] {
  return task === 'reverse' ? [...x].reverse() : [...x].sort((a, b) => a - b);
}

export function seqBatch(task: Seq2SeqTask, sources: number[][], caption?: string): Batch {
  const B = sources.length;
  const L = sources[0].length;
  const src = new Float32Array(B * L);
  const dec = new Float32Array(B * L);
  const tgt = new Float32Array(B * L);
  sources.forEach((x, b) => {
    const y = targetOf(task, x);
    for (let t = 0; t < L; t++) {
      src[b * L + t] = x[t];
      tgt[b * L + t] = y[t];
      dec[b * L + t] = t === 0 ? BOS : y[t - 1];
    }
  });
  return {
    inputs: { src: new Tensor([B, L], src), dec: new Tensor([B, L], dec) },
    targets: { tgt: new Tensor([B, L], tgt) },
    // Labels for the first sequence's attention maps: encoder positions and decoder inputs.
    meta: {
      classNames: SEQ_VOCAB,
      caption,
      sourceTokens: Array.from(src.subarray(0, L), (v) => SEQ_VOCAB[v]),
      decoderTokens: Array.from(dec.subarray(0, L), (v) => SEQ_VOCAB[v]),
    },
  };
}

function randomDigits(rng: Rng, L: number): number[] {
  return Array.from({ length: L }, () => rng.int(10));
}

const exampleSource = (index: number, L: number) =>
  randomDigits(new Rng(77).fork(`example:${index}`), L);

export function seq2seqDataset<C extends Config>(
  task: (c: C) => Seq2SeqTask,
  length: (c: C) => number,
): Dataset<C> {
  return {
    id: 'digits',
    name: 'Digit sequences',
    description:
      'Uniformly random digit strings, generated in the browser; the target is the reversed or sorted string. 256 fixed validation strings.',
    source: 'synthetic',
    train(rng, batchSize, c) {
      const L = length(c);
      return seqBatch(
        task(c),
        Array.from({ length: batchSize }, () => randomDigits(rng, L)),
      );
    },
    validation(c) {
      const rng = new Rng(4242);
      const L = length(c);
      return seqBatch(
        task(c),
        Array.from({ length: 256 }, () => randomDigits(rng, L)),
        '256 validation strings',
      );
    },
    exampleCount: 24,
    example(index, c) {
      const x = exampleSource(index, length(c));
      return seqBatch(task(c), [x], `${x.join(' ')} → ${targetOf(task(c), x).join(' ')}`);
    },
    exampleLabel(index, c) {
      const x = exampleSource(index, length(c));
      return `${x.join('')} → ${targetOf(task(c), x).join('')}`;
    },
  };
}
