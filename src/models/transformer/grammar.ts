import { Rng, Tensor } from '../../engine';
import type { Batch, Config, Dataset } from '../types';

/**
 * A small synthetic language with number agreement, for masked language
 * modelling. Each example is a sentence pair
 *
 *   [CLS] Det_n (Adj) N_n (near the N_m | that V_n) V_n . [SEP] Pron_n Cop_n Adj . [SEP] [PAD]…
 *
 * where n ∈ {singular, plural} is shared by the whole example and m is
 * independent ("near the dogs" is an attractor). Determiners, verbs, the
 * pronoun and the copula are predictable from context; nouns, adjectives
 * and verb stems are not.
 */
export const SPECIAL = ['[PAD]', '[CLS]', '[SEP]', '[MASK]'] as const;
export const PAD = 0;
export const CLS = 1;
export const SEP = 2;
export const MASK = 3;

const NOUNS_SG = ['cat', 'dog', 'bird', 'fox', 'horse'];
const NOUNS_PL = ['cats', 'dogs', 'birds', 'foxes', 'horses'];
const VERBS_SG = ['runs', 'sleeps', 'sings', 'waits'];
const VERBS_PL = ['run', 'sleep', 'sing', 'wait'];
const ADJS = ['big', 'small', 'old', 'young', 'red'];

export const GRAMMAR_VOCAB: string[] = [
  ...SPECIAL,
  '.',
  'the',
  'a',
  'two',
  'that',
  'near',
  ...NOUNS_SG,
  ...NOUNS_PL,
  ...VERBS_SG,
  ...VERBS_PL,
  ...ADJS,
  'it',
  'they',
  'is',
  'are',
];
const ID = new Map(GRAMMAR_VOCAB.map((w, i) => [w, i]));
const id = (w: string) => ID.get(w)!;
/** Words that may replace a selected token in the 10 % "random token" case. */
export const WORD_IDS: number[] = GRAMMAR_VOCAB.map((_, i) => i).filter((i) => i >= SPECIAL.length);
export const GRAMMAR_T = 16;

export type ProbeKind = 'pronoun' | 'verb' | 'copula' | 'determiner';

export interface Sentence {
  ids: number[];
  segments: number[];
  /** Positions of the agreement-bearing words, for probes. */
  positions: Record<ProbeKind, number>;
}

export function sentence(rng: Rng): Sentence {
  const plural = rng.uniform() < 0.5;
  const pick = <T>(xs: T[]) => xs[rng.int(xs.length)];
  const nounIdx = rng.int(NOUNS_SG.length);
  const a: string[] = [];
  const det = plural ? pick(['the', 'two']) : pick(['the', 'a']);
  a.push(det);
  if (rng.uniform() < 0.5) a.push(pick(ADJS));
  a.push((plural ? NOUNS_PL : NOUNS_SG)[nounIdx]);
  const verbIdx = rng.int(VERBS_SG.length);
  const extra = rng.uniform();
  if (extra < 0.3) a.push('near', 'the', rng.uniform() < 0.5 ? pick(NOUNS_PL) : pick(NOUNS_SG));
  else if (extra < 0.6) a.push('that', (plural ? VERBS_PL : VERBS_SG)[rng.int(VERBS_SG.length)]);
  const verbPos = a.length;
  a.push((plural ? VERBS_PL : VERBS_SG)[verbIdx], '.');
  const bWords = [plural ? 'they' : 'it', plural ? 'are' : 'is', pick(ADJS), '.'];
  const ids = [CLS, ...a.map(id), SEP, ...bWords.map(id), SEP];
  const segments = [0, ...a.map(() => 0), 0, ...bWords.map(() => 1), 1];
  const bStart = a.length + 2;
  while (ids.length < GRAMMAR_T) {
    ids.push(PAD);
    segments.push(0);
  }
  return {
    ids,
    segments,
    positions: { determiner: 1, verb: verbPos + 1, pronoun: bStart, copula: bStart + 1 },
  };
}

export interface MlmCorruption {
  /** Model input after corruption. */
  input: number[];
  /** Original token at selected positions, −1 elsewhere. */
  target: number[];
  /** 0 = not selected, 1 = replaced by [MASK], 2 = replaced by a random word, 3 = kept unchanged. */
  kind: number[];
}

/**
 * BERT's masking rule (Devlin et al. 2019, §3.1): select each maskable
 * position with probability `rate`; replace a selected token by [MASK] with
 * probability 0.8, by a uniformly random word with probability 0.1, and keep
 * it with probability 0.1. A sequence with no selected position gets one,
 * chosen uniformly, so every sequence contributes to the loss.
 */
export function mlmCorrupt(
  ids: ArrayLike<number>,
  T: number,
  rng: Rng,
  rate = 0.15,
  maskable = (t: number) => t >= SPECIAL.length,
): MlmCorruption {
  const n = ids.length;
  const input = Array.from(ids);
  const target = new Array<number>(n).fill(-1);
  const kind = new Array<number>(n).fill(0);
  const corrupt = (i: number) => {
    target[i] = input[i];
    const u = rng.uniform();
    if (u < 0.8) {
      input[i] = MASK;
      kind[i] = 1;
    } else if (u < 0.9) {
      input[i] = WORD_IDS[rng.int(WORD_IDS.length)];
      kind[i] = 2;
    } else kind[i] = 3;
  };
  for (let s = 0; s < n; s += T) {
    const cand: number[] = [];
    let any = false;
    for (let i = s; i < s + T; i++) {
      if (!maskable(ids[i])) continue;
      cand.push(i);
      if (rng.uniform() < rate) {
        corrupt(i);
        any = true;
      }
    }
    if (!any && cand.length) corrupt(cand[rng.int(cand.length)]);
  }
  return { input, target, kind };
}

/** Batch of clean sentences; the forward pass corrupts them with graphRng(). */
function cleanBatch(sents: Sentence[]): Batch {
  const B = sents.length;
  const ids = new Float32Array(B * GRAMMAR_T);
  const seg = new Float32Array(B * GRAMMAR_T);
  sents.forEach((s, b) => {
    ids.set(s.ids, b * GRAMMAR_T);
    seg.set(s.segments, b * GRAMMAR_T);
  });
  return {
    inputs: { ids: new Tensor([B, GRAMMAR_T], ids), segments: new Tensor([B, GRAMMAR_T], seg) },
    meta: { vocab: GRAMMAR_VOCAB },
  };
}

/** Batch with a fixed corruption: `targets.mlm` holds the original token at selected positions and −1 elsewhere. */
export function corruptedBatch(sents: Sentence[], c: MlmCorruption, caption?: string): Batch {
  const b = cleanBatch(sents);
  b.inputs.ids = new Tensor(b.inputs.ids.shape, Float32Array.from(c.input));
  b.targets = { mlm: new Tensor(b.inputs.ids.shape, Float32Array.from(c.target)) };
  b.meta = {
    vocab: GRAMMAR_VOCAB,
    tokens: c.input.slice(0, GRAMMAR_T).map((t) => GRAMMAR_VOCAB[t]),
    caption,
  };
  return b;
}

const PROBES: ProbeKind[] = ['pronoun', 'verb', 'copula', 'determiner'];

function probe(index: number) {
  const s = sentence(new Rng(31).fork(`example:${index}`));
  const kind = PROBES[index % PROBES.length];
  const pos = s.positions[kind];
  const target = s.ids.map(() => -1);
  const input = [...s.ids];
  target[pos] = s.ids[pos];
  input[pos] = MASK;
  return { s, kind, pos, c: { input, target, kind: target.map((t) => (t >= 0 ? 1 : 0)) } };
}

export function grammarDataset<C extends Config>(maskRate: (c: C) => number): Dataset<C> {
  return {
    id: 'agreement',
    name: 'Agreement sentence pairs',
    description:
      'Generated sentence pairs such as “the cat near the dogs runs . [SEP] it is old .”, 37-symbol vocabulary, 16 positions with padding. Number (singular or plural) must agree across the determiner, noun, verbs, pronoun and copula.',
    source: 'synthetic',
    train(rng: Rng, batchSize: number) {
      return cleanBatch(Array.from({ length: batchSize }, () => sentence(rng)));
    },
    validation(c: C) {
      const rng = new Rng(2024);
      const sents = Array.from({ length: 256 }, () => sentence(rng));
      const flat = sents.flatMap((s) => s.ids);
      return corruptedBatch(
        sents,
        mlmCorrupt(flat, GRAMMAR_T, new Rng(2025), maskRate(c)),
        '256 validation pairs, fixed 80/10/10 corruption',
      );
    },
    exampleCount: 24,
    example(index: number) {
      const { s, kind, pos, c } = probe(index);
      return corruptedBatch([s], c, `probe: ${kind} at position ${pos}`);
    },
    exampleLabel(index: number) {
      const { s, kind, pos } = probe(index);
      const words = s.ids
        .filter((t) => t !== PAD)
        .map((t, i) => (i === pos ? '[MASK]' : GRAMMAR_VOCAB[t]));
      return `${kind}: ${words.slice(1, -1).join(' ')}`;
    },
  };
}
