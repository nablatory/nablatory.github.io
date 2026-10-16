import { Tensor, type Rng } from '../../engine';
import type { AssetLoader, Batch, Config, Dataset } from '../types';

/**
 * Character-level text from tiny-shakespeare (A. Karpathy, char-rnn), a
 * concatenation of passages from Shakespeare's plays. The text is in the
 * public domain; public/data/text/shakespeare.txt holds the first ~300 KB
 * (written by scripts/data/fetch-text.ts).
 */
export const SHAKESPEARE_LICENSE =
  'Shakespeare’s plays, public domain (tiny-shakespeare subset, A. Karpathy)';
export const SHAKESPEARE_PATH = 'data/text/shakespeare.txt';

/** The 65 characters of tiny-shakespeare, sorted by code point. Fixed so parameter shapes are known before loading. */
export const SHAKESPEARE_VOCAB =
  "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
export const VOCAB_CHARS: string[] = [...SHAKESPEARE_VOCAB];
const CHAR_ID = new Map(VOCAB_CHARS.map((c, i) => [c, i]));

export function encodeText(s: string): number[] {
  return [...s].map((c) => {
    const id = CHAR_ID.get(c);
    if (id === undefined)
      throw new Error(`character ${JSON.stringify(c)} is not in the vocabulary`);
    return id;
  });
}

export function decodeText(ids: ArrayLike<number>): string {
  let s = '';
  for (let i = 0; i < ids.length; i++) s += VOCAB_CHARS[ids[i]] ?? '?';
  return s;
}

export interface TextCorpus {
  ids: Uint8Array;
  /** First index of the validation split (last 10 % of the file). */
  split: number;
}

let cache: Promise<TextCorpus> | null = null;
let loaded: TextCorpus | null = null;

function corpusOf(text: string): TextCorpus {
  const ids = Uint8Array.from(encodeText(text));
  return { ids, split: Math.floor(ids.length * 0.9) };
}

export function loadShakespeare(loader: AssetLoader): Promise<TextCorpus> {
  cache ??= loader(SHAKESPEARE_PATH).then((b) => {
    loaded = corpusOf(new TextDecoder().decode(b));
    return loaded;
  });
  return cache;
}

/** Inject the text directly (offline training scripts). */
export function setShakespeare(text: string): TextCorpus {
  loaded = corpusOf(text);
  cache = Promise.resolve(loaded);
  return loaded;
}

export function shakespeare(): TextCorpus {
  if (!loaded) throw new Error('the Shakespeare text has not been loaded');
  return loaded;
}

/** Windows of `T + 1` characters starting at `starts`: inputs ids[s … s+T−1], targets ids[s+1 … s+T]. */
export function windowBatch(
  corpus: TextCorpus,
  starts: number[],
  T: number,
  caption?: string,
): Batch {
  const B = starts.length;
  const x = new Float32Array(B * T);
  const y = new Float32Array(B * T);
  starts.forEach((s, b) => {
    for (let t = 0; t < T; t++) {
      x[b * T + t] = corpus.ids[s + t];
      y[b * T + t] = corpus.ids[s + t + 1];
    }
  });
  const first = Array.from(x.subarray(0, T), (v) => VOCAB_CHARS[v]);
  return {
    inputs: { ids: new Tensor([B, T], x) },
    targets: { next: new Tensor([B, T], y) },
    meta: { tokens: first, vocab: VOCAB_CHARS, caption },
  };
}

/** Validation windows: `count` evenly spaced windows in the held-out last 10 %. */
export function validationStarts(corpus: TextCorpus, T: number, count: number): number[] {
  const span = corpus.ids.length - corpus.split - T - 1;
  return Array.from({ length: count }, (_, i) => corpus.split + Math.floor((i * span) / count));
}

/** Example windows begin at the start of a line in the validation split. */
function exampleStart(corpus: TextCorpus, index: number, T: number): number {
  const span = corpus.ids.length - corpus.split - T - 1;
  let s = corpus.split + Math.floor(((index + 0.5) * span) / 24);
  const nl = CHAR_ID.get('\n')!;
  for (let k = 0; k < 400 && s > corpus.split; k++, s--) if (corpus.ids[s - 1] === nl) break;
  return Math.min(s, corpus.ids.length - T - 1);
}

export function shakespeareDataset<C extends Config>(
  context: (c: C) => number,
  opts: { validationWindows?: number } = {},
): Dataset<C> {
  const valN = opts.validationWindows ?? 32;
  return {
    id: 'tiny-shakespeare',
    name: 'Tiny Shakespeare (characters)',
    description:
      'About 300 KB of Shakespeare’s plays as a stream of 65 characters. Training windows are drawn from the first 90 %; validation uses the last 10 %.',
    source: 'static',
    license: SHAKESPEARE_LICENSE,
    load: async (loader) => {
      await loadShakespeare(loader);
    },
    train(rng: Rng, batchSize: number, c: C) {
      const corpus = shakespeare();
      const T = context(c);
      const starts = Array.from({ length: batchSize }, () => rng.int(corpus.split - T - 1));
      return windowBatch(corpus, starts, T);
    },
    validation(c: C) {
      const corpus = shakespeare();
      const T = context(c);
      return windowBatch(corpus, validationStarts(corpus, T, valN), T, `${valN} held-out windows`);
    },
    exampleCount: 24,
    example(index: number, c: C) {
      const corpus = shakespeare();
      const T = context(c);
      return windowBatch(corpus, [exampleStart(corpus, index, T)], T, `held-out window ${index}`);
    },
    exampleLabel(index: number, c: C) {
      if (!loaded) return `Window ${index}`;
      const s = exampleStart(loaded, index, context(c));
      const text = decodeText(loaded.ids.subarray(s, s + 28)).replace(/\n/g, ' ⏎ ');
      return `“${text.trim()}…”`;
    },
  };
}
