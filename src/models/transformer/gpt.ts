import {
  ParamBuilder,
  Tensor,
  add,
  crossEntropy,
  declareLinear,
  declareNorm,
  defaultOptimizer,
  dropout,
  linear,
  p,
  reshape,
  scope,
  slice,
  softmax,
  softmaxT,
  take,
  type Var,
} from '../../engine';
import { defineModel, type ForwardResult } from '../types';
import { VOCAB_CHARS, decodeText, encodeText, shakespeareDataset } from '../data/text';
import {
  ATTENTION_TEX,
  LN_TEX,
  causalMask,
  declareEncoderBlock,
  encoderBlock,
  ffnTex,
  headsProblem,
  nestedMath,
  norm,
  widthFields,
} from './attention';

export type GptConfig = {
  layers: number;
  dModel: number;
  heads: number;
  context: number;
  tie: boolean;
  mask: 'causal' | 'none';
  dropout: number;
};

const V = VOCAB_CHARS.length;
const MAX_LAYERS = 6;

export interface GptOutputs {
  logits: Var;
  hidden: Var;
  h0: Var;
  attn: Var[];
}

/** Token + learned position embeddings, N pre-LN causal blocks, final LN, (tied) unembedding. */
export function gptForward(
  ids: ArrayLike<number>,
  B: number,
  T: number,
  c: GptConfig,
  lastOnly = false,
): GptOutputs {
  const h0 = scope('embed', () => {
    const tok = take(p('embed.tok'), ids, [B, T]);
    const pos = slice(p('embed.pos'), 0, 0, T);
    return dropout(add(tok, pos), c.dropout);
  });
  const mask = c.mask === 'causal' ? causalMask(T) : undefined;
  let h = h0;
  const attn: Var[] = [];
  for (let l = 1; l <= c.layers; l++) {
    const r = encoderBlock(h, `block${l}`, {
      heads: c.heads,
      norm: 'pre',
      act: 'gelu',
      dropout: c.dropout,
      mask,
    });
    h = r.y;
    attn.push(r.probs);
  }
  const hidden = scope('lnf', () => norm(h, 'lnf'));
  const logits = scope('head', () => {
    const x = lastOnly ? slice(hidden, 1, T - 1, T) : hidden;
    return linear(x, c.tie ? p('embed.tok') : p('head.weight'));
  });
  return { logits, hidden, h0, attn };
}

/** Prompts as typed: the two characters \\n stand for a line break. */
const PROMPTS = [
  'First Citizen:\\n',
  'ROMEO:\\n',
  'KING RICHARD III:\\n',
  'To be, or not',
  'O',
  '\\n',
];
const unescapePrompt = (s: string) => s.replace(/\\n/g, '\n');

function checkPrompt(s: string): string | null {
  const text = unescapePrompt(s);
  if (!text.length) return 'Type at least one character.';
  const bad = [...new Set([...text].filter((c) => !VOCAB_CHARS.includes(c)))];
  if (bad.length)
    return `Not in the 65-character vocabulary: ${bad.map((c) => JSON.stringify(c)).join(' ')}. Allowed: letters, space, the digit 3, ! $ & ' , - . : ; ? and \\n for a line break.`;
  return null;
}

const gpt = defineModel<GptConfig>({
  id: 'gpt',
  family: 'transformer',
  name: 'GPT (character level)',
  summary:
    'A decoder-only transformer trained to predict the next character of Shakespeare, with a causal mask so each position sees only its past.',
  signature:
    'p(x_{t+1}\\mid x_{\\le t}) = \\operatorname{softmax}\\big(\\operatorname{LN}(h^{(N)}_t)\\, W_e^{\\top}\\big),\\quad h^{(\\ell)} = \\operatorname{Block}_{\\text{causal}}(h^{(\\ell-1)})',
  fidelity: 'reduced',
  scaleNote:
    'Reduced from GPT-2 small (12 layers, d = 768, 12 heads, 1,024-token context, 50,257 BPE tokens, ~40 GB WebText) to 2 layers, d = 64, 4 heads, a 64-character context and a 65-character vocabulary, trained on ~270 KB of Shakespeare. Block structure (pre-LN, GELU, learned positions, final LayerNorm, tied unembedding) follows GPT-2.',
  references: [
    {
      authors: 'A. Radford, K. Narasimhan, T. Salimans, I. Sutskever',
      title: 'Improving language understanding by generative pre-training',
      year: 2018,
      url: 'https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf',
    },
    {
      authors: 'A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever',
      title: 'Language models are unsupervised multitask learners',
      year: 2019,
      url: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
    },
    {
      authors:
        'A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin',
      title: 'Attention is all you need',
      year: 2017,
      url: 'https://arxiv.org/abs/1706.03762',
    },
    {
      authors: 'A. Karpathy',
      title: 'char-rnn: tiny-shakespeare dataset',
      year: 2015,
      url: 'https://github.com/karpathy/char-rnn',
    },
  ],
  limitations: [
    'Characters, not byte-pair tokens: the model must learn spelling, so text is word-like rather than fluent at this size.',
    'Only ~300 KB of text is shipped. The checkpoint saw 1.6 million characters (about six passes over the training split) and reaches 1.71 nats (2.47 bits) per character on the held-out 10 %; samples have the layout of a play and many real short words, but few coherent phrases.',
    'GPT-2’s 1/√N scaling of residual projections at initialisation is omitted; weights start from N(0, 0.02²) as in GPT-1, except the position embeddings, which start from N(0, 0.01²) as in the GPT-2 code.',
    'Sampling recomputes the whole context for every new character (no key–value cache), which is exact but slower.',
    'With the mask set to “none” every position except the last in the window can read the character it is asked to predict; after a few hundred steps the loss falls towards zero (from the checkpoint it first jumps to about 3.7) without the model learning to generate. This setting exists to show what the mask prevents.',
  ],
  guide: {
    what: 'Text goes in one character at a time and, at every position, the model outputs a probability for each of the 65 characters that could come next. A position may attend only to itself and the characters before it (the causal mask), so the same network generates text by sampling a character, appending it and repeating. The weights are learned by making the true next character of Shakespeare more probable; at this size the model learns spelling and the layout of a play, not meaning.',
    watch:
      'This page opens with a checkpoint already trained on Shakespeare (validation loss 1.75 nats per character): press “Sample text” to see what it produces, about three quarters real words laid out as a play. Then press “Start from random weights” to watch training from scratch. The loss starts at ln 65 ≈ 4.2 nats, the cost of uniform guessing. Within ten steps the model predicts a space most of the time; by step 50 it also uses frequent letters such as e and t; by step 400 the loss is about 2.5 and samples are word-shaped strings with a few common short words (the, and, he). The next-character view sharpens accordingly. Training the checkpoint itself further here raises its loss, to about 2.0 after 50 steps and still 1.95 after 200: browser training uses a constant learning rate and batches of six windows.',
    try: [
      {
        label: 'Remove the causal mask',
        note: 'Each position now sees the character it must predict. The trained weights were never used that way, so the loss first jumps from 1.75 to 3.7; it then falls slowly for about 300 steps and quickly after that, to about 0.3 by step 500, as the model learns to copy the next character. Sampled text, where the next character does not exist yet, degenerates into one letter repeated.',
        patch: { mask: 'none' },
        train: 500,
      },
      {
        label: 'Eight-character context',
        note: 'Re-initialises (the position table changes shape). Each prediction sees only eight characters. After 200 steps the validation loss is about 2.8 nats, against 2.6 for the 64-character context trained the same way from random weights; at this stage both produce word-like fragments, so the difference shows in the loss more than in the samples.',
        patch: { context: 8 },
        train: 200,
      },
    ],
    views: ['Next-character distribution (teacher forcing)', 'Attention weights, last block'],
  },
  configSchema: [
    ...widthFields({
      dMin: 16,
      dMax: 128,
      dStep: 16,
      maxLayers: MAX_LAYERS,
      layerLabel: 'Blocks N',
    }),
    {
      key: 'context',
      label: 'Context length T',
      kind: 'int',
      min: 8,
      max: 128,
      step: 8,
      affects: 'architecture',
      help: 'Learned position embeddings exist for T positions.',
    },
    { key: 'tie', label: 'Tie input and output embeddings', kind: 'bool', affects: 'architecture' },
    {
      key: 'mask',
      label: 'Attention mask',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'causal', label: 'Causal (positions see only the past)' },
        { value: 'none', label: 'None (future visible; leaks the target)' },
      ],
    },
    {
      key: 'dropout',
      label: 'Dropout rate',
      kind: 'float',
      min: 0,
      max: 0.5,
      step: 0.05,
      affects: 'forward',
    },
  ],
  defaults: { layers: 2, dModel: 64, heads: 4, context: 64, tie: true, mask: 'causal', dropout: 0 },
  validate: (c) => headsProblem(c.dModel, c.heads),

  init(c, rng) {
    const b = new ParamBuilder(rng);
    const d = c.dModel;
    const std = { std: 0.02 };
    b.add('embed.tok', [V, d], { kind: 'normal', std: 0.02 });
    b.add('embed.pos', [c.context, d], { kind: 'normal', std: 0.01 });
    for (let l = 1; l <= c.layers; l++) declareEncoderBlock(b, `block${l}`, d, 4 * d, std);
    declareNorm(b, 'lnf', d);
    if (!c.tie) declareLinear(b, 'head', d, V, { std: 0.02, bias: false });
    return b.store;
  },

  datasets: [shakespeareDataset<GptConfig>((c) => c.context)],

  forward(batch, c): ForwardResult {
    const ids = batch.inputs.ids;
    const [B, T] = ids.shape;
    const r = gptForward(ids.data, B, T, c);
    const probs = softmax(r.logits, -1);
    let loss: Var | undefined;
    let metrics: Record<string, number> | undefined;
    const next = batch.targets?.next;
    if (next) {
      loss = scope('loss', () => crossEntropy(reshape(r.logits, [B * T, V]), next.data));
      const L = loss.value.data[0];
      metrics = { bitsPerChar: L / Math.LN2, accuracy: topOneAccuracy(probs.value, next.data) };
    }
    const outputs: Record<string, Var> = { h0: r.h0, logits: r.logits, probs };
    outputs.attn1 = r.attn[0];
    outputs.attnLast = r.attn[r.attn.length - 1];
    return { outputs, loss, metrics };
  },

  training: {
    support: 'full',
    note: 'The default checkpoint was trained offline by scripts/train/gpt-shakespeare.ts with this engine; training in the browser starts from it or from a fresh initialisation.',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 2e-3, clip: 'norm', clipValue: 1 },
    batchSize: 6,
    steps: 400,
    validateEvery: 50,
  },

  procedures: [
    {
      id: 'sample',
      label: 'Sample text',
      description:
        'Autoregressive sampling: feed the last T characters, divide the final logits by the temperature, keep the k most probable characters, renormalise and draw one. Repeat.',
      args: [
        {
          key: 'prompt',
          label: 'Prompt',
          kind: 'text',
          affects: 'forward',
          maxLength: 200,
          suggestions: PROMPTS,
          validate: checkPrompt,
          help: 'Any text in the training vocabulary; \\n is a line break.',
        },
        {
          key: 'length',
          label: 'Characters to generate',
          kind: 'int',
          min: 10,
          max: 600,
          step: 10,
          affects: 'forward',
        },
        {
          key: 'temperature',
          label: 'Temperature τ',
          kind: 'float',
          min: 0.05,
          max: 2,
          step: 0.05,
          affects: 'forward',
        },
        { key: 'topK', label: 'Top-k (0 = all)', kind: 'int', min: 0, max: V, affects: 'forward' },
      ],
      defaults: { prompt: PROMPTS[0], length: 240, temperature: 0.8, topK: 10 },
      run(ctx) {
        const c = ctx.config;
        const n = Number(ctx.args.length);
        const tau = Number(ctx.args.temperature);
        const k = Number(ctx.args.topK);
        const err = checkPrompt(String(ctx.args.prompt));
        if (err) throw new Error(err);
        const ids = encodeText(unescapePrompt(String(ctx.args.prompt)));
        const dist = new Float32Array(n * V);
        const contextTokens: string[] = [];
        for (let i = 0; i < n; i++) {
          if (ctx.shouldStop()) break;
          const window = ids.slice(-c.context);
          const r = gptForward(window, 1, window.length, c, true);
          const q = sampleDistribution(r.logits.value.data, tau, k);
          dist.set(q, i * V);
          contextTokens.push(VOCAB_CHARS[ids[ids.length - 1]]);
          ids.push(ctx.rng.categorical(q));
          if (i % 10 === 0) ctx.progress(i / n);
        }
        const steps = contextTokens.length;
        return {
          tensors: { dist: new Tensor([steps, V], dist.subarray(0, steps * V)) },
          text: decodeText(ids),
          meta: {
            tokens: contextTokens,
            vocab: VOCAB_CHARS,
            caption: `τ = ${tau}, top-k = ${k || 'all'}`,
          },
          scalars: { generated: steps },
        };
      },
    },
  ],

  views: [
    { kind: 'text', title: 'Sampled text', source: 'proc:sample:text' },
    {
      kind: 'tokens',
      title: 'Sampling distribution at each generated step',
      source: 'proc:sample:dist',
      topK: 8,
    },
    {
      kind: 'tokens',
      title: 'Next-character distribution (teacher forcing)',
      source: 'out:probs',
      topK: 8,
    },
    {
      kind: 'attention',
      title: 'Attention weights, first block',
      source: 'out:attn1',
      tokens: 'meta',
    },
    {
      kind: 'attention',
      title: 'Attention weights, last block',
      source: 'out:attnLast',
      tokens: 'meta',
    },
  ],

  math: {
    embed: {
      tex: 'h^{(0)}_t = W_e[x_t] + W_p[t]',
      text: 'Each character indexes a row of the token embedding; a learned position embedding is added.',
    },
    block: {
      tex: "a = h + \\operatorname{MHA}(\\operatorname{LN}_1 h),\\quad h' = a + \\operatorname{FFN}(\\operatorname{LN}_2 a)",
      text: 'Pre-LN residual block (GPT-2). Normalisation is applied to the input of each sublayer, and the residual path is left untouched.',
    },
    ...nestedMath('block', MAX_LAYERS, {
      attn: {
        tex: ATTENTION_TEX,
        text: 'M is the causal mask: M_{tk} = −∞ for k > t, so position t reads only positions ≤ t.',
      },
      ffn: { tex: ffnTex('gelu') },
    }),
    lnf: { tex: LN_TEX, text: 'Final LayerNorm after the last block (GPT-2).' },
    head: {
      tex: 'z_t = \\operatorname{LN}(h^{(N)}_t)\\, W_e^{\\top}',
      text: 'With tied embeddings the output projection is the transpose of the token embedding; otherwise it is a separate matrix.',
    },
    loss: {
      tex: 'L = -\\frac{1}{BT}\\sum_{b,t} \\log p(x_{b,t+1}\\mid x_{b,\\le t})',
      text: 'Divide by ln 2 for bits per character.',
    },
  },

  checkpoints: [
    {
      id: 'shakespeare',
      label: 'Trained on Shakespeare',
      path: 'weights/gpt-shakespeare',
      description:
        'Default configuration trained offline on the first 90 % of the shipped text (provenance and validation loss in the file).',
    },
  ],
});

/** Temperature and top-k applied to one row of logits; returns the sampling distribution. */
export function sampleDistribution(
  logits: ArrayLike<number>,
  temperature: number,
  topK: number,
): Float32Array {
  const n = logits.length;
  const z = new Tensor(
    [n],
    Float32Array.from(logits, (v) => v / temperature),
  );
  if (topK > 0 && topK < n) {
    const sorted = Array.from(z.data).sort((a, b) => b - a);
    const cut = sorted[topK - 1];
    let kept = 0;
    for (let i = 0; i < n; i++) {
      if (z.data[i] < cut || kept >= topK) z.data[i] = -Infinity;
      else kept++;
    }
  }
  return Float32Array.from(softmaxT(z).data);
}

function topOneAccuracy(probs: Tensor, targets: ArrayLike<number>): number {
  const rows = targets.length;
  let k = 0;
  for (let r = 0; r < rows; r++) {
    let best = 0;
    for (let v = 1; v < V; v++) if (probs.data[r * V + v] > probs.data[r * V + best]) best = v;
    if (best === targets[r]) k++;
  }
  return k / rows;
}

export default gpt;
