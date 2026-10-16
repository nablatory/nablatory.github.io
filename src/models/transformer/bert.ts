import {
  ParamBuilder,
  add,
  crossEntropy,
  declareLinear,
  declareNorm,
  defaultOptimizer,
  dense,
  dropout,
  graphRng,
  linear,
  p,
  reshape,
  scope,
  slice,
  softmax,
  take,
  type Var,
} from '../../engine';
import { defineModel, type ForwardResult } from '../types';
import {
  ATTENTION_TEX,
  declareEncoderBlock,
  encoderBlock,
  ffnTex,
  headsProblem,
  keyPaddingMask,
  nestedMath,
  norm,
  widthFields,
} from './attention';
import { geluTanh } from './ops';
import { GRAMMAR_T, GRAMMAR_VOCAB, PAD, grammarDataset, mlmCorrupt } from './grammar';

type BertConfig = {
  layers: number;
  dModel: number;
  heads: number;
  segments: boolean;
  maskRate: number;
  dropout: number;
};

const V = GRAMMAR_VOCAB.length;
const MAX_LAYERS = 6;
/** Words fixed entirely by grammatical number: the pronoun and copula of the second sentence. */
const AGREEMENT = new Set(['it', 'they', 'is', 'are'].map((w) => GRAMMAR_VOCAB.indexOf(w)));

const bert = defineModel<BertConfig>({
  id: 'bert',
  family: 'transformer',
  name: 'BERT (masked language model)',
  summary:
    'A bidirectional transformer encoder pre-trained by selecting 15 % of the tokens, replacing most of them with [MASK], and predicting them from both sides, here on generated sentence pairs with number agreement.',
  signature:
    'L_{\\text{MLM}} = -\\textstyle\\sum_{i \\in \\mathcal{M}} \\log p\\big(x_i \\mid \\tilde{x}\\big),\\quad p(\\cdot \\mid \\tilde x) = \\operatorname{softmax}\\!\\big(g(h^{(N)}_i)\\, E^{\\top} + b\\big)',
  fidelity: 'reduced',
  scaleNote:
    'Reduced from BERT-base (12 layers, d = 768, 12 heads, d_ff = 3,072, 512 positions, 30,522 WordPiece tokens, BooksCorpus + Wikipedia) to 2 layers, d = 32, 4 heads, d_ff = 128, 16 positions and a 37-symbol synthetic vocabulary. Embeddings (token + learned position + segment, then LayerNorm), post-LN GELU blocks, the masking rule and the tied MLM output layer are as published.',
  references: [
    {
      authors: 'J. Devlin, M.-W. Chang, K. Lee, K. Toutanova',
      title: 'BERT: pre-training of deep bidirectional transformers for language understanding',
      year: 2019,
      url: 'https://arxiv.org/abs/1810.04805',
    },
    {
      authors: 'T. Linzen, E. Dupoux, Y. Goldberg',
      title: 'Assessing the ability of LSTMs to learn syntax-sensitive dependencies',
      year: 2016,
      url: 'https://arxiv.org/abs/1611.01368',
    },
    {
      authors: 'Y. Liu et al.',
      title: 'RoBERTa: a robustly optimized BERT pretraining approach',
      year: 2019,
      url: 'https://arxiv.org/abs/1907.11692',
    },
  ],
  limitations: [
    'Masked language modelling only: the next-sentence-prediction objective and its [CLS] classifier are omitted (RoBERTa showed it is not needed). Segment embeddings are kept.',
    'Masks are drawn afresh for every training batch (dynamic masking, as in RoBERTa); BERT duplicated its data ten times with static masks.',
    'A sequence in which no token is selected gets one selected position, so the effective selection rate is slightly above the configured rate on these short sentences.',
    'The corpus is a generated toy language. Nouns, adjectives and verb stems are chosen at random, so even a perfect model cannot predict them; only the agreement-bearing words are predictable.',
    'Weights start from N(0, 0.02²) without truncation; LayerNorm uses ε = 10⁻⁵ instead of 10⁻¹².',
  ],
  guide: {
    what: 'A short two-sentence text goes in with about 15 % of its words selected, most of them replaced by a [MASK] token; for each selected position the model outputs a probability for every symbol in its 37-symbol vocabulary (33 words and punctuation, 4 special tokens). Every position reads every other through attention in both directions, so a masked verb can consult the noun before it and the pronoun after it. Training rewards putting probability on the original word. In this toy language that means learning the fixed sentence pattern and singular or plural agreement; which noun, adjective or verb was used is random and cannot be learned.',
    watch:
      'Watch the loss fall, from 3.6 to about 1.1 in 400 steps, and the two accuracies in the metrics: overall accuracy reaches only about 0.57, because the random nouns, adjectives and verb stems cannot be guessed, while agreement accuracy (pronoun and copula) climbs from 0 to about 0.5 by step 200 and about 0.9 by step 400. In the last-block attention map, the row of the masked pronoun in the opening example comes to put most of its weight on the copula next to it, which carries the same number, in some of the four heads (the head buttons switch between them). Averaged over the probe sentences, number-bearing words get only about 1.2 times the weight that uniform attention would give them.',
    try: [
      {
        label: 'Select half the tokens',
        note: 'About five words per sentence are now selected instead of 1.7 (selected per sequence in the metrics), so each prediction has less context but each step trains on three times as many. After 200 steps the two effects roughly cancel: overall accuracy is about 0.5 and agreement accuracy about 0.55, close to the default rate’s 0.48 and 0.53.',
        patch: { maskRate: 0.5 },
        train: 200,
      },
      {
        label: 'Drop segment embeddings',
        note: 'Re-initialises. Expect little change: [SEP] already marks the sentence boundary. After 200 steps both accuracies are within a few points of the default run’s (overall about 0.49, agreement about 0.49).',
        patch: { segments: false },
        train: 200,
      },
      {
        label: 'One block, one head',
        note: 'Re-initialises. A single attention map is easier to read. After 200 steps agreement accuracy is about the same as with two blocks of four heads (about 0.5), and overall accuracy a little lower (0.44 against 0.48).',
        patch: { layers: 1, heads: 1 },
        train: 200,
      },
    ],
    views: ['MLM prediction at the first selected position', 'Self-attention, last block'],
  },
  configSchema: [
    ...widthFields({ dMin: 8, dMax: 96, dStep: 8, maxLayers: MAX_LAYERS }),
    {
      key: 'segments',
      label: 'Segment embeddings',
      kind: 'bool',
      affects: 'architecture',
      help: 'Add E_A to the first sentence and E_B to the second.',
    },
    {
      key: 'maskRate',
      label: 'Selection rate',
      kind: 'float',
      min: 0.05,
      max: 0.5,
      step: 0.05,
      affects: 'forward',
      help: 'Fraction of word positions selected for prediction (paper: 0.15).',
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
  defaults: { layers: 2, dModel: 32, heads: 4, segments: true, maskRate: 0.15, dropout: 0 },
  validate: (c) => headsProblem(c.dModel, c.heads),

  init(c, rng) {
    const b = new ParamBuilder(rng);
    const d = c.dModel;
    const std = { std: 0.02 };
    b.add('embed.tok', [V, d], { kind: 'normal', std: 0.02 });
    b.add('embed.pos', [GRAMMAR_T, d], { kind: 'normal', std: 0.02 });
    if (c.segments) b.add('embed.seg', [2, d], { kind: 'normal', std: 0.02 });
    declareNorm(b, 'embed.ln', d);
    for (let l = 1; l <= c.layers; l++) declareEncoderBlock(b, `block${l}`, d, 4 * d, std);
    declareLinear(b, 'head.transform', d, d, std);
    declareNorm(b, 'head.ln', d);
    b.add('head.bias', [V], { kind: 'zeros' });
    return b.store;
  },

  datasets: [grammarDataset<BertConfig>((c) => c.maskRate)],

  forward(batch, c): ForwardResult {
    const [B, T] = batch.inputs.ids.shape;
    // Fixed corruption (validation, probes) or BERT's 80/10/10 rule drawn from the graph's generator.
    const given = batch.targets?.mlm;
    const corr = given
      ? { input: Array.from(batch.inputs.ids.data), target: Array.from(given.data) }
      : mlmCorrupt(batch.inputs.ids.data, T, graphRng(), c.maskRate);
    const pad = Array.from({ length: B }, (_, b) =>
      Array.from({ length: T }, (_, t) => corr.input[b * T + t] === PAD),
    );
    const mask = keyPaddingMask(pad, c.heads, T);

    const h0 = scope('embed', () => {
      let e = add(take(p('embed.tok'), corr.input, [B, T]), slice(p('embed.pos'), 0, 0, T));
      if (c.segments) e = add(e, take(p('embed.seg'), batch.inputs.segments.data, [B, T]));
      return dropout(norm(e, 'embed.ln'), c.dropout);
    });
    let h = h0;
    const attn: Var[] = [];
    for (let l = 1; l <= c.layers; l++) {
      const r = encoderBlock(h, `block${l}`, {
        heads: c.heads,
        norm: 'post',
        act: 'gelu',
        dropout: c.dropout,
        mask,
      });
      h = r.y;
      attn.push(r.probs);
    }

    const selected: number[] = [];
    const targets: number[] = [];
    corr.target.forEach((t, i) => {
      if (t >= 0) {
        selected.push(i);
        targets.push(t);
      }
    });
    const logits = scope('head', () => {
      const rows = take(reshape(h, [B * T, c.dModel]), selected, [selected.length]);
      const g = norm(geluTanh(dense(rows, 'head.transform')), 'head.ln');
      return linear(g, p('embed.tok'), p('head.bias'));
    });
    const probs = softmax(logits, -1);
    const loss = selected.length ? scope('loss', () => crossEntropy(logits, targets)) : undefined;

    let hit = 0;
    let agreeN = 0;
    let agreeHit = 0;
    targets.forEach((t, r) => {
      let best = 0;
      for (let v = 1; v < V; v++)
        if (probs.value.data[r * V + v] > probs.value.data[r * V + best]) best = v;
      if (best === t) hit++;
      if (AGREEMENT.has(t)) {
        agreeN++;
        if (best === t) agreeHit++;
      }
    });
    const metrics: Record<string, number> = {
      accuracy: targets.length ? hit / targets.length : 0,
      selectedPerSequence: targets.length / B,
    };
    if (agreeN) metrics.agreementAccuracy = agreeHit / agreeN;

    return {
      outputs: { h0, attn1: attn[0], attnLast: attn[attn.length - 1], logits, mlmProbs: probs },
      loss,
      metrics,
    };
  },

  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 2e-3, clip: 'norm', clipValue: 1 },
    batchSize: 32,
    steps: 400,
    validateEvery: 50,
  },

  views: [
    {
      kind: 'probs',
      title: 'MLM prediction at the first selected position',
      source: 'out:mlmProbs',
      classNames: GRAMMAR_VOCAB,
    },
    {
      kind: 'attention',
      title: 'Self-attention, first block',
      source: 'out:attn1',
      tokens: 'meta',
    },
    {
      kind: 'attention',
      title: 'Self-attention, last block',
      source: 'out:attnLast',
      tokens: 'meta',
    },
  ],

  math: {
    embed: {
      tex: 'h^{(0)}_i = \\operatorname{LN}\\big(E[\\tilde x_i] + P_i + S_{s_i}\\big)',
      text: 'Token, learned position and segment embeddings are summed and normalised. Before embedding, selected tokens are replaced: 80 % by [MASK], 10 % by a random word, 10 % unchanged.',
    },
    block: {
      tex: "a = \\operatorname{LN}(h + \\operatorname{MHA}(h)),\\quad h' = \\operatorname{LN}(a + \\operatorname{FFN}(a))",
      text: 'Post-LN encoder layer with GELU. Attention is bidirectional; only [PAD] keys are masked.',
    },
    ...nestedMath('block', MAX_LAYERS, {
      attn: { tex: ATTENTION_TEX, text: 'M_{ik} = −∞ where key k is [PAD], 0 elsewhere.' },
      ffn: { tex: ffnTex('gelu') },
    }),
    head: {
      tex: 'g_i = \\operatorname{LN}\\big(\\operatorname{GELU}(W h^{(N)}_i + b)\\big),\\quad p(\\cdot\\mid\\tilde x) = \\operatorname{softmax}(g_i E^{\\top} + b_{\\text{out}})',
      text: 'Applied only at the selected positions. The output matrix is the token embedding E.',
    },
    loss: {
      tex: 'L = -\\frac{1}{|\\mathcal M|}\\sum_{i\\in\\mathcal M} \\log p(x_i \\mid \\tilde x)',
      text: 'Only selected positions contribute, including those left unchanged or replaced by a random word.',
    },
  },
});

export default bert;
