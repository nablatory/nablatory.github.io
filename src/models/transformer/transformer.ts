import {
  ParamBuilder,
  Tensor,
  add,
  crossEntropy,
  declareNorm,
  defaultOptimizer,
  dropout,
  linear,
  mulScalar,
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
  LN_TEX,
  NORM_OPTIONS,
  causalMask,
  declareDecoderBlock,
  declareEncoderBlock,
  decoderBlock,
  encoderBlock,
  ffnTex,
  headsProblem,
  nestedMath,
  norm,
  sinusoidalEncoding,
  widthFields,
  type NormPlacement,
} from './attention';
import { BOS, MAX_SEQ, SEQ_VOCAB, seq2seqDataset, targetOf, type Seq2SeqTask } from './seq2seq';

type TransformerConfig = {
  task: Seq2SeqTask;
  length: number;
  layers: number;
  dModel: number;
  heads: number;
  norm: NormPlacement;
  decoderMask: 'causal' | 'none';
  dropout: number;
};

const V = SEQ_VOCAB.length;
const MAX_LAYERS = 6;

/** Shared embedding scaled by √d plus the fixed sinusoidal encoding (§3.4–3.5). */
function embed(ids: ArrayLike<number>, B: number, T: number, c: TransformerConfig): Var {
  const e = mulScalar(take(p('embed.tok'), ids, [B, T]), Math.sqrt(c.dModel));
  return dropout(add(e, slice(p('embed.pe'), 0, 0, T)), c.dropout);
}

function encode(src: ArrayLike<number>, B: number, L: number, c: TransformerConfig) {
  let h = scope('embed', () => embed(src, B, L, c));
  const attn: Var[] = [];
  for (let l = 1; l <= c.layers; l++) {
    const r = encoderBlock(h, `enc${l}`, {
      heads: c.heads,
      norm: c.norm,
      act: 'relu',
      dropout: c.dropout,
    });
    h = r.y;
    attn.push(r.probs);
  }
  if (c.norm === 'pre') h = scope('encNorm', () => norm(h, 'encNorm'));
  return { memory: h, attn };
}

function decode(dec: ArrayLike<number>, B: number, T: number, memory: Var, c: TransformerConfig) {
  const y0 = scope('embedTgt', () => embed(dec, B, T, c));
  const mask = c.decoderMask === 'causal' ? causalMask(T) : undefined;
  let h = y0;
  const selfAttn: Var[] = [];
  const crossAttn: Var[] = [];
  for (let l = 1; l <= c.layers; l++) {
    const r = decoderBlock(h, memory, `dec${l}`, {
      heads: c.heads,
      norm: c.norm,
      act: 'relu',
      dropout: c.dropout,
      mask,
    });
    h = r.y;
    selfAttn.push(r.selfProbs);
    crossAttn.push(r.crossProbs);
  }
  if (c.norm === 'pre') h = scope('decNorm', () => norm(h, 'decNorm'));
  // Pre-softmax projection shares the embedding matrix (§3.4).
  const logits = scope('head', () => linear(h, p('embed.tok')));
  return { y0, logits, selfAttn, crossAttn };
}

const transformer = defineModel<TransformerConfig>({
  id: 'transformer',
  family: 'transformer',
  name: 'Transformer (encoder–decoder)',
  summary:
    'The original attention-only sequence-to-sequence model: an encoder of self-attention layers and a decoder that also attends to the encoder output, here learning to reverse or sort digit strings.',
  signature:
    '\\operatorname{Attention}(Q,K,V) = \\operatorname{softmax}\\!\\big(QK^{\\top}/\\sqrt{d_k}\\big)V,\\quad x \\mapsto \\operatorname{LN}(x + \\operatorname{Sublayer}(x))',
  fidelity: 'reduced',
  scaleNote:
    'Reduced from the base model (N = 6 encoder and 6 decoder layers, d_model = 512, 8 heads, d_ff = 2048, 37,000 shared BPE tokens, WMT translation) to N = 2 + 2 layers, d_model = 32, 4 heads, d_ff = 4·d_model = 128, an 11-symbol vocabulary (digits and <s>) and sequences of 6 digits. Sinusoidal encodings, √d embedding scaling, tied embeddings and post-LN residual blocks are as published.',
  references: [
    {
      authors:
        'A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin',
      title: 'Attention is all you need',
      year: 2017,
      url: 'https://arxiv.org/abs/1706.03762',
    },
    {
      authors:
        'R. Xiong, Y. Yang, D. He, K. Zheng, S. Zheng, C. Xing, H. Zhang, Y. Lan, L. Wang, T.-Y. Liu',
      title: 'On layer normalization in the transformer architecture',
      year: 2020,
      url: 'https://arxiv.org/abs/2002.04745',
    },
  ],
  limitations: [
    'A synthetic digit task instead of machine translation; sequences have a fixed length, so no padding mask, end-of-sequence symbol or beam search is needed.',
    'Trained with constant-rate Adam; the paper’s warm-up schedule (lr ∝ d^{−0.5}·min(step^{−0.5}, step·warmup^{−1.5})) and label smoothing (ε = 0.1) are not used.',
    'Dropout applies to sublayer outputs and to the embedding sums, as in the paper, and also to the attention weights (as in the reference implementation; the paper does not mention it). It is 0 by default because the task has unlimited fresh data.',
    'In the attention maps rows are query positions and columns key positions. Encoder positions are labelled with the source digits and decoder positions with the decoder input (<s>, y_1, …, y_{L−1}), so decoder row t is the query that predicts y_{t+1}.',
  ],
  guide: {
    what: 'A string of digits (six by default) goes in and the same digits come out reversed, or sorted. The encoder turns each digit into a vector and lets every position read every other through attention, a weighted average whose weights the network computes from the data itself; the decoder then emits one digit at a time, attending to the encoder’s vectors and to the digits before it (in training, the correct ones). Training adjusts the weights so that the digit predicted at each step matches the target.',
    watch:
      'From random weights the loss falls from about 2.9 to below 1 within the first hundred steps, and to below 0.1 by step 400. In the cross-attention map, rows are decoder positions and columns source digits: for the reverse task it turns from an even blur into an anti-diagonal, the first decoder row reading the last source digit, the next row the one before it. By step 400 about two thirds of each row’s weight lies on that diagonal. Accuracy and sequence accuracy in the metrics rise with it, to about 0.99 and 0.93.',
    try: [
      {
        label: 'Sort instead of reverse',
        note: 'The anti-diagonal disappears: where each output digit sits in the source depends on all the digits. Even so, this mapping is learned faster than reversal here: accuracy is about 0.9 after 50 steps (reversal: about 0.37) and 0.99 after 200.',
        patch: { task: 'sort' },
        train: 200,
      },
      {
        label: 'Remove the causal mask',
        note: 'Each decoder position can now also read the decoder inputs after it, which include its own target. After 400 steps the loss (about 0.01) and accuracy look as good as with the mask, yet “Greedy decoding”, which feeds back its own outputs and so has no later inputs to read, gets only 41 % of the digits of the 24 example strings right (with the mask: 97 %).',
        patch: { decoderMask: 'none' },
        train: 400,
      },
      {
        label: 'Pre-LN blocks',
        note: 'Re-initialises. Normalisation moves to each sublayer’s input, as in GPT-2 and ViT. Here the loss starts higher but falls faster: about 0.04 after 200 steps, against about 0.2 for post-LN.',
        patch: { norm: 'pre' },
        train: 200,
      },
    ],
    views: [
      'Cross-attention, last decoder layer (rows: decoder positions labelled by their input symbol, columns: source positions)',
      'Output distribution at the first decoder position',
    ],
  },
  configSchema: [
    {
      key: 'task',
      label: 'Task',
      kind: 'select',
      affects: 'data',
      options: [
        { value: 'reverse', label: 'Reverse the digits' },
        { value: 'sort', label: 'Sort the digits' },
      ],
    },
    {
      key: 'length',
      label: 'Sequence length L',
      kind: 'int',
      min: 2,
      max: MAX_SEQ,
      affects: 'data',
    },
    ...widthFields({
      dMin: 8,
      dMax: 96,
      dStep: 8,
      maxLayers: MAX_LAYERS,
      layerLabel: 'Layers N (each stack)',
    }),
    {
      key: 'norm',
      label: 'LayerNorm placement',
      kind: 'select',
      affects: 'architecture',
      options: NORM_OPTIONS,
      help: 'Pre-LN adds a final LayerNorm to each stack.',
    },
    {
      key: 'decoderMask',
      label: 'Decoder self-attention mask',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'causal', label: 'Causal (paper)' },
        { value: 'none', label: 'None (decoder sees future targets)' },
      ],
    },
    {
      key: 'dropout',
      label: 'Dropout rate P_drop',
      kind: 'float',
      min: 0,
      max: 0.5,
      step: 0.05,
      affects: 'forward',
    },
  ],
  defaults: {
    task: 'reverse',
    length: 6,
    layers: 2,
    dModel: 32,
    heads: 4,
    norm: 'post',
    decoderMask: 'causal',
    dropout: 0,
  },
  validate: (c) => headsProblem(c.dModel, c.heads),

  init(c, rng) {
    const b = new ParamBuilder(rng);
    const d = c.dModel;
    const xavier = { init: 'xavier' as const };
    b.add('embed.tok', [V, d], { kind: 'normal', std: 1 / Math.sqrt(d) });
    b.fixed(
      'embed.pe',
      sinusoidalEncoding(MAX_SEQ, d),
      'Sinusoidal encoding, sin/cos(pos / 10000^{2i/d}) (fixed)',
    );
    for (let l = 1; l <= c.layers; l++) declareEncoderBlock(b, `enc${l}`, d, 4 * d, xavier);
    if (c.norm === 'pre') declareNorm(b, 'encNorm', d);
    for (let l = 1; l <= c.layers; l++) declareDecoderBlock(b, `dec${l}`, d, 4 * d, xavier);
    if (c.norm === 'pre') declareNorm(b, 'decNorm', d);
    return b.store;
  },

  datasets: [
    seq2seqDataset<TransformerConfig>(
      (c) => c.task,
      (c) => c.length,
    ),
  ],

  forward(batch, c): ForwardResult {
    const { src, dec } = batch.inputs;
    const [B, L] = src.shape;
    const enc = encode(src.data, B, L, c);
    const out = decode(dec.data, B, L, enc.memory, c);
    const probs = softmax(out.logits, -1);
    const tgt = batch.targets?.tgt;
    let loss: Var | undefined;
    let metrics: Record<string, number> | undefined;
    if (tgt) {
      loss = scope('loss', () => crossEntropy(reshape(out.logits, [B * L, V]), tgt.data));
      metrics = sequenceMetrics(probs.value, tgt.data, B, L);
    }
    return {
      outputs: {
        decoderInput: out.y0,
        logits: out.logits,
        probs,
        encAttn: enc.attn[enc.attn.length - 1],
        selfAttn: out.selfAttn[out.selfAttn.length - 1],
        crossAttn: out.crossAttn[out.crossAttn.length - 1],
      },
      loss,
      metrics,
    };
  },

  training: {
    support: 'full',
    optimizer: {
      ...defaultOptimizer,
      kind: 'adam',
      lr: 2e-3,
      beta2: 0.98,
      eps: 1e-9,
      clip: 'norm',
      clipValue: 1,
    },
    batchSize: 32,
    steps: 400,
    validateEvery: 50,
  },

  procedures: [
    {
      id: 'greedy',
      label: 'Greedy decoding',
      description:
        'Encode the source once, then generate the output one symbol at a time: feed (<s>, ŷ_1, …, ŷ_{t−1}) to the decoder and take the most probable digit (the start symbol is never emitted). No target symbols are used.',
      args: [
        {
          key: 'example',
          label: 'Source string (example index)',
          kind: 'int',
          min: 0,
          max: 23,
          affects: 'data',
        },
      ],
      defaults: { example: 0 },
      run(ctx) {
        const c = ctx.config;
        const batch = ctx.dataset.example(Number(ctx.args.example), c);
        const src = Array.from(batch.inputs.src.data);
        const L = src.length;
        const enc = encode(src, 1, L, c);
        const decIds = [BOS];
        const probs = new Float32Array(L * V);
        let cross: Tensor | null = null;
        for (let t = 0; t < L; t++) {
          if (ctx.shouldStop()) break;
          const out = decode(decIds, 1, decIds.length, enc.memory, c);
          const row = softmax(out.logits, -1).value.data.subarray(t * V, (t + 1) * V);
          probs.set(row, t * V);
          let best = 0;
          for (let v = 1; v < 10; v++) if (row[v] > row[best]) best = v;
          if (t === L - 1) cross = out.crossAttn[out.crossAttn.length - 1].value;
          decIds.push(best);
          ctx.progress((t + 1) / L);
        }
        const yHat = decIds.slice(1);
        const y = targetOf(c.task, src);
        const correct = yHat.filter((v, i) => v === y[i]).length;
        const text = [
          `source   ${src.join(' ')}`,
          `greedy   ${yHat.join(' ')}`,
          `target   ${y.join(' ')}`,
          `${correct} of ${L} positions correct`,
        ].join('\n');
        const tensors: Record<string, Tensor> = {
          probs: new Tensor([yHat.length, V], probs.subarray(0, yHat.length * V)),
        };
        if (cross) tensors.cross = new Tensor(cross.shape.slice(1), cross.data);
        const meta = {
          sourceTokens: src.map((v) => SEQ_VOCAB[v]),
          // Rows of the last step's cross-attention: the decoder inputs <s>, ŷ_1, …, ŷ_{L−1}.
          decoderTokens: decIds.slice(0, L).map((v) => SEQ_VOCAB[v]),
        };
        return { tensors, text, scalars: { correct }, meta };
      },
    },
  ],

  views: [
    { kind: 'text', title: 'Greedy decoding', source: 'proc:greedy:text' },
    {
      kind: 'attention',
      title:
        'Cross-attention while decoding greedily (last decoder layer; rows: decoder positions labelled by their input symbol, columns: source positions)',
      source: 'proc:greedy:cross',
      queryTokens: 'decoderTokens',
      keyTokens: 'sourceTokens',
    },
    {
      kind: 'attention',
      title:
        'Cross-attention, last decoder layer (rows: decoder positions labelled by their input symbol, columns: source positions)',
      source: 'out:crossAttn',
      queryTokens: 'decoderTokens',
      keyTokens: 'sourceTokens',
    },
    {
      kind: 'attention',
      title: 'Encoder self-attention, last layer',
      source: 'out:encAttn',
      queryTokens: 'sourceTokens',
      keyTokens: 'sourceTokens',
    },
    {
      kind: 'attention',
      title: 'Decoder masked self-attention, last layer',
      source: 'out:selfAttn',
      queryTokens: 'decoderTokens',
      keyTokens: 'decoderTokens',
    },
    {
      kind: 'probs',
      title: 'Output distribution at the first decoder position',
      source: 'out:probs',
      classNames: SEQ_VOCAB,
    },
  ],

  math: {
    embed: {
      tex: 'h^{(0)}_i = \\sqrt{d_{\\text{model}}}\\, E[x_i] + PE_i,\\quad PE_{(pos,2i)} = \\sin\\!\\big(pos/10000^{2i/d}\\big),\\; PE_{(pos,2i+1)} = \\cos(\\cdot)',
      text: 'Source symbols are embedded with the shared matrix E, scaled by √d, and the fixed sinusoidal encoding of the position is added.',
    },
    embedTgt: {
      tex: 'g^{(0)}_t = \\sqrt{d_{\\text{model}}}\\, E[y_{t-1}] + PE_t,\\quad y_0 = \\langle s\\rangle',
      text: 'The decoder input is the target shifted right by one (teacher forcing).',
    },
    enc: {
      tex: "a = \\operatorname{LN}(h + \\operatorname{MHA}(h,h,h)),\\quad h' = \\operatorname{LN}(a + \\operatorname{FFN}(a))",
      text: 'Post-LN encoder layer (paper). With pre-LN: a = h + MHA(LN h), h′ = a + FFN(LN a).',
    },
    dec: {
      tex: "a = \\operatorname{LN}(g + \\operatorname{MHA}_{M}(g,g,g)),\\; b = \\operatorname{LN}(a + \\operatorname{MHA}(a, m, m)),\\; g' = \\operatorname{LN}(b + \\operatorname{FFN}(b))",
      text: 'Masked self-attention, then cross-attention whose keys and values m come from the encoder output, then the feed-forward network.',
    },
    ...nestedMath('enc', MAX_LAYERS, {
      attn: {
        tex: ATTENTION_TEX,
        text: 'Encoder self-attention: no mask (M = 0); every source position reads every other.',
      },
      ffn: { tex: ffnTex('relu') },
    }),
    ...nestedMath('dec', MAX_LAYERS, {
      attn: {
        tex: ATTENTION_TEX,
        text: 'Causal mask: M_{tk} = −∞ for k > t, so output t cannot read targets it has not produced yet.',
      },
      cross: {
        tex: '\\operatorname{MultiHead}(Q = a,\\; K = V = m)',
        text: 'Queries from the decoder, keys and values from the encoder output m. No mask.',
      },
      ffn: { tex: ffnTex('relu') },
    }),
    encNorm: { tex: LN_TEX, text: 'Final LayerNorm of the encoder stack (pre-LN only).' },
    decNorm: { tex: LN_TEX, text: 'Final LayerNorm of the decoder stack (pre-LN only).' },
    head: {
      tex: 'p(y_t \\mid y_{<t}, x) = \\operatorname{softmax}(g^{(N)}_t E^{\\top})',
      text: 'The pre-softmax projection is the transposed embedding matrix.',
    },
    loss: { tex: 'L = -\\frac{1}{BL}\\sum_{b,t} \\log p(y_{b,t} \\mid y_{b,<t}, x_b)' },
  },
});

function sequenceMetrics(
  probs: Tensor,
  targets: ArrayLike<number>,
  B: number,
  L: number,
): Record<string, number> {
  let tok = 0;
  let seq = 0;
  for (let b = 0; b < B; b++) {
    let all = true;
    for (let t = 0; t < L; t++) {
      const r = b * L + t;
      // Arg max over the ten digits, as in greedy decoding (the start symbol is never emitted).
      let best = 0;
      for (let v = 1; v < 10; v++) if (probs.data[r * V + v] > probs.data[r * V + best]) best = v;
      if (best === targets[r]) tok++;
      else all = false;
    }
    if (all) seq++;
  }
  return { accuracy: tok / (B * L), sequenceAccuracy: seq / B };
}

export default transformer;
