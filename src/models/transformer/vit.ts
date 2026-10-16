import {
  ParamBuilder,
  Tensor,
  add,
  concat,
  crossEntropy,
  declareLinear,
  declareNorm,
  defaultOptimizer,
  dense,
  dropout,
  mean,
  p,
  reshape,
  scope,
  select,
  slice,
  softmax,
  take,
  transpose,
  type Var,
} from '../../engine';
import { defineModel, type Batch, type Config, type Dataset, type ForwardResult } from '../types';
import { DIGITS, MNIST_PREPROCESSING, encodeDigit, mnistDataset } from '../data/mnist';
import { accuracy, input } from '../util';
import {
  ATTENTION_TEX,
  declareEncoderBlock,
  encoderBlock,
  ffnTex,
  headsProblem,
  nestedMath,
  norm,
  widthFields,
} from './attention';

type VitConfig = {
  patch: '4' | '7' | '14';
  layers: number;
  dModel: number;
  heads: number;
  dropout: number;
};

const IMG = 28;
const MAX_LAYERS = 6;

function patchGeometry(c: VitConfig) {
  const P = Number(c.patch);
  const g = IMG / P;
  return { P, g, n: g * g };
}

/** [B, 1, 28, 28] → [B, N, P²]: non-overlapping P×P patches in row-major order, each flattened row by row. */
function patchify(x: Var, P: number): Var {
  const B = x.shape[0];
  const g = IMG / P;
  const blocks = reshape(x, [B, g, P, g, P]); // b, gy, py, gx, px
  return reshape(transpose(blocks, [0, 1, 3, 2, 4]), [B, g * g, P * P]);
}

export const vitTokens = (n: number) => ['CLS', ...Array.from({ length: n }, (_, i) => `p${i}`)];

/** MNIST in [0, 1], with token labels ([CLS], p0 …) attached to inspection examples for the attention view. */
const mnist = mnistDataset<VitConfig>(() => 'unit');
const dataset: Dataset<VitConfig> = {
  ...mnist,
  example(index, c) {
    const b = mnist.example(index, c);
    return { ...b, meta: { ...b.meta, tokens: vitTokens(patchGeometry(c).n) } };
  },
};

const vit = defineModel<VitConfig>({
  id: 'vit',
  family: 'transformer',
  name: 'Vision Transformer (ViT)',
  summary:
    'An image cut into patches, each patch linearly embedded as a token, a learned [CLS] token prepended, and a transformer encoder classifying from the [CLS] output.',
  signature:
    'z_0 = [x_{\\text{cls}};\\, x_p^1 E;\\, \\dots;\\, x_p^N E] + E_{\\text{pos}},\\quad y = \\operatorname{softmax}\\big(W\\,\\operatorname{LN}(z_L^0)\\big)',
  fidelity: 'reduced',
  scaleNote:
    'Reduced from ViT-B/16 (224×224 RGB images, 16×16 patches → 196 tokens, 12 layers, D = 768, 12 heads, MLP 3,072, pre-trained on ImageNet-21k or JFT-300M) to 28×28 MNIST digits with 7×7 patches → 16 tokens, 3 layers, D = 32, 4 heads, MLP 128, trained from scratch on MNIST.',
  references: [
    {
      authors:
        'A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby',
      title: 'An image is worth 16x16 words: transformers for image recognition at scale',
      year: 2021,
      url: 'https://arxiv.org/abs/2010.11929',
    },
    {
      authors: 'Y. LeCun, C. Cortes, C. J. C. Burges',
      title: 'The MNIST database of handwritten digits',
      year: 1998,
      url: 'http://yann.lecun.com/exdb/mnist/',
    },
  ],
  limitations: [
    'No pre-training: the shipped checkpoint was trained from scratch on the 60,000 MNIST training images by scripts/train/vit-mnist.ts, and browser training uses the 6,000-image subset. ViT’s published results depend on large-scale pre-training; without it ViTs underperform convolutional networks of similar size, and this one reaches 95.96 % MNIST test accuracy against 98.29 % for the LeNet-5 checkpoint.',
    'The classification head is a single linear layer on the final [CLS] state (the paper’s fine-tuning head); the one-hidden-layer pre-training head is not used.',
    'No data augmentation and no weight decay. Browser training uses constant-rate Adam without warm-up; the shipped checkpoint used a 100-step linear warm-up and cosine decay (stated in its provenance).',
    'The attention-from-[CLS] image shows weights of the last layer averaged over heads. It shows where [CLS] reads from, not which pixels the prediction depends on.',
  ],
  guide: {
    what: 'A 28×28 digit is cut into 16 patches of 7×7 pixels, and each patch is flattened and mapped to a 32-number vector called a token. A learned [CLS] token is added, every token updates itself by attending to the others through three transformer blocks, and the final [CLS] vector is read by a linear layer that scores the ten digit classes. Training adjusts the patch embedding, the attention weights and the classifier so that the correct class scores highest.',
    watch:
      'This page opens with a checkpoint trained on MNIST: 95.96 % accuracy on the 10,000 MNIST test images, 95.4 % on the 500 validation digits used here. Press “Start from random weights” to train from scratch. Accuracy on the 500 validation digits, reported every 50 steps, reaches about 0.8 after 400 steps, and the class probabilities become peaked: the top class gets 0.82 on average, up from 0.10. The [CLS] attention image shows which patches [CLS] reads from; neither here nor at the checkpoint does it reliably single out the patches that carry ink. Training the checkpoint further here, at a constant learning rate, lowers its accuracy a little (about 0.92 after 400 steps).',
    try: [
      {
        label: 'Four quadrant patches',
        note: 'Re-initialises. Each token is a 14×14 quadrant, so the [CLS] attention image has only four cells and the attention map five tokens. Four coarse tokens are learned quickly: accuracy is about 0.87 after 200 steps, against about 0.71 for 7×7 patches trained from scratch for as long.',
        patch: { patch: '14' },
        train: 200,
      },
      {
        label: 'Small 4×4 patches',
        note: 'Re-initialises. 49 patch tokens make the [CLS] attention map a 7×7 image of 4×4-pixel cells. Each step is several times slower, and after 200 steps accuracy (about 0.77) is close to that of 7×7 patches.',
        patch: { patch: '4' },
        train: 200,
      },
      {
        label: 'One block',
        note: 'Re-initialises. A single block still learns to classify: about 0.74 accuracy after 200 steps, as good as three blocks trained from scratch for as long (0.71). Compare its class probabilities with those of the three-block checkpoint, trained far longer, on the same digit.',
        patch: { layers: 1 },
        train: 200,
      },
    ],
    views: [
      'Attention from [CLS] to each patch, last layer, mean over heads',
      'Class probabilities',
    ],
  },
  configSchema: [
    {
      key: 'patch',
      label: 'Patch size P',
      kind: 'select',
      affects: 'architecture',
      options: [
        { value: '4', label: '4×4 → 49 tokens' },
        { value: '7', label: '7×7 → 16 tokens' },
        { value: '14', label: '14×14 → 4 tokens' },
      ],
    },
    ...widthFields({ dMin: 8, dMax: 96, dStep: 8, maxLayers: MAX_LAYERS }),
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
  defaults: { patch: '7', layers: 3, dModel: 32, heads: 4, dropout: 0 },
  validate: (c) => headsProblem(c.dModel, c.heads),

  init(c, rng) {
    const b = new ParamBuilder(rng);
    const d = c.dModel;
    const { P, n } = patchGeometry(c);
    const xavier = { init: 'xavier' as const };
    declareLinear(b, 'embed.patch', P * P, d, xavier);
    b.add('embed.cls', [1, d], { kind: 'zeros' });
    b.add('embed.pos', [n + 1, d], { kind: 'normal', std: 0.02 });
    for (let l = 1; l <= c.layers; l++) declareEncoderBlock(b, `block${l}`, d, 4 * d, xavier);
    declareNorm(b, 'head.ln', d);
    declareLinear(b, 'head', d, 10, { init: 'zeros' });
    return b.store;
  },

  datasets: [dataset],

  forward(batch, c): ForwardResult {
    const x = input('image', batch.inputs.image);
    const B = x.shape[0];
    const { P, g, n } = patchGeometry(c);
    const { patches, z0 } = scope('embed', () => {
      const patches = patchify(x, P); // [B, N, P²]
      const tokens = dense(patches, 'embed.patch'); // [B, N, D]
      const cls = take(p('embed.cls'), new Array(B).fill(0), [B, 1]); // [B, 1, D]
      return { patches, z0: dropout(add(concat([cls, tokens], 1), p('embed.pos')), c.dropout) };
    });
    let z = z0;
    const attn: Var[] = [];
    for (let l = 1; l <= c.layers; l++) {
      const r = encoderBlock(z, `block${l}`, {
        heads: c.heads,
        norm: 'pre',
        act: 'gelu',
        dropout: c.dropout,
      });
      z = r.y;
      attn.push(r.probs);
    }
    const logits = scope('head', () => dense(norm(select(z, 1, 0), 'head.ln'), 'head'));
    const probs = softmax(logits, -1);
    const loss = batch.labels
      ? scope('loss', () => crossEntropy(logits, batch.labels!))
      : undefined;
    // Display tensors, computed from the forward values by engine ops.
    const last = attn[attn.length - 1];
    const { clsToPatches, patchImages } = scope('readout', () => ({
      clsToPatches: reshape(mean(slice(slice(last, 2, 0, 1), 3, 1, n + 1), 1), [B, 1, g, g]),
      patchImages: reshape(patches, [B * n, P, P]),
    }));
    return {
      outputs: {
        input: x,
        patches: patchImages,
        attn1: attn[0],
        attnLast: last,
        clsAttention: clsToPatches,
        logits,
        probs,
      },
      loss,
      metrics: batch.labels ? { accuracy: accuracy(probs.value, batch.labels) } : undefined,
    };
  },

  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 3e-3, clip: 'norm', clipValue: 1 },
    batchSize: 32,
    steps: 400,
    validateEvery: 50,
  },

  views: [
    { kind: 'image', title: 'Input (28×28)', source: 'out:input', range: [0, 1] },
    // One grid per patch size so the layout reproduces the image: 28/P patches per row.
    ...([4, 7, 14] as const).map((P) => ({
      kind: 'imageGrid' as const,
      title: 'Patches x_p¹ … x_p^N (row-major)',
      source: 'out:patches',
      range: [0, 1] as [number, number],
      columns: 28 / P,
      when: (c: Config) => Number(c.patch) === P,
    })),
    {
      kind: 'image',
      title: 'Attention from [CLS] to each patch, last layer, mean over heads',
      source: 'out:clsAttention',
    },
    {
      kind: 'attention',
      title: 'Self-attention, last layer (token 0 is [CLS])',
      source: 'out:attnLast',
      tokens: 'meta',
    },
    {
      kind: 'attention',
      title: 'Self-attention, first layer',
      source: 'out:attn1',
      tokens: 'meta',
    },
    { kind: 'probs', title: 'Class probabilities', source: 'out:probs', classNames: DIGITS },
  ],

  math: {
    embed: {
      tex: 'z_0 = [x_{\\text{cls}};\\, x_p^1 E;\\, \\dots;\\, x_p^N E] + E_{\\text{pos}},\\quad E \\in \\mathbb{R}^{P^2 \\times D},\\; E_{\\text{pos}} \\in \\mathbb{R}^{(N+1)\\times D}',
      text: 'Each P×P patch is flattened and multiplied by the same matrix E. A learned [CLS] vector is prepended and learned position embeddings are added.',
    },
    block: {
      tex: "z'_\\ell = \\operatorname{MSA}(\\operatorname{LN}(z_{\\ell-1})) + z_{\\ell-1},\\quad z_\\ell = \\operatorname{MLP}(\\operatorname{LN}(z'_\\ell)) + z'_\\ell",
      text: 'Pre-LN encoder block, equations (2)–(3) of the paper. No mask: every token attends to every token.',
    },
    ...nestedMath('block', MAX_LAYERS, {
      attn: { tex: ATTENTION_TEX },
      ffn: { tex: ffnTex('gelu') },
    }),
    head: {
      tex: 'y = \\operatorname{softmax}\\big(W\\, \\operatorname{LN}(z_L^0) + b\\big)',
      text: 'Only the final state of the [CLS] token is classified. The head starts at zero, as in the paper’s fine-tuning setup.',
    },
    loss: { tex: 'L = -\\frac{1}{B}\\sum_b \\log y_{b, c_b}' },
    readout: {
      tex: '\\bar\\alpha_j = \\frac{1}{h}\\sum_{i=1}^{h} \\alpha^{(i)}_{\\text{cls},\\,j}',
      text: 'Display only: the last layer’s attention from [CLS] to each patch, averaged over heads and arranged on the patch grid; and the patches themselves. Not part of the model.',
    },
  },

  checkpoints: [
    {
      id: 'mnist',
      label: 'Trained on MNIST',
      path: 'weights/vit-mnist',
      description:
        'Default configuration trained offline on the 60,000 MNIST training images (test accuracy in the file’s provenance).',
    },
  ],

  customInput: {
    kind: 'image',
    channels: 1,
    height: 28,
    width: 28,
    invertUploads: true,
    preprocessing: MNIST_PREPROCESSING.unit,
  },

  buildCustomBatch(t, c) {
    const u8 = new Float32Array(784);
    for (let i = 0; i < 784; i++) u8[i] = Math.max(0, Math.min(1, t.data[i])) * 255;
    const x = new Float32Array(784);
    encodeDigit(u8, 0, 'unit', x, 0);
    const { n } = patchGeometry(c);
    return {
      inputs: { image: new Tensor([1, 1, 28, 28], x) },
      meta: { classNames: DIGITS, tokens: vitTokens(n), caption: 'Your drawing' },
    } satisfies Batch;
  },
});

export default vit;
