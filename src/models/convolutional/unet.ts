import {
  ParamBuilder,
  Tensor,
  add,
  addScalar,
  concat,
  conv,
  convTranspose2d,
  declareConv,
  defaultOptimizer,
  div,
  maxPool2d,
  mean,
  mul,
  mulScalar,
  p,
  relu,
  scope,
  slice,
  softmax,
  sum,
  type Var,
} from '../../engine';
import { defineModel, type ForwardResult } from '../types';
import { shapesDataset, type ShapesSpec } from '../data/shapes';
import { constant, input } from '../util';
import { pixelCrossEntropy } from './common';

export type UNetConfig = {
  depth: number;
  baseChannels: number;
  classes: '2' | '3';
  padding: 'same' | 'valid';
  tile: number;
  noise: number;
  diceWeight: number;
};

export interface UNetGeometry {
  /** Input side. */
  input: number;
  /** Side of each contracting level's conv–conv output (the skip tensor), level 1 first. */
  skips: number[];
  /** Side of the bottleneck conv–conv output. */
  bottleneck: number;
  /** Side after each up-convolution, deepest level first. */
  upsampled: number[];
  /** Side of the segmentation map. */
  output: number;
}

/** Channels at level l (1-based); the bottleneck is level depth + 1. */
export const unetChannels = (c: UNetConfig, l: number) => c.baseChannels * 2 ** (l - 1);

/**
 * Spatial sizes through the network, or null when a size reaches zero or a
 * 2×2 max-pool would see an odd size (the pooled map could then not be
 * aligned with its skip connection after up-sampling).
 */
export function unetGeometry(c: UNetConfig): UNetGeometry | null {
  const shrink = c.padding === 'valid' ? 4 : 0;
  const skips: number[] = [];
  let n = c.tile;
  for (let l = 0; l < c.depth; l++) {
    n -= shrink;
    if (n < 2 || n % 2 !== 0) return null;
    skips.push(n);
    n /= 2;
  }
  n -= shrink;
  if (n < 1) return null;
  const bottleneck = n;
  const upsampled: number[] = [];
  for (let l = c.depth - 1; l >= 0; l--) {
    n *= 2;
    if (n > skips[l]) return null;
    upsampled.push(n);
    n -= shrink;
    if (n < 1) return null;
  }
  return { input: c.tile, skips, bottleneck, upsampled, output: n };
}

const classesOf = (c: UNetConfig): 2 | 3 => (c.classes === '2' ? 2 : 3);

function shapesSpec(c: UNetConfig): ShapesSpec {
  const g = unetGeometry(c);
  return { size: c.tile, crop: g ? g.output : c.tile, classes: classesOf(c), noise: c.noise };
}

/** Two 3×3 convolutions with ReLU. */
function doubleConv(x: Var, name: string, c: UNetConfig): Var {
  const padding = c.padding === 'same' ? 1 : 0;
  const a = scope('conv1', () => relu(conv(x, `${name}.conv1`, { padding })));
  return scope('conv2', () => relu(conv(a, `${name}.conv2`, { padding })));
}

/** Centre crop of an NCHW tensor to size × size. */
function centerCrop(x: Var, size: number): Var {
  const H = x.shape[2];
  if (H === size) return x;
  const o = (H - size) / 2;
  return slice(slice(x, 2, o, o + size), 3, o, o + size);
}

export interface UNetTrace {
  logits: Var;
  skips: Var[];
  bottleneck: Var;
  /** Concatenated [cropped skip, up-sampled] tensors, deepest level first. */
  concats: Var[];
  decoder: Var[];
}

export function unetNetwork(x: Var, c: UNetConfig): UNetTrace {
  const skips: Var[] = [];
  let h = x;
  for (let l = 1; l <= c.depth; l++) {
    if (l > 1) {
      const prev = h;
      h = scope(`pool${l - 1}`, () => maxPool2d(prev, 2));
    }
    const inp = h;
    h = scope(`down${l}`, () => doubleConv(inp, `down${l}`, c));
    skips.push(h);
  }
  const last = h;
  h = scope(`pool${c.depth}`, () => maxPool2d(last, 2));
  const pooled = h;
  const bottleneck = scope('bottleneck', () => doubleConv(pooled, 'bottleneck', c));
  h = bottleneck;
  const concats: Var[] = [];
  const decoder: Var[] = [];
  for (let l = c.depth; l >= 1; l--) {
    const below = h;
    const up = scope(`upconv${l}`, () =>
      convTranspose2d(below, p(`upconv${l}.weight`), p(`upconv${l}.bias`), { stride: 2 }),
    );
    h = scope(`up${l}`, () => {
      const skip = scope('crop', () => centerCrop(skips[l - 1], up.shape[2]));
      const cat = concat([skip, up], 1);
      concats.push(cat);
      return doubleConv(cat, `up${l}`, c);
    });
    decoder.push(h);
  }
  const top = h;
  const logits = scope('output', () => conv(top, 'output'));
  return { logits, skips, bottleneck, concats, decoder };
}

/** Per-class intersection-over-union of argmax predictions; mean over foreground classes present. */
export function segmentationMetrics(
  logits: Tensor,
  labels: ArrayLike<number>,
): { iou: number; pixelAccuracy: number } {
  const [N, K, H, W] = logits.shape;
  const P = H * W;
  const inter = new Array<number>(K).fill(0);
  const union = new Array<number>(K).fill(0);
  let correct = 0;
  for (let n = 0; n < N; n++)
    for (let q = 0; q < P; q++) {
      let best = 0;
      for (let k = 1; k < K; k++)
        if (logits.data[(n * K + k) * P + q] > logits.data[(n * K + best) * P + q]) best = k;
      const t = labels[n * P + q];
      if (best === t) {
        correct++;
        inter[t]++;
        union[t]++;
      } else {
        union[t]++;
        union[best]++;
      }
    }
  let s = 0;
  let m = 0;
  for (let k = 1; k < K; k++)
    if (union[k] > 0) {
      s += inter[k] / union[k];
      m++;
    }
  return { iou: m ? s / m : 1, pixelAccuracy: correct / (N * P) };
}

/** Display colours: ellipse orange, rectangle blue; with two classes the shape class is white. */
const COLOURS: Record<2 | 3, number[][]> = {
  2: [
    [0, 0, 0],
    [1, 1, 1],
  ],
  3: [
    [0, 0, 0],
    [1, 0.6, 0],
    [0.3, 0.7, 1],
  ],
};

/** RGB composite Σ_k w_k · colour_k of per-pixel class weights [N, K, H, W] (probabilities or one-hot). */
function colourise(w: Tensor, K: 2 | 3): Tensor {
  const [N, , H, W] = w.shape;
  const P = H * W;
  const out = new Float32Array(N * 3 * P);
  for (let n = 0; n < N; n++)
    for (let k = 1; k < K; k++)
      for (let ch = 0; ch < 3; ch++) {
        const col = COLOURS[K][k][ch];
        if (!col) continue;
        for (let q = 0; q < P; q++) out[(n * 3 + ch) * P + q] += col * w.data[(n * K + k) * P + q];
      }
  return new Tensor([N, 3, H, W], out);
}

function oneHot(labels: Tensor, K: number): Tensor {
  const [N, , H, W] = labels.shape;
  const P = H * W;
  const t = Tensor.zeros([N, K, H, W]);
  for (let n = 0; n < N; n++)
    for (let q = 0; q < P; q++) t.data[(n * K + labels.data[n * P + q]) * P + q] = 1;
  return t;
}

/** Soft Dice loss 1 − mean_k (2 Σ p_k g_k + ε) / (Σ p_k + Σ g_k + ε) over the foreground classes. */
function diceLoss(logits: Var, target: Tensor): Var {
  const K = logits.shape[1];
  const probs = softmax(logits, 1);
  const g = constant(target);
  const inter = sum(mul(probs, g), [0, 2, 3]);
  const denom = add(sum(probs, [0, 2, 3]), sum(g, [0, 2, 3]));
  const dice = div(addScalar(mulScalar(inter, 2), 1), addScalar(denom, 1));
  return addScalar(mulScalar(mean(slice(dice, 0, 1, K)), -1), 1);
}

const unet = defineModel<UNetConfig>({
  id: 'unet',
  family: 'convolutional',
  name: 'U-Net',
  summary:
    'Encoder–decoder for per-pixel classification: a contracting path of convolutions and pooling, an expanding path of up-convolutions, and skip connections that concatenate encoder features into the decoder at the same resolution.',
  signature:
    'd_l = \\mathrm{conv}^2\\big([\\,\\mathrm{crop}(e_l)\\,;\\ \\mathrm{upconv}(d_{l+1})\\,]\\big),\\quad e_l = \\mathrm{conv}^2(\\mathrm{pool}(e_{l-1}))',
  fidelity: 'reduced',
  scaleNote:
    'Published: 572×572 input tiles, 4 poolings, 64–1024 channels, unpadded convolutions with cropped skip connections, 388×388 output, 31 M parameters. Here: 32×32 tiles, 2 poolings, 8–32 channels (29,330 parameters at the defaults) on synthetic shapes with 2 classes (3 optional). Padded convolutions are the default, so the output has the input size; the paper’s unpadded convolutions with centre-cropped skips are selectable and need a larger tile.',
  references: [
    {
      authors: 'O. Ronneberger, P. Fischer, T. Brox',
      title: 'U-Net: convolutional networks for biomedical image segmentation',
      year: 2015,
      url: 'https://arxiv.org/abs/1505.04597',
    },
    {
      authors: 'F. Milletari, N. Navab, S.-A. Ahmadi',
      title: 'V-Net: fully convolutional neural networks for volumetric medical image segmentation',
      year: 2016,
      url: 'https://arxiv.org/abs/1606.04797',
    },
  ],
  limitations: [
    'The data are synthetic ellipses and rectangles, not microscopy images; there is no elastic-deformation augmentation.',
    'The paper’s pixel-wise loss weight map w(x), which emphasises thin borders between touching cells, is not used; all pixels have weight 1.',
    'In the unpadded variant the label map is the centre crop of a larger generated scene; the paper instead mirrors the image at its borders (overlap-tile strategy), which is not implemented.',
    'Trained with Adam rather than SGD with momentum 0.99 and batch size 1.',
    'The Dice term is an optional addition (Milletari et al., 2016), not part of the original U-Net.',
    'With 3 classes the network must tell ellipses from rectangles by their outlines, which this small model learns slowly: an offline run of the default size (1,800 Adam steps, batch 8) reached a mean shape-class IoU of only 0.28 on held-out scenes, so no 3-class checkpoint is shipped. The 2-class checkpoint reached 0.96.',
  ],
  guide: {
    what: 'U-Net labels every pixel of an image: here a 32×32 synthetic tile with ellipses and rectangles, each pixel classified as background or shape. A contracting path of convolutions and 2×2 max-pools shrinks the image to 8×8 while increasing the number of feature channels, an expanding path of up-convolutions grows it back, and skip connections copy the full-resolution features from the contracting side across so the output can keep sharp edges. Training learns all the filters from a pixel-wise classification loss.',
    watch:
      'This page opens with trained weights (IoU 0.97 on held-out scenes; IoU is the overlap between predicted and true shape pixels). Press “Start from random weights” to train from scratch: the prediction starts out meaningless, the background is learned within 50 steps, and the shapes fill in over 400 steps of 4 tiles (IoU about 0.85).',
    try: [
      {
        label: 'Three classes',
        note: 'Ellipse and rectangle must be told apart by outline: pixel accuracy reaches 90 % but shape IoU only about 0.2 after 400 steps.',
        patch: { classes: '3' },
        train: 400,
      },
      {
        label: 'Half the channels',
        note: '4, 8 and 16 channels (7,400 parameters): still IoU 0.8 after 400 steps, at a quarter of the time per step.',
        patch: { baseChannels: 4 },
        train: 400,
      },
      {
        label: 'Five times the noise',
        note: 'The trained network on much noisier tiles, without retraining: IoU falls from 0.97 to about 0.2.',
        patch: { noise: 0.5 },
      },
    ],
    views: ['Ground truth', 'Predicted class probabilities'],
  },
  configSchema: [
    {
      key: 'depth',
      label: 'Pooling levels',
      kind: 'int',
      min: 1,
      max: 3,
      affects: 'architecture',
      help: 'The paper uses 4.',
    },
    {
      key: 'baseChannels',
      label: 'Channels at level 1',
      kind: 'int',
      min: 4,
      max: 16,
      step: 4,
      affects: 'architecture',
      help: 'Doubled after every pooling, halved by every up-convolution.',
    },
    {
      key: 'classes',
      label: 'Classes',
      kind: 'select',
      affects: 'architecture',
      options: [
        { value: '2', label: '2: background / shape' },
        { value: '3', label: '3: background / ellipse / rectangle' },
      ],
    },
    {
      key: 'padding',
      label: 'Convolutions',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'same', label: 'Padded (output = input size)' },
        { value: 'valid', label: 'Unpadded + crop (paper)' },
      ],
      help: 'Same parameters. Unpadded 3×3 convolutions remove 2 pixels per layer, so skips are centre-cropped and the output is smaller than the input.',
    },
    {
      key: 'tile',
      label: 'Input tile',
      kind: 'int',
      min: 24,
      max: 96,
      step: 4,
      affects: 'data',
      help: 'Side of the generated image. Unpadded convolutions need 44 or more at depth 2.',
    },
    {
      key: 'noise',
      label: 'Pixel noise σ',
      kind: 'float',
      min: 0,
      max: 0.5,
      step: 0.05,
      affects: 'data',
    },
    {
      key: 'diceWeight',
      label: 'Dice loss weight λ',
      kind: 'float',
      min: 0,
      max: 1,
      step: 0.1,
      affects: 'forward',
      help: 'L = cross-entropy + λ · soft Dice.',
    },
  ],
  defaults: {
    depth: 2,
    baseChannels: 8,
    classes: '2',
    padding: 'same',
    tile: 32,
    noise: 0.1,
    diceWeight: 0,
  },

  validate(c) {
    if (unetGeometry(c)) return null;
    if (c.padding === 'same') {
      return `A ${c.tile}×${c.tile} tile cannot be halved ${c.depth} times without an odd size; with ${c.depth} pooling levels the tile must be a multiple of ${2 ** c.depth}.`;
    }
    return `With unpadded convolutions each conv–conv pair removes 4 pixels and every pooled size must be even; a ${c.tile}×${c.tile} tile with ${c.depth} pooling levels violates this or leaves no output. Increase the tile (e.g. 44, 60 or 64 at depth 2).`;
  },

  init(c, rng) {
    const b = new ParamBuilder(rng);
    let cin = 1;
    for (let l = 1; l <= c.depth; l++) {
      const ch = unetChannels(c, l);
      declareConv(b, `down${l}.conv1`, cin, ch, 3, { init: 'he' });
      declareConv(b, `down${l}.conv2`, ch, ch, 3, { init: 'he' });
      cin = ch;
    }
    const bc = unetChannels(c, c.depth + 1);
    declareConv(b, 'bottleneck.conv1', cin, bc, 3, { init: 'he' });
    declareConv(b, 'bottleneck.conv2', bc, bc, 3, { init: 'he' });
    let below = bc;
    for (let l = c.depth; l >= 1; l--) {
      const ch = unetChannels(c, l);
      // 2×2 up-convolution halving the channels; each output pixel sees C_in inputs.
      b.add(`upconv${l}.weight`, [below, ch, 2, 2], { kind: 'heNormal', fanIn: below });
      b.add(`upconv${l}.bias`, [ch], { kind: 'zeros' });
      declareConv(b, `up${l}.conv1`, 2 * ch, ch, 3, { init: 'he' });
      declareConv(b, `up${l}.conv2`, ch, ch, 3, { init: 'he' });
      below = ch;
    }
    declareConv(b, 'output', below, classesOf(c), 1, { init: 'he' });
    return b.store;
  },

  datasets: [shapesDataset<UNetConfig>(shapesSpec)],

  forward(batch, c): ForwardResult {
    const K = classesOf(c);
    const x = input('image', batch.inputs.image);
    const t = unetNetwork(x, c);
    const mask = batch.targets?.mask;
    let loss: Var | undefined;
    let metrics: Record<string, number> | undefined;
    let truth: Tensor | undefined;
    if (mask) {
      if (mask.shape[2] !== t.logits.shape[2]) {
        throw new Error(
          `label map is ${mask.shape[2]}×${mask.shape[3]} but the network output is ${t.logits.shape[2]}×${t.logits.shape[3]}`,
        );
      }
      const oh = oneHot(mask, K);
      truth = colourise(oh, K);
      loss = scope('loss', () => {
        const ce = pixelCrossEntropy(t.logits, mask.data);
        return c.diceWeight > 0 ? add(ce, mulScalar(diceLoss(t.logits, oh), c.diceWeight)) : ce;
      });
      metrics = segmentationMetrics(t.logits.value, mask.data);
    }
    const probs = softmax(t.logits, 1);
    const outputs: Record<string, Var> = {
      input: x,
      logits: t.logits,
      probs,
      prediction: constant(colourise(probs.value, K)),
      bottleneck: t.bottleneck,
      decoder: t.decoder[t.decoder.length - 1],
    };
    if (truth) outputs.truth = constant(truth);
    t.skips.forEach((v, i) => (outputs[`down${i + 1}`] = v));
    return { outputs, loss, metrics };
  },

  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 3e-3 },
    batchSize: 4,
    steps: 400,
    validateEvery: 50,
  },

  checkpoints: [
    {
      id: 'shapes2',
      label: 'Trained on generated scenes',
      path: 'weights/unet-shapes2',
      description:
        'Default configuration trained offline by scripts/train/unet.ts on 14,400 generated scenes.',
    },
  ],

  views: [
    { kind: 'image', title: 'Input tile', source: 'out:input' },
    {
      kind: 'image',
      title: 'Ground truth',
      source: 'out:truth',
      range: [0, 1],
      caption:
        'Orange: ellipse, blue: rectangle (3 classes); white: shape (2 classes); black: background.',
    },
    {
      kind: 'image',
      title: 'Predicted class probabilities',
      source: 'out:prediction',
      range: [0, 1],
      caption:
        'Each pixel is Σ_k p_k · colour_k with the ground-truth colours, so mixed colours mean uncertain pixels.',
    },
    {
      kind: 'featureMaps',
      title: 'Probability map per class (0 = background)',
      source: 'out:probs',
    },
    { kind: 'featureMaps', title: 'Bottleneck features', source: 'out:bottleneck' },
    {
      kind: 'featureMaps',
      title: 'Last decoder level (input to the 1×1 output convolution)',
      source: 'out:decoder',
    },
  ],

  math: {
    down: {
      tex: 'e_l = \\mathrm{ReLU}\\big(W_{l,2} * \\mathrm{ReLU}(W_{l,1} * \\mathrm{pool}(e_{l-1}) + b_{l,1}) + b_{l,2}\\big)',
      text: 'Contracting path: two 3×3 convolutions with ReLU at each level; the channel count doubles after every pooling. At level 1 the input image takes the place of pool(e_0). e_l is also sent across to the expanding path.',
    },
    pool: { tex: '\\mathrm{pool}(e)(y,x) = \\max_{i,j\\in\\{0,1\\}} e(2y+i,\\,2x+j)' },
    bottleneck: {
      tex: 'b = \\mathrm{conv}^2\\big(\\mathrm{pool}(e_D)\\big)',
      text: 'The lowest resolution: each unit has the largest receptive field in the network.',
    },
    upconv: {
      tex: 'u_l(2y+i,\\,2x+j) = b + \\sum_{c} W_{c,\\cdot,i,j}\\; d_{l+1,c}(y,x),\\quad i,j\\in\\{0,1\\}',
      text: '2×2 transposed convolution with stride 2: doubles the spatial size and halves the channels.',
    },
    up: {
      tex: 'd_l = \\mathrm{conv}^2\\big(\\big[\\,\\mathrm{crop}(e_l)\\;;\\;u_l\\,\\big]\\big)',
      text: 'The skip tensor e_l is concatenated along the channel axis with the up-sampled u_l (C + C = 2C channels), then two 3×3 convolutions reduce it to C. With unpadded convolutions e_l is larger than u_l and is centre-cropped first.',
    },
    'up/crop': {
      tex: '\\mathrm{crop}(e)(y,x) = e(y+o,\\,x+o),\\quad o = (H_e - H_u)/2',
      text: 'Only with unpadded convolutions; with padding, e_l and u_l already have the same size and the crop is the identity.',
    },
    output: {
      tex: 'z_k(y,x) = b_k + \\sum_c W_{k,c}\\, d_{1,c}(y,x),\\qquad p_k = \\operatorname{softmax}_k(z)',
      text: '1×1 convolution to one logit per class at every pixel.',
    },
    loss: {
      tex: 'L = -\\frac{1}{|\\Omega|}\\sum_{\\mathbf p\\in\\Omega} \\log p_{\\ell(\\mathbf p)}(\\mathbf p) \\;+\\; \\lambda\\Big(1 - \\frac{1}{K-1}\\sum_{k\\ge1}\\frac{2\\sum p_k g_k + 1}{\\sum p_k + \\sum g_k + 1}\\Big)',
      text: 'Pixel-wise cross-entropy over all pixels of the batch, plus an optional soft Dice term. The IoU metric is |pred ∩ truth| / |pred ∪ truth| of the arg-max map, averaged over the shape classes.',
    },
  },
});

export default unet;
