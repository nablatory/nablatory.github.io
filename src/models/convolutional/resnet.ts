import {
  Graph,
  ParamBuilder,
  Rng,
  Tensor,
  add,
  backward,
  conv,
  crossEntropy,
  declareConv,
  declareLinear,
  defaultOptimizer,
  dense,
  mean,
  pad,
  relu,
  scope,
  softmax,
  type Var,
} from '../../engine';
import { defineModel, type ForwardResult } from '../types';
import { DIGITS, MNIST_PREPROCESSING, mnistDataset } from '../data/mnist';
import { accuracy, constant, input } from '../util';
import { bn, declareBatchNorm, digitCustomBatch, l2, rms } from './common';
import { subsample2d } from './ops';

export type ResNetConfig = {
  stages: number;
  blocksPerStage: number;
  width: number;
  shortcut: 'A' | 'B';
  shortcuts: boolean;
  preActivation: boolean;
};

export const RESNET_INPUT = 28;

export interface BlockSpec {
  index: number;
  stage: number;
  cin: number;
  cout: number;
  stride: number;
  /** Shortcut kind used when shortcuts are on. */
  shortcut: 'identity' | 'A' | 'B';
}

/** Basic blocks in execution order; stage s ≥ 2 starts with a stride-2 block that doubles the width. */
export function resnetBlocks(c: ResNetConfig): BlockSpec[] {
  const out: BlockSpec[] = [];
  let cin = c.width;
  let k = 1;
  for (let s = 0; s < c.stages; s++) {
    const cout = c.width * 2 ** s;
    for (let j = 0; j < c.blocksPerStage; j++) {
      const stride = s > 0 && j === 0 ? 2 : 1;
      const same = stride === 1 && cin === cout;
      out.push({
        index: k++,
        stage: s + 1,
        cin,
        cout,
        stride,
        shortcut: same ? 'identity' : c.shortcut,
      });
      cin = cout;
    }
  }
  return out;
}

/** Spatial side per stage (3×3, padding 1, stride 2 halves with rounding up); null if empty. */
export function resnetGeometry(c: ResNetConfig): number[] | null {
  const sizes: number[] = [];
  let n = RESNET_INPUT;
  for (let s = 0; s < c.stages; s++) {
    if (s > 0) n = Math.floor((n - 1) / 2) + 1;
    if (n < 1) return null;
    sizes.push(n);
  }
  return sizes;
}

export interface ResNetTrace {
  logits: Var;
  /** x_l: the input of every block. */
  blockInputs: Var[];
  /** F(x_l): the residual branch of every block (before the addition). */
  residuals: Var[];
  /** The shortcut term added to F(x_l); null when shortcuts are off. */
  shortcuts: (Var | null)[];
  blockOutputs: Var[];
}

/** The whole network. `shortcuts` is taken from the config; parameters are identical either way. */
export function resnetNetwork(x: Var, c: ResNetConfig): ResNetTrace {
  const pre = c.preActivation;
  let h = scope('stem', () => {
    const z = conv(x, 'stem.conv', { padding: 1 });
    return pre ? z : relu(bn(z, 'stem.bn'));
  });
  const t: ResNetTrace = {
    logits: h,
    blockInputs: [],
    residuals: [],
    shortcuts: [],
    blockOutputs: [],
  };
  for (const b of resnetBlocks(c)) {
    const xin = h;
    const F = `F${b.index}`;
    const f = scope(F, () => {
      if (pre) {
        // BN → ReLU → conv → BN → ReLU → conv (He et al., 2016b)
        const a1 = relu(bn(xin, `${F}.bn1`));
        const z1 = conv(a1, `${F}.conv1`, { stride: b.stride, padding: 1 });
        const a2 = relu(bn(z1, `${F}.bn2`));
        return conv(a2, `${F}.conv2`, { padding: 1 });
      }
      // conv → BN → ReLU → conv → BN (He et al., 2016a)
      const z1 = relu(bn(conv(xin, `${F}.conv1`, { stride: b.stride, padding: 1 }), `${F}.bn1`));
      return bn(conv(z1, `${F}.conv2`, { padding: 1 }), `${F}.bn2`);
    });
    let s: Var | null = null;
    if (c.shortcuts) {
      const S = `shortcut${b.index}`;
      if (b.shortcut === 'identity') s = xin;
      else if (b.shortcut === 'A')
        s = scope(S, () => {
          // Option A: parameter-free; subsample by the stride and zero-pad the new channels.
          const sub = b.stride > 1 ? subsample2d(xin, b.stride) : xin;
          return b.cout > b.cin
            ? pad(sub, [
                [0, 0],
                [0, b.cout - b.cin],
                [0, 0],
                [0, 0],
              ])
            : sub;
        });
      else
        s = scope(S, () => {
          // Option B: 1×1 projection W_s with the block's stride.
          const z = conv(xin, `${S}.conv`, { stride: b.stride });
          return pre ? z : bn(z, `${S}.bn`);
        });
    }
    const y = scope(`add${b.index}`, () => {
      const sum = s ? add(f, s) : f;
      return pre ? sum : relu(sum);
    });
    t.blockInputs.push(xin);
    t.residuals.push(f);
    t.shortcuts.push(s);
    t.blockOutputs.push(y);
    h = y;
  }
  if (pre) h = scope('post', () => relu(bn(h, 'post.bn')));
  const pooled = scope('pool', () => mean(h, [2, 3]));
  t.logits = scope('fc', () => dense(pooled, 'fc'));
  return t;
}

const resnet = defineModel<ResNetConfig>({
  id: 'resnet',
  family: 'convolutional',
  name: 'ResNet',
  summary:
    'Residual network: each block learns a residual F(x) that is added to an identity (or projection) shortcut, so information and gradients can bypass the convolutions.',
  signature:
    'x_{l+1} = \\mathrm{ReLU}\\big(\\mathcal F(x_l;\\,W_l) + W_s\\,x_l\\big),\\quad \\mathcal F = \\mathrm{BN}\\,W_2\\,\\mathrm{ReLU}(\\mathrm{BN}\\,W_1 x_l)',
  fidelity: 'reduced',
  scaleNote:
    'Follows the CIFAR-10 ResNets of the paper (Section 4.2): 3×3 stem, three stages of basic blocks with widths w, 2w, 4w and strides 1, 2, 2, global average pooling and one FC layer, with option A shortcuts. Published: w = 16 and n = 3–18 blocks per stage (ResNet-20 to ResNet-110, 0.27–1.7 M parameters) on 32×32 RGB images. Here: w = 6 and n = 1 at the defaults (8 weighted layers, 10,852 trainable parameters) on 28×28 MNIST.',
  references: [
    {
      authors: 'K. He, X. Zhang, S. Ren, J. Sun',
      title: 'Deep residual learning for image recognition',
      year: 2016,
      url: 'https://arxiv.org/abs/1512.03385',
    },
    {
      authors: 'K. He, X. Zhang, S. Ren, J. Sun',
      title: 'Identity mappings in deep residual networks',
      year: 2016,
      url: 'https://arxiv.org/abs/1603.05027',
    },
  ],
  limitations: [
    'Trained with Adam on the 6,000-image MNIST subset, not SGD with momentum 0.9, weight decay 10⁻⁴ and learning-rate drops on CIFAR-10 or ImageNet.',
    'No data augmentation (the paper pads by 4 and takes random 32×32 crops and flips).',
    'Option A appends the zero channels after the existing ones rather than splitting them on both sides; this only permutes channels.',
    'In the pre-activation variant the stem is a bare convolution and every block, including the first, applies its BN–ReLU inside the residual branch only, so identity and projection shortcuts read the raw block input x. He et al. (2016b) place the first BN–ReLU right after the stem, before the split, and let projection shortcuts read the pre-activated signal.',
    'Only basic two-convolution blocks; the bottleneck blocks used for ResNet-50 and deeper are not included.',
    'The shipped checkpoint was trained offline on the full 60,000-image MNIST training set by scripts/train/resnet.ts; browser training samples the 6,000-image subset, and the validation metric uses the first 200 test images.',
  ],
  guide: {
    what: 'A residual network stacks blocks of two 3×3 convolutions whose output F(x) is added to the block’s input through a shortcut, so each block learns a correction. This one reads a 28×28 digit, passes it through a stem convolution and three blocks (6, 12 and 24 channels, halving the image twice), averages each final map to one number and classifies with a linear layer. Batch normalisation (rescaling each channel by the batch’s mean and spread) follows each convolution; training learns the filters, normalisation scales and final layer.',
    watch:
      'This page opens with trained weights, which get 99.5 % of the 200 validation digits right. Press “Start from random weights” to train from scratch: the loss falls from 2.3 to about 0.4 in 300 steps of 16 digits (90 % accuracy). Watch block 1’s residual branch F(x) next to its shortcut, which here is the input itself.',
    try: [
      {
        label: 'Pre-activation blocks',
        note: 'BN and ReLU move in front of each convolution and nothing follows the addition; trains about as well (92 % in this run).',
        patch: { preActivation: true },
        train: 300,
      },
      {
        label: 'Three blocks per stage',
        note: '20 weighted layers and 38,000 parameters; each step takes about three times longer and this run reached 87 %.',
        patch: { blocksPerStage: 3 },
        train: 300,
      },
      {
        label: 'Deep and no shortcuts',
        note: 'At this depth the plain network still learns (96 % in this run); run “Gradient at each block input, with and without shortcuts” to compare the two.',
        patch: { blocksPerStage: 3, shortcuts: false },
        train: 300,
      },
    ],
    views: ['Block 1: residual branch F(x)', 'Class probabilities'],
  },
  configSchema: [
    {
      key: 'stages',
      label: 'Stages',
      kind: 'int',
      min: 1,
      max: 3,
      affects: 'architecture',
      help: 'Widths w, 2w, 4w; every stage after the first starts with stride 2 (28 → 14 → 7).',
    },
    {
      key: 'blocksPerStage',
      label: 'Blocks per stage (n)',
      kind: 'int',
      min: 1,
      max: 3,
      affects: 'architecture',
      help: 'Depth 2·n·stages + 2 weighted layers.',
    },
    {
      key: 'width',
      label: 'Base width w',
      kind: 'int',
      min: 2,
      max: 16,
      step: 2,
      affects: 'architecture',
      help: 'The paper uses 16 on CIFAR-10.',
    },
    {
      key: 'shortcut',
      label: 'Shortcut when shape changes',
      kind: 'select',
      affects: 'architecture',
      options: [
        { value: 'A', label: 'A: subsample + zero-pad' },
        { value: 'B', label: 'B: 1×1 projection' },
      ],
    },
    {
      key: 'shortcuts',
      label: 'Shortcuts',
      kind: 'bool',
      affects: 'forward',
      help: 'Off: x_{l+1} = ReLU(F(x_l)), a plain network with the same parameters.',
    },
    {
      key: 'preActivation',
      label: 'Pre-activation blocks',
      kind: 'bool',
      affects: 'architecture',
      help: 'x_{l+1} = x_l + F(x_l) with F = W₂ ReLU(BN(W₁ ReLU(BN(x_l)))) (He et al., 2016b).',
    },
  ],
  defaults: {
    stages: 3,
    blocksPerStage: 1,
    width: 6,
    shortcut: 'A',
    shortcuts: true,
    preActivation: false,
  },

  validate(c) {
    if (!resnetGeometry(c))
      return `${c.stages} stride-2 stages leave no spatial extent from a 28×28 input.`;
    return null;
  },

  init(c, rng) {
    const b = new ParamBuilder(rng);
    const pre = c.preActivation;
    declareConv(b, 'stem.conv', 1, c.width, 3, { init: 'he', bias: false });
    if (!pre) declareBatchNorm(b, 'stem.bn', c.width);
    let last = c.width;
    for (const blk of resnetBlocks(c)) {
      const F = `F${blk.index}`;
      if (pre) declareBatchNorm(b, `${F}.bn1`, blk.cin);
      declareConv(b, `${F}.conv1`, blk.cin, blk.cout, 3, { init: 'he', bias: false });
      declareBatchNorm(b, pre ? `${F}.bn2` : `${F}.bn1`, blk.cout);
      declareConv(b, `${F}.conv2`, blk.cout, blk.cout, 3, { init: 'he', bias: false });
      if (!pre) declareBatchNorm(b, `${F}.bn2`, blk.cout);
      if (blk.shortcut === 'B') {
        const S = `shortcut${blk.index}`;
        declareConv(b, `${S}.conv`, blk.cin, blk.cout, 1, { init: 'he', bias: false });
        if (!pre) declareBatchNorm(b, `${S}.bn`, blk.cout);
      }
      last = blk.cout;
    }
    if (pre) declareBatchNorm(b, 'post.bn', last);
    declareLinear(b, 'fc', last, 10);
    return b.store;
  },

  datasets: [mnistDataset<ResNetConfig>(() => 'unit', { validationSize: 200 })],

  forward(batch, c): ForwardResult {
    const x = input('image', batch.inputs.image);
    const t = resnetNetwork(x, c);
    const loss = batch.labels
      ? scope('loss', () => crossEntropy(t.logits, batch.labels!))
      : undefined;
    const probs = softmax(t.logits, -1);
    const L = t.residuals.length;
    const normF = new Tensor(
      [L],
      t.residuals.map((v) => rms(v.value)),
    );
    const normX = new Tensor(
      [L],
      t.blockInputs.map((v) => rms(v.value)),
    );
    const outputs: Record<string, Var> = {
      input: x,
      logits: t.logits,
      probs,
      lastBlock: t.blockOutputs[L - 1],
      normF: constant(normF),
      normX: constant(normX),
    };
    t.residuals.forEach((v, i) => (outputs[`F${i + 1}`] = v));
    t.shortcuts.forEach((v, i) => v && (outputs[`shortcut${i + 1}`] = v));
    t.blockOutputs.forEach((v, i) => (outputs[`block${i + 1}`] = v));
    return {
      outputs,
      loss,
      metrics: batch.labels ? { accuracy: accuracy(probs.value, batch.labels) } : undefined,
    };
  },

  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 5e-3 },
    batchSize: 16,
    steps: 300,
    validateEvery: 50,
  },

  procedures: [
    {
      id: 'gradient-norms',
      label: 'Gradient at each block input, with and without shortcuts',
      description:
        'Draws one training batch, runs forward and backward passes with the current parameters twice — once with shortcuts, once as the plain network — and reports ‖∂L/∂x_l‖ at the input x_l of every block. Batch norm uses batch statistics, as in training, on a copy of the parameters so the running averages are not changed.',
      args: [
        {
          key: 'batch',
          label: 'Batch size',
          kind: 'int',
          min: 4,
          max: 64,
          step: 4,
          affects: 'data',
        },
      ],
      defaults: { batch: 32 },
      run(ctx) {
        const c = ctx.config;
        const batch = ctx.dataset.train(ctx.rng, Number(ctx.args.batch), c);
        const res: Record<string, Tensor> = {};
        const scalars: Record<string, number> = {};
        const modes: [string, boolean][] = [
          ['withShortcuts', true],
          ['plain', false],
        ];
        modes.forEach(([name, on], i) => {
          if (ctx.shouldStop()) return;
          const store = ctx.params.clone();
          const g = new Graph({
            params: store,
            record: false,
            grad: true,
            training: true,
            rng: new Rng(0),
          });
          const { loss, t } = g.run(() => {
            const xi = input('image', batch.inputs.image, true);
            const t = resnetNetwork(xi, { ...c, shortcuts: on });
            return { loss: crossEntropy(t.logits, batch.labels!), t };
          });
          backward(loss);
          const norms = t.blockInputs.map((v) => l2(v.grad));
          res[name] = new Tensor([norms.length], norms);
          scalars[`loss (${on ? 'shortcuts' : 'plain'})`] = loss.value.data[0];
          scalars[`‖∂L/∂x₁‖ / ‖∂L/∂x_L‖ (${on ? 'shortcuts' : 'plain'})`] =
            norms[0] / Math.max(norms[norms.length - 1], 1e-30);
          ctx.progress((i + 1) / modes.length);
        });
        return {
          tensors: res,
          scalars,
          meta: { caption: 'Block index on the horizontal axis (0 = first block).' },
        };
      },
    },
  ],

  views: [
    { kind: 'image', title: 'Input (28×28)', source: 'out:input', range: [0, 1] },
    { kind: 'featureMaps', title: 'Block 1: residual branch F(x)', source: 'out:F1' },
    {
      kind: 'featureMaps',
      title: 'Block 1: shortcut',
      source: 'out:shortcut1',
      when: (c) => Boolean(c.shortcuts),
    },
    { kind: 'featureMaps', title: 'Last block output', source: 'out:lastBlock' },
    {
      kind: 'series',
      title: 'RMS of F(x_l) and of the block input x_l',
      xLabel: 'block',
      sources: [
        { ref: 'out:normF', label: 'rms F(x_l)' },
        { ref: 'out:normX', label: 'rms x_l' },
      ],
    },
    { kind: 'probs', title: 'Class probabilities', source: 'out:probs', classNames: DIGITS },
    {
      kind: 'series',
      title: '‖∂L/∂x_l‖ at each block input',
      xLabel: 'block',
      sources: [
        { ref: 'proc:gradient-norms:withShortcuts', label: 'with shortcuts' },
        { ref: 'proc:gradient-norms:plain', label: 'plain' },
      ],
    },
  ],

  math: {
    stem: {
      tex: 'x_1 = \\mathrm{ReLU}(\\mathrm{BN}(W_0 * x))',
      text: 'One 3×3 convolution to w channels. In the pre-activation variant the BN and ReLU move into the first block.',
    },
    F: {
      tex: '\\mathcal F(x_l) = \\mathrm{BN}\\big(W_2 * \\mathrm{ReLU}(\\mathrm{BN}(W_1 *_{s} x_l))\\big)',
      text: 'Residual branch: two 3×3 convolutions, the first with the block stride s. Pre-activation: F(x) = W₂ * ReLU(BN(W₁ * ReLU(BN(x)))).',
    },
    shortcut: {
      tex: '\\text{A: } (S x)_{c} = \\begin{cases} x_{c}(s y, s x) & c < C_{\\text{in}} \\\\ 0 & \\text{otherwise}\\end{cases}\\qquad \\text{B: } S x = \\mathrm{BN}(W_s *_{s} x)',
      text: 'Used only where the block changes the shape; elsewhere the shortcut is the identity and adds no operations. Option A has no parameters. In the pre-activation variant option B is the bare projection W_s x, without BN.',
    },
    add: {
      tex: 'x_{l+1} = \\mathrm{ReLU}\\big(\\mathcal F(x_l) + S x_l\\big)',
      text: 'Since ∂x_{l+1}/∂x_l contains the term S (the identity for most blocks), the gradient reaching earlier blocks does not have to pass through the convolutions. Pre-activation: x_{l+1} = F(x_l) + S x_l with no ReLU after the sum. With shortcuts off, x_{l+1} = ReLU(F(x_l)).',
    },
    post: { tex: 'h = \\mathrm{ReLU}(\\mathrm{BN}(x_L))' },
    pool: { tex: 'g_c = \\frac{1}{HW}\\sum_{y,x} h_c(y,x)' },
    fc: { tex: 'z = W g + b' },
    loss: { tex: 'L = -\\frac{1}{N}\\sum_n \\log \\operatorname{softmax}(z_n)_{y_n}' },
  },

  checkpoints: [
    {
      id: 'mnist',
      label: 'Trained on MNIST',
      path: 'weights/resnet-mnist',
      description:
        'Default configuration trained offline by scripts/train/resnet.ts on the 60,000 MNIST training images.',
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

  buildCustomBatch(t) {
    return digitCustomBatch(t, 'unit', 28);
  },
});

export default resnet;
