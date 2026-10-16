import {
  Graph,
  ParamBuilder,
  Rng,
  Tensor,
  backward,
  conv,
  declareConv,
  declareLinear,
  defaultOptimizer,
  dense,
  dropout,
  flatten,
  maxPool2d,
  relu,
  scope,
  softmax,
  crossEntropy,
  type Var,
} from '../../engine';
import { defineModel, type ForwardResult } from '../types';
import { DIGITS, MNIST_PREPROCESSING, mnistDataset } from '../data/mnist';
import { accuracy, input } from '../util';
import {
  bn,
  declareBatchNorm,
  digitCustomBatch,
  receptiveFields,
  supportExtent,
  type RfLayer,
} from './common';

export type VggConfig = {
  stages: number;
  convsPerStage: number;
  baseChannels: number;
  batchNorm: boolean;
  fcWidth: number;
  dropout: number;
};

/** Input side length: MNIST digits at their native 28×28, intensities in [0, 1]. */
export const VGG_INPUT = 28;

/** Channels of stage s (0-based): doubled at every stage, as in Table 1 of the paper. */
export const vggChannels = (c: VggConfig, s: number) => c.baseChannels * 2 ** s;

/** Spatial side after each stage's 2×2 max-pool; null if a stage would be empty. */
export function vggGeometry(c: VggConfig): number[] | null {
  const sizes: number[] = [];
  let n = VGG_INPUT;
  for (let s = 0; s < c.stages; s++) {
    n = Math.floor(n / 2);
    if (n < 1) return null;
    sizes.push(n);
  }
  return sizes;
}

/** Analytic receptive field (in input pixels) at the output of each stage, after its pooling layer. */
export function vggReceptiveFields(c: VggConfig): number[] {
  const layers: RfLayer[] = [];
  const ends: number[] = [];
  for (let s = 0; s < c.stages; s++) {
    for (let k = 0; k < c.convsPerStage; k++) layers.push({ kernel: 3, stride: 1 });
    layers.push({ kernel: 2, stride: 2 });
    ends.push(layers.length - 1);
  }
  const rf = receptiveFields(layers);
  return ends.map((i) => rf[i].size);
}

/** Convolutional stages. Returns the pre-pooling activation of each stage and the final pooled map. */
function features(x: Var, c: VggConfig): { stages: Var[]; pooled: Var } {
  const stages: Var[] = [];
  let h = x;
  for (let s = 1; s <= c.stages; s++) {
    h = scope(`stage${s}`, () => {
      let a = h;
      for (let k = 1; k <= c.convsPerStage; k++) {
        const name = `stage${s}.conv${k}`;
        a = scope(`conv${k}`, () => {
          const z = conv(a, name, { padding: 1 });
          return relu(c.batchNorm ? scope('bn', () => bn(z, `${name}.bn`)) : z);
        });
      }
      stages.push(a);
      return scope('pool', () => maxPool2d(a, 2));
    });
  }
  return { stages, pooled: h };
}

function classifier(pooled: Var, c: VggConfig): Var {
  const f1 = scope('fc1', () => dropout(relu(dense(flatten(pooled), 'fc1')), c.dropout));
  const f2 = scope('fc2', () => dropout(relu(dense(f1, 'fc2')), c.dropout));
  return scope('fc3', () => dense(f2, 'fc3'));
}

const vgg = defineModel<VggConfig>({
  id: 'vgg',
  family: 'convolutional',
  name: 'VGG',
  summary:
    'Stacks of 3×3 convolutions with ReLU, 2×2 max-pooling between stages and channel doubling, followed by a fully connected classifier.',
  signature:
    'x \\to \\big[(\\mathrm{conv}_{3\\times3} \\to \\mathrm{ReLU})^{m} \\to \\mathrm{maxpool}_{2\\times2}\\big]^{S} \\to \\mathrm{FC} \\to \\mathrm{FC} \\to \\mathrm{FC}',
  fidelity: 'reduced',
  scaleNote:
    'Published VGG-16 (configuration D): 224×224 RGB input, 5 stages of 2–3 convolutions with 64–512 channels, FC-4096, FC-4096, FC-1000, 138 M parameters. Here: 28×28 greyscale MNIST, 3 stages of 2 convolutions with 6, 12 and 24 channels, FC-128, FC-128, FC-10 (55,760 parameters at the defaults). Kernel size, padding, max-pooling, channel doubling and the three-layer head with dropout 0.5 are as published.',
  references: [
    {
      authors: 'K. Simonyan, A. Zisserman',
      title: 'Very deep convolutional networks for large-scale image recognition',
      year: 2015,
      url: 'https://arxiv.org/abs/1409.1556',
    },
    {
      authors: 'S. Ioffe, C. Szegedy',
      title:
        'Batch normalization: accelerating deep network training by reducing internal covariate shift',
      year: 2015,
      url: 'https://arxiv.org/abs/1502.03167',
    },
  ],
  limitations: [
    'Trained with Adam on the 6,000-image MNIST subset, not SGD with momentum, weight decay and learning-rate drops on ImageNet.',
    'No multi-scale training, scale jittering, horizontal flips or dense multi-crop evaluation.',
    'The batch-norm variant (VGG-BN) postdates the paper; it places batch norm between each convolution and its ReLU and removes the convolution bias, which β makes redundant.',
    'Channels double at every stage without the paper’s cap at 512.',
    'Convolutions and the first two FC layers start from He normal, the last FC layer from Glorot uniform, all biases from zero; the paper drew weights from a zero-mean Gaussian with variance 10⁻² and zero biases, and initialised its deeper configurations from a trained configuration A.',
    'The shipped checkpoint was trained offline on the full 60,000-image MNIST training set by scripts/train/vgg.ts; browser training samples the 6,000-image subset, and the validation metric uses the first 200 test images.',
  ],
  guide: {
    what: 'VGG classifies a 28×28 digit image into ten classes using only small 3×3 filters slid across the image (convolutions), two per stage so that together they cover a 5×5 region. After each stage a 2×2 max-pool keeps the largest value of every block, halving the image, while the number of filters doubles; three fully connected layers then turn the final 3×3 maps into class scores. Training learns every filter and weight, with dropout randomly silencing half of the hidden units in the fully connected layers at each training step.',
    watch:
      'This page opens with trained weights, which get 99 % of the 200 validation digits right. Press “Start from random weights” to train from scratch: with dropout on, the loss falls slowly at first and reaches about 0.4 after 300 steps of 16 digits (87 % accuracy). Watch the stage-1 feature maps change as the filters are learned.',
    try: [
      {
        label: 'One stage only',
        note: 'Six 14×14 maps feed the FC layers directly (169,000 parameters, three times the default); this run reached 94 % in 300 steps.',
        patch: { stages: 1 },
        train: 300,
      },
      {
        label: 'Two filters in stage 1',
        note: 'Stages of 2, 4 and 8 filters (28,000 parameters) learn much more slowly: 73 % after 300 steps in this run.',
        patch: { baseChannels: 2 },
        train: 300,
      },
    ],
    views: ['Stage 1, last convolution (before pooling)', 'Class probabilities'],
  },
  configSchema: [
    {
      key: 'stages',
      label: 'Stages',
      kind: 'int',
      min: 1,
      max: 4,
      affects: 'architecture',
      help: 'Each stage ends with a 2×2 max-pool that halves the spatial size (28 → 14 → 7 → 3 → 1).',
    },
    {
      key: 'convsPerStage',
      label: 'Convolutions per stage',
      kind: 'int',
      min: 1,
      max: 3,
      affects: 'architecture',
    },
    {
      key: 'baseChannels',
      label: 'Channels in stage 1',
      kind: 'int',
      min: 2,
      max: 16,
      step: 2,
      affects: 'architecture',
      help: 'Doubled at every later stage.',
    },
    { key: 'batchNorm', label: 'Batch norm (VGG-BN)', kind: 'bool', affects: 'architecture' },
    {
      key: 'fcWidth',
      label: 'Hidden FC width',
      kind: 'int',
      min: 16,
      max: 256,
      step: 16,
      affects: 'architecture',
      help: 'The paper uses 4,096.',
    },
    {
      key: 'dropout',
      label: 'Dropout in FC layers',
      kind: 'float',
      min: 0,
      max: 0.7,
      step: 0.05,
      affects: 'forward',
      help: 'The paper uses 0.5 after the first two FC layers.',
    },
  ],
  defaults: {
    stages: 3,
    convsPerStage: 2,
    baseChannels: 6,
    batchNorm: false,
    fcWidth: 128,
    dropout: 0.5,
  },

  validate(c) {
    if (!vggGeometry(c)) {
      return `${c.stages} stages of 2×2 max-pooling reduce a 28×28 input below 1×1 (28 → 14 → 7 → 3 → 1 → 0).`;
    }
    return null;
  },

  init(c, rng) {
    const b = new ParamBuilder(rng);
    let cin = 1;
    for (let s = 0; s < c.stages; s++) {
      const cout = vggChannels(c, s);
      for (let k = 1; k <= c.convsPerStage; k++) {
        const name = `stage${s + 1}.conv${k}`;
        declareConv(b, name, cin, cout, 3, { init: 'he', bias: !c.batchNorm });
        if (c.batchNorm) declareBatchNorm(b, `${name}.bn`, cout);
        cin = cout;
      }
    }
    const side = vggGeometry(c)![c.stages - 1];
    declareLinear(b, 'fc1', cin * side * side, c.fcWidth, { init: 'he' });
    declareLinear(b, 'fc2', c.fcWidth, c.fcWidth, { init: 'he' });
    declareLinear(b, 'fc3', c.fcWidth, 10, { init: 'xavier' });
    return b.store;
  },

  datasets: [mnistDataset<VggConfig>(() => 'unit', { validationSize: 200 })],

  forward(batch, c): ForwardResult {
    const x = input('image', batch.inputs.image);
    const { stages, pooled } = features(x, c);
    const logits = classifier(pooled, c);
    const loss = batch.labels
      ? scope('loss', () => crossEntropy(logits, batch.labels!))
      : undefined;
    const probs = softmax(logits, -1);
    const outputs: Record<string, Var> = { input: x, logits, probs };
    stages.forEach((v, i) => (outputs[`stage${i + 1}`] = v));
    return {
      outputs,
      loss,
      metrics: batch.labels ? { accuracy: accuracy(probs.value, batch.labels) } : undefined,
    };
  },

  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 1e-3 },
    batchSize: 16,
    steps: 300,
    validateEvery: 50,
  },

  procedures: [
    {
      id: 'receptive-field',
      label: 'Receptive field per stage',
      description:
        'For each stage, the analytic receptive field of one unit at the centre of the stage output (○), and the extent of the input pixels that actually receive a non-zero gradient ∂y/∂x for that unit on the chosen test digit (●). Max-pooling routes the gradient through one input per window and ReLU blocks inactive paths, so the measured extent is at most the analytic one.',
      args: [
        { key: 'example', label: 'Test digit', kind: 'int', min: 0, max: 23, affects: 'data' },
      ],
      defaults: { example: 0 },
      run(ctx) {
        const c = ctx.config;
        const batch = ctx.dataset.example(Number(ctx.args.example), c);
        const rf = vggReceptiveFields(c);
        const maps = new Float32Array(c.stages * VGG_INPUT * VGG_INPUT);
        const scalars: Record<string, number> = {};
        for (let s = 0; s < c.stages; s++) {
          if (ctx.shouldStop()) break;
          const g = new Graph({
            params: ctx.params,
            record: false,
            grad: true,
            training: false,
            rng: new Rng(0),
          });
          const x = g.run(() => {
            const xi = input('image', batch.inputs.image, true);
            // Pooled output of stage s + 1: run the stages up to it.
            const y = features(xi, { ...c, stages: s + 1 }).pooled;
            const [, C, H, W] = y.shape;
            const seed = Tensor.zeros(y.shape, y.value.data);
            const cy = Math.floor(H / 2);
            const cx = Math.floor(W / 2);
            for (let ch = 0; ch < C; ch++) seed.data[(ch * H + cy) * W + cx] = 1;
            if (y.requiresGrad) backward(y, seed);
            return xi;
          });
          const gd = x.grad?.data ?? new Float32Array(VGG_INPUT * VGG_INPUT);
          for (let i = 0; i < VGG_INPUT * VGG_INPUT; i++)
            maps[s * VGG_INPUT * VGG_INPUT + i] = Math.abs(gd[i]);
          scalars[`stage ${s + 1} analytic RF ○`] = rf[s];
          scalars[`stage ${s + 1} non-zero ∂y/∂x extent ●`] = supportExtent(
            gd,
            VGG_INPUT,
            VGG_INPUT,
          );
          ctx.progress((s + 1) / c.stages);
        }
        return {
          tensors: { gradient: new Tensor([c.stages, 1, VGG_INPUT, VGG_INPUT], maps) },
          scalars,
          meta: {
            caption: `|∂y/∂x| for the centre unit of each stage (summed over channels), test digit #${ctx.args.example}`,
          },
        };
      },
    },
  ],

  views: [
    { kind: 'image', title: 'Input (28×28)', source: 'out:input', range: [0, 1] },
    {
      kind: 'featureMaps',
      title: 'Stage 1, last convolution (before pooling)',
      source: 'out:stage1',
    },
    {
      kind: 'featureMaps',
      title: 'Stage 2, last convolution (before pooling)',
      source: 'out:stage2',
      when: (c) => Number(c.stages) >= 2,
    },
    {
      kind: 'featureMaps',
      title: 'Stage 3, last convolution (before pooling)',
      source: 'out:stage3',
      when: (c) => Number(c.stages) >= 3,
    },
    {
      kind: 'featureMaps',
      title: 'Stage 4, last convolution (before pooling)',
      source: 'out:stage4',
      when: (c) => Number(c.stages) >= 4,
    },
    { kind: 'probs', title: 'Class probabilities', source: 'out:probs', classNames: DIGITS },
    {
      kind: 'imageGrid',
      title: '|∂y/∂x| of the centre unit, one panel per stage',
      source: 'proc:receptive-field:gradient',
    },
  ],

  math: {
    stage: {
      tex: 'h^{(s)} = \\mathrm{maxpool}_{2\\times2}\\Big(\\mathrm{ReLU}\\big(W_m * \\cdots \\mathrm{ReLU}(W_1 * h^{(s-1)} + b_1) \\cdots + b_m\\big)\\Big)',
      text: 'A stage applies m 3×3 convolutions with padding 1 (spatial size kept), each followed by ReLU, then a 2×2 max-pool with stride 2. Two stacked 3×3 convolutions see a 5×5 region and three see 7×7, with fewer weights (2·9C² or 3·9C²) than one 5×5 (25C²) or 7×7 (49C²) layer.',
    },
    'stage/conv': {
      tex: 'a_{o}(y,x) = \\mathrm{ReLU}\\Big(b_o + \\sum_{c}\\sum_{i,j=-1}^{1} W_{o,c,i,j}\\, h_c(y+i,\\,x+j)\\Big)',
      text: '3×3 convolution with padding 1, then ReLU. With batch norm, the convolution has no bias and its output is normalised before the ReLU.',
    },
    'stage/conv/bn': {
      tex: '\\hat z = \\gamma\\,\\frac{z - \\mu_{\\mathcal B}}{\\sqrt{\\sigma^2_{\\mathcal B} + \\epsilon}} + \\beta',
      text: 'Batch statistics μ_B, σ²_B in training (running averages updated with momentum 0.1); running averages at evaluation.',
    },
    'stage/pool': { tex: 'h(y,x) = \\max_{i,j\\in\\{0,1\\}} a(2y+i,\\,2x+j)' },
    fc: {
      tex: 'h = \\mathrm{dropout}_{p}\\big(\\mathrm{ReLU}(W h_{\\text{in}} + b)\\big)',
      text: 'The last FC layer has no ReLU or dropout and produces the ten logits.',
    },
    loss: { tex: 'L = -\\frac{1}{N}\\sum_n \\log \\operatorname{softmax}(z_n)_{y_n}' },
  },

  checkpoints: [
    {
      id: 'mnist',
      label: 'Trained on MNIST',
      path: 'weights/vgg-mnist',
      description:
        'Default configuration trained offline by scripts/train/vgg.ts on the 60,000 MNIST training images.',
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

export default vgg;
