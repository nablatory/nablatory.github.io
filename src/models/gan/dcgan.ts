import {
  Graph,
  ParamBuilder,
  Rng,
  Tensor,
  conv2d,
  convTranspose2d,
  defaultOptimizer,
  flatten,
  leakyRelu,
  linear,
  p,
  relu,
  reshape,
  scope,
  tanh,
} from '../../engine';
import { defineModel, type Batch, type Config } from '../types';
import type { ParamStore } from '../../engine';
import { mnistDataset } from '../data/mnist';
import { constant, input } from '../util';
import {
  G_LOSS_OPTIONS,
  bn,
  declareBatchNorm,
  discriminatorLoss,
  ganForward,
  ganPhases,
  gaussianNoise,
  generatorLoss,
  meanProb,
  type GanParts,
  type GeneratorLoss,
} from './common';

export type DcganConfig = {
  zDim: number;
  gf: number;
  df: number;
  gLoss: GeneratorLoss;
};

/** Generated images shown by the forward pass when not training. */
const INSPECT_SAMPLES = 16;

/*
 * Generator   z ─ project ─▶ 2g@7×7 ─ deconv1 ─▶ g@14×14 ─ deconv2 ─▶ 1@28×28
 *             (linear, BN, ReLU)   (4×4 stride 2, BN, ReLU)  (4×4 stride 2, tanh)
 * Discriminator x ─ conv1 ─▶ d@14×14 ─ conv2 ─▶ 2d@7×7 ─ out ─▶ logit
 *             (4×4 stride 2, LReLU)  (4×4 stride 2, BN, LReLU) (linear)
 */
export const dcganParts: GanParts<DcganConfig> = {
  real: (batch) => input('image', batch.inputs.image),
  noise: (n, c) => gaussianNoise(n, c.zDim),
  generate: (z, c) =>
    scope('G', () => {
      const N = z.shape[0];
      const h0 = scope('project', () =>
        relu(bn(reshape(linear(z, p('G.project.weight')), [N, 2 * c.gf, 7, 7]), 'G.project.bn')),
      );
      const h1 = scope('deconv1', () =>
        relu(
          bn(
            convTranspose2d(h0, p('G.deconv1.weight'), undefined, { stride: 2, padding: 1 }),
            'G.deconv1.bn',
          ),
        ),
      );
      return scope('deconv2', () =>
        tanh(
          convTranspose2d(h1, p('G.deconv2.weight'), p('G.deconv2.bias'), {
            stride: 2,
            padding: 1,
          }),
        ),
      );
    }),
  discriminate: (x, _c, name) =>
    scope(name, () => {
      const h1 = scope('conv1', () =>
        leakyRelu(
          conv2d(x, p('D.conv1.weight'), p('D.conv1.bias'), { stride: 2, padding: 1 }),
          0.2,
        ),
      );
      const h2 = scope('conv2', () =>
        leakyRelu(
          bn(conv2d(h1, p('D.conv2.weight'), undefined, { stride: 2, padding: 1 }), 'D.conv2.bn'),
          0.2,
        ),
      );
      return scope('out', () => linear(flatten(h2), p('D.out.weight'), p('D.out.bias')));
    }),
};

/** G(z) in evaluation mode (batch norm with running statistics) for [n, zDim] noise. */
function generateFrom(z: Float32Array, n: number, c: DcganConfig): Tensor {
  return dcganParts.generate(constant(new Tensor([n, c.zDim], z), 'z'), c).value;
}

/**
 * Evaluation-mode measurements on a batch of real images and n generated
 * images from noise drawn with Rng(seed): mean D outputs and both losses.
 * Used by the training script and the checkpoint test.
 */
export function dcganEvaluate(
  params: ParamStore,
  config: Config,
  real: Batch,
  seed: number,
  n: number,
) {
  const c = config as DcganConfig;
  const rng = new Rng(seed);
  const z = new Float32Array(n * c.zDim);
  for (let i = 0; i < z.length; i++) z[i] = rng.normal();
  const g = new Graph({ params, record: false, grad: false, training: false, rng });
  return g.run(() => {
    const fake = generateFrom(z, n, c);
    const dReal = dcganParts.discriminate(input('image', real.inputs.image, false), c, 'D_real');
    const dFake = dcganParts.discriminate(constant(fake), c, 'D_fake');
    return {
      samples: fake,
      metrics: {
        dLoss: discriminatorLoss(dReal, dFake).value.data[0],
        gLoss: generatorLoss(dFake, 'nonsaturating').value.data[0],
        meanDReal: meanProb(dReal),
        meanDFake: meanProb(dFake),
      },
    };
  });
}

const dcgan = defineModel<DcganConfig>({
  id: 'dcgan',
  family: 'gan',
  name: 'DCGAN',
  summary:
    'A GAN whose generator is a stack of transposed convolutions with batch normalisation and whose discriminator is a strided convolutional network, here generating 28 × 28 digits.',
  signature:
    'G: z \\xrightarrow{\\text{proj}} 2g@7^2 \\xrightarrow{\\text{deconv}} g@14^2 \\xrightarrow{\\text{deconv}} \\tanh \\to 1@28^2,\\quad D: x \\xrightarrow{\\text{conv}} d@14^2 \\xrightarrow{\\text{conv}} 2d@7^2 \\to \\sigma',
  fidelity: 'reduced',
  scaleNote:
    'Follows the architecture guidelines of Radford et al. (2016): strided (transposed) convolutions instead of pooling, batch normalisation in both networks except the generator output and discriminator input, ReLU in G with tanh output, LeakyReLU(0.2) in D, weights from N(0, 0.02²), Adam with learning rate 2·10⁻⁴ and β₁ = 0.5. The paper generated 64 × 64 images with 1024→512→256→128 generator channels; here the output is 28 × 28 MNIST in [−1, 1] with g = 16 and d = 16 base channels by default, two (transposed) convolutions per network, and browser batches of 16 instead of 128 (the shipped checkpoint used 32).',
  references: [
    {
      authors: 'A. Radford, L. Metz, S. Chintala',
      title:
        'Unsupervised representation learning with deep convolutional generative adversarial networks',
      year: 2016,
      url: 'https://arxiv.org/abs/1511.06434',
    },
    {
      authors:
        'I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio',
      title: 'Generative adversarial nets',
      year: 2014,
      url: 'https://arxiv.org/abs/1406.2661',
    },
  ],
  limitations: [
    'Two upsampling stages instead of four, with far fewer channels, so samples are coarser than in the paper.',
    'The noise is z ~ N(0, I) with 32 dimensions by default; the paper drew a 100-dimensional z from a uniform distribution.',
    'Batch normalisation uses batch statistics during training and running averages (momentum 0.1) when sampling; generated images can differ slightly between the two modes.',
    'The discriminator sees real and generated batches in separate passes, so its batch statistics are never mixed, as in the reference PyTorch implementation.',
    'No sample-quality score such as FID is computed; the training script records the losses and discriminator outputs it measured.',
    'Browser training uses the 6,000-image MNIST subset; the shipped checkpoint was trained on the full 60,000-image training set.',
  ],
  guide: {
    what: 'The same two-player game as the GAN, now with images. The generator maps a 32-number noise vector through a linear layer and two transposed convolutions (learned upsampling) to a 28×28 digit, and the discriminator is a two-layer convolutional network that scores whether an image is a real MNIST digit. Both networks use batch normalisation, as in the paper. After training, new digits are produced by drawing fresh noise; the 64 samples from the checkpoint differ from one another about as much as 64 real test digits do.',
    watch:
      'This page opens with a trained checkpoint, so the generated grid already shows digits; press “Start from random weights” to train from scratch. The samples then start as one flat grey image, the same for every z, turn into dark, nearly empty frames by step 100, and grow strokes and digit-like shapes over the default 600 steps, still less varied than the checkpoint’s. D(x) climbs above D(G(z)) within 20 steps and the discriminator then wins clearly (D(x) about 0.98, D(G(z)) about 0.02 near step 200); the gap narrows only slowly afterwards (about 0.9 against 0.1 at step 600). Trained on from the checkpoint, the two stay near 0.67 and 0.33.',
    try: [
      {
        label: 'Minimax generator loss',
        note: 'Continues from the checkpoint. Generator gradients now scale with D(G(z)) instead of 1 − D(G(z)); with D(G(z)) near 0.33 here, they are about half as large. Over 200 steps D(x) and D(G(z)) stay near 0.67 and 0.33 and the samples do not change visibly, just as with the default loss.',
        patch: { gLoss: 'minimax' },
        train: 200,
      },
      {
        label: 'Narrow generator',
        note: 'Re-initialises. With four base channels the discriminator wins at once: D(x) passes 0.9 within about 30 steps, and D(G(z)) falls below 0.05 by step 100 and stays there. After 400 steps the samples are still dark, nearly empty frames with faint marks, where the default width already draws strokes.',
        patch: { gf: 4 },
        train: 400,
      },
    ],
  },
  configSchema: [
    {
      key: 'zDim',
      label: 'Noise dimension',
      kind: 'int',
      min: 2,
      max: 128,
      affects: 'architecture',
    },
    {
      key: 'gf',
      label: 'Generator base channels g',
      kind: 'int',
      min: 4,
      max: 64,
      step: 4,
      affects: 'architecture',
    },
    {
      key: 'df',
      label: 'Discriminator base channels d',
      kind: 'int',
      min: 4,
      max: 64,
      step: 4,
      affects: 'architecture',
    },
    {
      key: 'gLoss',
      label: 'Generator loss',
      kind: 'select',
      options: G_LOSS_OPTIONS,
      affects: 'forward',
    },
  ],
  defaults: { zDim: 32, gf: 16, df: 16, gLoss: 'nonsaturating' },

  init(c, rng) {
    const b = new ParamBuilder(rng);
    const w = { kind: 'normal' as const, std: 0.02 };
    b.add('G.project.weight', [2 * c.gf * 49, c.zDim], w);
    declareBatchNorm(b, 'G.project.bn', 2 * c.gf);
    b.add('G.deconv1.weight', [2 * c.gf, c.gf, 4, 4], w);
    declareBatchNorm(b, 'G.deconv1.bn', c.gf);
    b.add('G.deconv2.weight', [c.gf, 1, 4, 4], w);
    b.add('G.deconv2.bias', [1], { kind: 'zeros' });
    b.add('D.conv1.weight', [c.df, 1, 4, 4], w);
    b.add('D.conv1.bias', [c.df], { kind: 'zeros' });
    b.add('D.conv2.weight', [2 * c.df, c.df, 4, 4], w);
    declareBatchNorm(b, 'D.conv2.bn', 2 * c.df);
    b.add('D.out.weight', [1, 2 * c.df * 49], w);
    b.add('D.out.bias', [1], { kind: 'zeros' });
    return b.store;
  },

  datasets: [mnistDataset<DcganConfig>(() => 'symmetric')],

  forward(batch, c, ctx) {
    const r = ganForward(dcganParts, batch, c, ctx, ctx.training ? undefined : INSPECT_SAMPLES);
    return {
      outputs: { real: r.x, fake: r.fake, dReal: r.dReal, dFake: r.dFake, gLoss: r.gLoss },
      loss: r.dLoss,
      metrics: {
        dLoss: r.dLoss.value.data[0],
        gLoss: r.gLoss.value.data[0],
        'D(x)': meanProb(r.dReal),
        'D(G(z))': meanProb(r.dFake),
      },
    };
  },

  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 2e-4, beta1: 0.5, beta2: 0.999 },
    batchSize: 16,
    steps: 600,
    validateEvery: 100,
    phases: ganPhases(dcganParts),
  },

  procedures: [
    {
      id: 'samples',
      label: 'Samples from fixed noise',
      description:
        'Generates 64 digits from 64 noise vectors determined by the noise seed alone, so the same z can be followed through training.',
      args: [
        {
          key: 'noiseSeed',
          label: 'Noise seed',
          kind: 'int',
          min: 0,
          max: 9999,
          affects: 'forward',
        },
      ],
      defaults: { noiseSeed: 0 },
      run({ config, args }) {
        const n = 64;
        const rng = new Rng(Number(args.noiseSeed ?? 0)).fork('dcgan-fixed-noise');
        const z = new Float32Array(n * config.zDim);
        for (let i = 0; i < z.length; i++) z[i] = rng.normal();
        return { tensors: { samples: generateFrom(z, n, config) } };
      },
    },
    {
      id: 'interpolate',
      label: 'Latent interpolation',
      description:
        'Each row decodes 10 points on the straight line between two random noise vectors z₀ and z₁ (Radford et al., section 6.1).',
      args: [{ key: 'rows', label: 'Rows', kind: 'int', min: 1, max: 8, affects: 'forward' }],
      defaults: { rows: 6 },
      run({ config, rng, args }) {
        const rows = Math.max(1, Math.min(8, Number(args.rows ?? 6)));
        const cols = 10;
        const d = config.zDim;
        const z = new Float32Array(rows * cols * d);
        for (let r = 0; r < rows; r++) {
          const z0 = Array.from({ length: d }, () => rng.normal());
          const z1 = Array.from({ length: d }, () => rng.normal());
          for (let k = 0; k < cols; k++) {
            const t = k / (cols - 1);
            for (let j = 0; j < d; j++) z[(r * cols + k) * d + j] = (1 - t) * z0[j] + t * z1[j];
          }
        }
        return { tensors: { grid: generateFrom(z, rows * cols, config) } };
      },
    },
  ],

  views: [
    { kind: 'image', title: 'Real digit x', source: 'out:real', range: [-1, 1] },
    {
      kind: 'imageGrid',
      title: 'Generated G(z), fresh noise each inspection',
      source: 'out:fake',
      range: [-1, 1],
      columns: 8,
    },
    {
      kind: 'imageGrid',
      title: 'Samples from fixed noise',
      source: 'proc:samples:samples',
      range: [-1, 1],
      columns: 8,
    },
    {
      kind: 'imageGrid',
      title: 'Interpolation between noise vectors',
      source: 'proc:interpolate:grid',
      range: [-1, 1],
      columns: 10,
    },
  ],

  checkpoints: [
    {
      id: 'mnist',
      label: 'Trained on MNIST',
      path: 'weights/dcgan-mnist',
      description:
        'Default configuration trained for 2,500 alternating steps with batches of 32 (1.3 epochs of the 60,000 training images) by scripts/train/dcgan-mnist.ts.',
    },
  ],

  math: {
    G: {
      tex: 'G(z) = \\tanh\\big(W_2 \\star^{\\top}_{2}\\, \\mathrm{relu}(\\mathrm{BN}(W_1 \\star^{\\top}_{2}\\, \\mathrm{relu}(\\mathrm{BN}(P z))))\\big)',
      text: 'A linear projection of z reshaped to 7 × 7 maps, then two 4 × 4 transposed convolutions with stride 2 (7 → 14 → 28).',
    },
    project: {
      tex: 'h_0 = \\mathrm{relu}\\big(\\mathrm{BN}(\\operatorname{reshape}(P z))\\big),\\; P \\in \\mathbb R^{(2g\\cdot 49) \\times d_z}',
    },
    deconv: {
      tex: 'y = W \\star^{\\top}_{2} h',
      text: 'Transposed convolution with stride 2: the adjoint of a strided convolution, doubling the spatial size.',
    },
    D_real: {
      tex: 'D(x) = \\sigma\\big(w^{\\top} \\operatorname{vec}(\\mathrm{lrelu}(\\mathrm{BN}(W_2 \\star_2 \\mathrm{lrelu}(W_1 \\star_2 x)))) + b\\big)',
      text: 'Strided convolutions halve the size twice (28 → 14 → 7); the last feature maps are flattened into a single logit.',
    },
    D_fake: {
      tex: 'D(G(z))',
      text: 'The same discriminator on a separate batch of generated images.',
    },
    conv: { tex: 'y = \\mathrm{lrelu}_{0.2}(W \\star_2 h)' },
    loss_D: {
      tex: 'L_D = -\\tfrac1m \\textstyle\\sum_i \\big[\\log D(x_i) + \\log(1 - D(G(z_i)))\\big]',
    },
    loss_G: {
      tex: 'L_G = -\\tfrac1m \\textstyle\\sum_i \\log D(G(z_i))',
      text: 'Non-saturating generator loss, as in the paper; the minimax form is selectable.',
    },
  },
});

export default dcgan;
