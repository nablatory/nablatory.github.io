import {
  Graph,
  ParamBuilder,
  Tensor,
  activeGraph,
  dense,
  declareLinear,
  defaultOptimizer,
  leakyRelu,
  relu,
  scope,
  sigmoidFn,
  type Var,
} from '../../engine';
import { defineModel } from '../types';
import { constant, input } from '../util';
import {
  TOY2D_EXTENT,
  TOY2D_OPTIONS,
  modeCoverage,
  toy2dDataset,
  toy2dGrid,
  toy2dSample,
  type Toy2dKind,
} from '../data/toy2d';
import {
  G_LOSS_OPTIONS,
  ganForward,
  ganPhases,
  gaussianNoise,
  meanProb,
  type GanParts,
  type GeneratorLoss,
} from './common';

type GanConfig = {
  dataset: Toy2dKind;
  zDim: number;
  width: number;
  gLoss: GeneratorLoss;
};

const FIELD_RES = 64;

const parts: GanParts<GanConfig> = {
  real: (batch) => input('x', batch.inputs.x),
  noise: (n, c) => gaussianNoise(n, c.zDim),
  generate: (z) =>
    scope('G', () => {
      const h1 = scope('fc1', () => relu(dense(z, 'G.fc1')));
      const h2 = scope('fc2', () => relu(dense(h1, 'G.fc2')));
      return scope('out', () => dense(h2, 'G.out'));
    }),
  discriminate: (x, _c, name) =>
    scope(name, () => {
      const h1 = scope('fc1', () => leakyRelu(dense(x, 'D.fc1'), 0.2));
      const h2 = scope('fc2', () => leakyRelu(dense(h1, 'D.fc2'), 0.2));
      return scope('out', () => dense(h2, 'D.out'));
    }),
};

/**
 * D(x) = σ(logit) on a 64 × 64 grid over the plane, evaluated in a separate
 * non-recording graph so that the 4,096 grid points do not enter the trace.
 */
function discriminatorField(c: GanConfig): Tensor {
  const params = activeGraph()?.params;
  const g = new Graph({ params, record: false, grad: false, training: false });
  const logits = g.run(() => parts.discriminate(constant(toy2dGrid(FIELD_RES)), c, 'D')).value;
  const out = new Float32Array(FIELD_RES * FIELD_RES);
  for (let i = 0; i < out.length; i++) out[i] = sigmoidFn(logits.data[i]);
  return new Tensor([FIELD_RES, FIELD_RES], out);
}

/** Real and generated points stacked into one [n_r + n_g, 2] tensor with labels 0 / 1. */
function stackPoints(real: Tensor, fake: Tensor): { points: Tensor; labels: Tensor } {
  const nr = real.shape[0];
  const nf = fake.shape[0];
  const pts = new Float32Array((nr + nf) * 2);
  pts.set(real.data.subarray(0, nr * 2));
  pts.set(fake.data.subarray(0, nf * 2), nr * 2);
  const labels = new Float32Array(nr + nf);
  labels.fill(1, nr);
  return { points: new Tensor([nr + nf, 2], pts), labels: new Tensor([nr + nf], labels) };
}

const gan = defineModel<GanConfig>({
  id: 'gan',
  family: 'gan',
  name: 'Generative adversarial network',
  summary:
    'A generator maps noise to points in the plane and a discriminator learns to tell them from samples of a target distribution; each is trained against the other.',
  signature:
    '\\min_G \\max_D\\; \\mathbb E_{x\\sim p_{\\text{data}}}[\\log D(x)] + \\mathbb E_{z\\sim p_z}[\\log(1 - D(G(z)))]',
  fidelity: 'reduced',
  scaleNote:
    'The objective, the alternating updates (k = 1) and both generator losses are those of Goodfellow et al. (2014). The paper trained on MNIST, TFD and CIFAR-10 with a maxout discriminator; here both networks are two-hidden-layer MLPs on a two-dimensional target so that the generated distribution and the discriminator can be drawn exactly.',
  references: [
    {
      authors:
        'I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio',
      title: 'Generative adversarial nets',
      year: 2014,
      url: 'https://arxiv.org/abs/1406.2661',
    },
    {
      authors: 'L. Metz, B. Poole, D. Pfau, J. Sohl-Dickstein',
      title: 'Unrolled generative adversarial networks',
      year: 2017,
      url: 'https://arxiv.org/abs/1611.02163',
    },
  ],
  limitations: [
    'Two-dimensional targets only; the image-scale version is the DCGAN model.',
    'The discriminator uses leaky ReLU units rather than maxout, and no dropout.',
    'Both networks share one optimiser configuration (each phase keeps its own Adam moments).',
    'Mode coverage counts a mode as covered when at least 5 of the generated points lie within 3σ of its centre; it is a diagnostic for the ring and grid targets, not a standard metric.',
  ],
  guide: {
    what: 'Two small networks are trained against each other on points in the plane. The generator turns random noise into points; the discriminator receives real points from the target (by default eight Gaussian clusters on a ring) together with generated points, and outputs the probability that a point is real. The discriminator is trained to tell the two apart and the generator to fool it; as both improve, the generated points move towards the target clusters.',
    watch:
      'Red generated points begin as a small cloud near the centre and spread towards the blue target clusters; the background is the discriminator’s D(x). In seeds 1–3 all eight clusters are reached by step 500–600 of the default 1,000. By step 1,000 about 95 % of the red points lie on the ring, but only half to two thirds lie within 3σ of a cluster centre; the rest are strung out between clusters. D(x) and D(G(z)) stay between about 0.4 and 0.6 throughout, close to 0.5. Modes covered, shown by “Sample and score”, counts the clusters reached.',
    try: [
      {
        label: 'Minimax generator loss',
        note: 'Generator gradients now scale with D(G(z)) instead of 1 − D(G(z)), so they vanish only where D confidently rejects generated points. Here D(G(z)) stays between about 0.4 and 0.5, so nothing saturates: after 300 steps the red points have reached the ring at least as fast as with the default loss (2–8 of the 8 clusters covered in seeds 1–3, against 0–2).',
        patch: { gLoss: 'minimax' },
        train: 300,
      },
      {
        label: 'Twenty-five modes',
        note: 'A harder target. Watch modes covered: after 500 steps only 6–7 of the 25 Gaussians are covered (seeds 1–3), with the red points bunched on a few neighbouring ones, a partial mode collapse. Longer training covers more (12–23 between steps 1,000 and 2,000), but the covered set keeps shifting.',
        patch: { dataset: 'grid' },
        train: 500,
      },
      {
        label: 'One-dimensional noise',
        note: 'Re-initialises. The generator maps a line into the plane, so its points lie on a single curve. After 500 steps the curve has not yet found the clusters (0–2 of 8 covered, seeds 1–3); trained 500 steps more, it threads through 7 or 8 of them, with about half of its points strung along the ring in between.',
        patch: { zDim: 1 },
        train: 500,
      },
    ],
  },
  configSchema: [
    {
      key: 'dataset',
      label: 'Target distribution',
      kind: 'select',
      options: TOY2D_OPTIONS,
      affects: 'data',
    },
    {
      key: 'zDim',
      label: 'Noise dimension',
      kind: 'int',
      min: 1,
      max: 16,
      affects: 'architecture',
    },
    {
      key: 'width',
      label: 'Hidden width (G and D)',
      kind: 'int',
      min: 8,
      max: 256,
      step: 8,
      affects: 'architecture',
    },
    {
      key: 'gLoss',
      label: 'Generator loss',
      kind: 'select',
      options: G_LOSS_OPTIONS,
      affects: 'forward',
      help: 'The minimax loss has vanishing gradients when D confidently rejects generated points.',
    },
  ],
  defaults: { dataset: 'ring', zDim: 2, width: 64, gLoss: 'nonsaturating' },

  init(c, rng) {
    const b = new ParamBuilder(rng);
    declareLinear(b, 'G.fc1', c.zDim, c.width);
    declareLinear(b, 'G.fc2', c.width, c.width);
    declareLinear(b, 'G.out', c.width, 2);
    declareLinear(b, 'D.fc1', 2, c.width);
    declareLinear(b, 'D.fc2', c.width, c.width);
    declareLinear(b, 'D.out', c.width, 1);
    return b.store;
  },

  datasets: [toy2dDataset<GanConfig>((c) => c.dataset)],

  forward(batch, c, ctx) {
    const r = ganForward(parts, batch, c, ctx);
    const loss = r.dLoss;
    const gLoss = r.gLoss;
    const { points, labels } = stackPoints(r.x.value, r.fake.value);
    const outputs: Record<string, Var> = {
      real: r.x,
      fake: r.fake,
      dReal: r.dReal,
      dFake: r.dFake,
      gLoss,
      points: constant(points, 'points'),
      pointLabels: constant(labels, 'pointLabels'),
    };
    if ((batch.meta as { field?: boolean } | undefined)?.field)
      outputs.field = constant(discriminatorField(c), 'field');
    const metrics: Record<string, number> = {
      dLoss: loss.value.data[0],
      gLoss: gLoss.value.data[0],
      'D(x)': meanProb(r.dReal),
      'D(G(z))': meanProb(r.dFake),
    };
    const cov = modeCoverage(c.dataset, r.fake.value);
    if (cov) metrics.modesCovered = cov.covered;
    return { outputs, loss, metrics };
  },

  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 2e-3, beta1: 0.5 },
    batchSize: 128,
    steps: 1000,
    validateEvery: 50,
    phases: ganPhases(parts),
  },

  procedures: [
    {
      id: 'samples',
      label: 'Sample and score',
      description:
        'Draws 1,000 generated and 1,000 target points, evaluates D on a 64 × 64 grid, and counts the modes the generator covers.',
      run({ config, rng }) {
        const z = new Tensor([1000, config.zDim]);
        for (let i = 0; i < z.size; i++) z.data[i] = rng.normal();
        const fake = parts.generate(constant(z, 'z'), config).value;
        const real = toy2dSample(config.dataset, 1000, rng);
        const { points, labels } = stackPoints(real, fake);
        const cov = modeCoverage(config.dataset, fake);
        return {
          tensors: { points, labels, field: discriminatorField(config) },
          scalars: (cov
            ? { modesCovered: cov.covered, modes: cov.total, fractionNearAMode: cov.quality }
            : {}) as Record<string, number>,
        };
      },
    },
  ],

  views: [
    {
      kind: 'scatter',
      title: 'Target (blue) and generated (red) points over D(x)',
      source: 'out:points',
      colorBy: 'out:pointLabels',
      field: 'out:field',
      extent: TOY2D_EXTENT,
    },
    {
      kind: 'scatter',
      title: '1,000 target and 1,000 generated points',
      source: 'proc:samples:points',
      colorBy: 'proc:samples:labels',
      field: 'proc:samples:field',
      extent: TOY2D_EXTENT,
    },
  ],

  math: {
    G: {
      tex: 'G(z) = W_3\\, \\mathrm{relu}(W_2\\, \\mathrm{relu}(W_1 z + b_1) + b_2) + b_3,\\quad z \\sim \\mathcal N(0, I)',
      text: 'The generator pushes the noise distribution forward to a distribution in the plane. Its samples are the red points.',
    },
    D_real: {
      tex: 'D(x) = \\sigma\\big(f_D(x)\\big)',
      text: 'The discriminator outputs the probability that x came from the target. The background shows D(x) on a grid.',
    },
    D_fake: {
      tex: 'D(G(z)) = \\sigma\\big(f_D(G(z))\\big)',
      text: 'The same discriminator, applied to generated points.',
    },
    loss_D: {
      tex: 'L_D = -\\tfrac1m \\textstyle\\sum_i \\big[\\log D(x_i) + \\log(1 - D(G(z_i)))\\big]',
      text: 'The discriminator phase minimises this with G frozen.',
    },
    loss_G: {
      tex: 'L_G = -\\tfrac1m \\textstyle\\sum_i \\log D(G(z_i)) \\;\\text{or}\\; \\tfrac1m \\sum_i \\log(1 - D(G(z_i)))',
      text: 'The generator phase minimises this with D frozen. The non-saturating form has the same fixed point but stronger gradients early in training (Goodfellow et al., section 3).',
    },
  },
});

export default gan;
