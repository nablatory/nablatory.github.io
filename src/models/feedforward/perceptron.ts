import {
  ParamBuilder,
  defaultOptimizer,
  dense,
  mean,
  mul,
  neg,
  relu,
  scope,
  type Var,
} from '../../engine';
import { defineModel } from '../types';
import { planarDataset, type PlanarKind } from '../data/planar';
import { input } from '../util';
import { fieldProcedure, fieldViews, signTargets } from './planar-common';

type PerceptronConfig = {
  dataset: PlanarKind;
  noise: number;
  init: 'small' | 'unit';
};

const score = (x: Var) => scope('unit', () => dense(x, 'unit'));

const perceptron = defineModel<PerceptronConfig>({
  id: 'perceptron',
  family: 'feedforward',
  name: 'Perceptron',
  summary:
    'A single linear threshold unit. It separates the plane with one line, and cannot represent XOR.',
  signature: '\\hat y = \\operatorname{sign}(w^{\\top} x + b)',
  fidelity: 'equations',
  references: [
    {
      authors: 'F. Rosenblatt',
      title:
        'The perceptron: a probabilistic model for information storage and organization in the brain',
      year: 1958,
      url: 'https://doi.org/10.1037/h0042519',
    },
    { authors: 'M. Minsky, S. Papert', title: 'Perceptrons', year: 1969 },
  ],
  limitations: [
    'The learning rule is expressed as gradient descent on the perceptron criterion; with batch size 1 and SGD at learning rate 1 this is exactly Rosenblatt’s update w ← w + t·x on misclassified points.',
    'At t·s = 0 the subgradient 0 is used, so a perceptron initialised exactly at zero never moves. Initialisation is therefore random.',
    'Rosenblatt’s 1958 perceptron had a fixed, randomly wired layer of association units between the sensory inputs and the response unit; this is the single response unit with learned weights on its inputs, the simple perceptron analysed by Minsky and Papert (1969).',
  ],
  guide: {
    what: 'A perceptron takes a point in the plane (two numbers) and answers one of two classes: it multiplies each coordinate by a weight, adds a bias, and reports class 1 if the total is positive. Training adjusts the weights and bias only when a point is misclassified, moving the line towards that point. The boundary is always one straight line, so the perceptron can only separate data that a line can separate.',
    watch:
      'Training feeds one point per step. On the separable blobs the line settles within a few dozen steps and the loss falls to almost zero, because correctly classified points contribute nothing to it. Watch the decision field (drawn by the “Decision field” procedure) and the accuracy: once every point is on the right side, nothing changes any more.',
    try: [
      {
        label: 'XOR blobs',
        note: 'No single line separates XOR: the line keeps jumping and accuracy stays between about 45 and 65 %.',
        patch: { dataset: 'xor' },
        train: 200,
      },
      {
        label: 'Two moons',
        note: 'The moons overlap along any line, so accuracy bounces between roughly 50 and 90 % and never settles.',
        patch: { dataset: 'moons' },
        train: 200,
      },
    ],
  },
  configSchema: [
    {
      key: 'dataset',
      label: 'Dataset',
      kind: 'select',
      affects: 'data',
      options: [
        { value: 'blobs', label: 'Separable blobs' },
        { value: 'moons', label: 'Two moons' },
        { value: 'xor', label: 'XOR blobs' },
        { value: 'circles', label: 'Concentric circles' },
      ],
    },
    {
      key: 'noise',
      label: 'Point noise σ',
      kind: 'float',
      min: 0,
      max: 0.4,
      step: 0.02,
      affects: 'data',
    },
    {
      key: 'init',
      label: 'Initialisation',
      kind: 'select',
      affects: 'architecture',
      options: [
        { value: 'small', label: 'N(0, 0.01²)' },
        { value: 'unit', label: 'N(0, 1)' },
      ],
    },
  ],
  defaults: { dataset: 'blobs', noise: 0.08, init: 'small' },
  init(c, rng) {
    const b = new ParamBuilder(rng);
    const std = c.init === 'small' ? 0.01 : 1;
    b.add('unit.weight', [1, 2], { kind: 'normal', std });
    b.add('unit.bias', [1], { kind: 'normal', std });
    return b.store;
  },
  datasets: [planarDataset<PerceptronConfig>((c) => c.dataset, { noise: (c) => c.noise })],
  forward(batch) {
    const x = input('x', batch.inputs.x);
    const s = score(x);
    if (!batch.labels) return { outputs: { score: s } };
    const t = signTargets(batch);
    // Perceptron criterion: L = mean(max(0, −t·s)).
    const loss = scope('loss', () => mean(relu(neg(mul(t, s)))));
    let k = 0;
    for (let i = 0; i < batch.labels.length; i++)
      if ((s.value.data[i] > 0 ? 1 : 0) === batch.labels[i]) k++;
    return { outputs: { score: s }, loss, metrics: { accuracy: k / batch.labels.length } };
  },
  training: {
    support: 'full',
    optimizer: { ...defaultOptimizer, kind: 'sgd', lr: 1, momentum: 0, weightDecay: 0 },
    batchSize: 1,
    steps: 200,
    validateEvery: 20,
  },
  procedures: [
    fieldProcedure<PerceptronConfig>(
      (b) => ({ outputs: { score: score(input('x', b.inputs.x, false)) } }),
      (c) => c.dataset,
      (r) => r.outputs.score.value,
      (c) => c.noise,
    ),
  ],
  views: [...fieldViews, { kind: 'vector', title: 'Score s = wᵀx + b', source: 'out:score' }],
  math: {
    unit: {
      tex: 's = w^{\\top}x + b,\\qquad \\hat y = \\mathbb{1}[s > 0]',
      text: 'The decision boundary is the line w₁x₁ + w₂x₂ + b = 0; the field shows s.',
    },
    loss: {
      tex: 'L = \\tfrac{1}{N}\\textstyle\\sum_n \\max(0, -t_n s_n),\\quad t_n = 2y_n - 1',
      text: 'Its gradient is −t·x for misclassified points and 0 otherwise, which is the perceptron learning rule.',
    },
  },
});

export default perceptron;
