import {
  ParamBuilder,
  Tensor,
  concat,
  crossEntropy,
  declareLinear,
  defaultOptimizer,
  dense,
  scope,
  softmax,
  softplus,
  silu,
  tanh,
  type Var,
} from '../../engine';
import { defineModel } from '../types';
import {
  PLANAR_EXTENT,
  PLANAR_OPTIONS,
  planarDataset,
  planarPoints,
  type PlanarKind,
} from '../data/planar';
import { accuracy, constant, input } from '../util';
import { fieldProcedure, logitMargin } from '../feedforward/planar-common';
import { METHOD_TEX, ODE_METHODS, STAGES, odeSolve, odeStep, type OdeMethod } from './solvers';

type NodeConfig = {
  dataset: PlanarKind;
  noise: number;
  augment: number;
  width: number;
  timeInput: boolean;
  activation: 'tanh' | 'softplus' | 'silu';
  method: OdeMethod;
  steps: number;
  time: number;
};

const ACT = { tanh, softplus, silu };

/** f_θ(h, t): two hidden layers; t enters as an extra input column when enabled. */
function vectorField(c: NodeConfig) {
  const act = ACT[c.activation];
  return (h: Var, t: number): Var =>
    scope('f', () => {
      const N = h.shape[0];
      const z = c.timeInput ? concat([h, constant(Tensor.full([N, 1], t), 't')], 1) : h;
      const a1 = act(dense(z, 'f.fc1'));
      const a2 = act(dense(a1, 'f.fc2'));
      return dense(a2, 'f.out');
    });
}

function augmentInput(x: Var, c: NodeConfig): Var {
  if (c.augment === 0) return x;
  const N = x.shape[0];
  return scope('augment', () => concat([x, constant(Tensor.zeros([N, c.augment]), '0')], 1));
}

function run(x: Var, c: NodeConfig) {
  const h0 = augmentInput(x, c);
  const sol = scope('ode', () => odeSolve(vectorField(c), h0, 0, c.time, c.steps, c.method));
  const logits = scope('classifier', () => dense(sol.h, 'classifier'));
  return { h0, sol, logits };
}

const GRID = 21;

const neuralOde = defineModel<NodeConfig>({
  id: 'neural-ode',
  family: 'neural-dynamics',
  name: 'Neural ODE',
  summary:
    'The hidden state is the solution of an ordinary differential equation whose right-hand side is a neural network; points are classified after flowing along it.',
  signature:
    '\\frac{dh}{dt} = f_\\theta(h(t), t),\\quad h(0) = x,\\quad \\hat y = \\operatorname{softmax}(W h(T))',
  fidelity: 'reduced',
  scaleNote:
    'The continuous-depth model and its use as a classifier follow Chen et al. (2018). Reduced: fixed-step Euler, midpoint or RK4 instead of the adaptive Dormand–Prince solver, gradients by backpropagation through the solver steps instead of the adjoint ODE, and two-dimensional data (optionally augmented with zero dimensions as in Dupont et al., 2019) instead of MNIST.',
  references: [
    {
      authors: 'R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud',
      title: 'Neural ordinary differential equations',
      year: 2018,
      url: 'https://arxiv.org/abs/1806.07366',
    },
    {
      authors: 'E. Dupont, A. Doucet, Y. W. Teh',
      title: 'Augmented neural ODEs',
      year: 2019,
      url: 'https://arxiv.org/abs/1904.01681',
    },
  ],
  limitations: [
    'Discretise-then-optimise: gradients are exact derivatives of the discretised solver, obtained by backpropagating through every stage. Memory grows linearly with the number of steps. The continuous adjoint method (constant memory) is not implemented.',
    'Fixed step size; there is no error control, so a coarse step can give a flow that differs from the true ODE solution.',
    'With zero augmentation the exact flow is a homeomorphism of the plane, so classes that enclose each other (concentric circles) cannot be separated by the final linear classifier; a coarse fixed-step solver is not injective in general, so the discretised map may separate them where the exact flow cannot.',
    'With augmentation the state has more than two dimensions. The vector field is then drawn on the plane where the extra coordinates are zero and trajectories are projected onto the first two coordinates.',
  ],
  guide: {
    what: 'The input is a point in the plane from a two-class toy dataset (two moons by default). Instead of passing it through a stack of layers, the model lets it move: a small network f_θ gives the point’s velocity at each moment, and a numerical solver (RK4 with 8 steps by default) follows that velocity from time 0 to 1, where a linear classifier reads the final position. Training adjusts f_θ so the flow carries the two classes to opposite sides of a line.',
    watch:
      'There is no checkpoint: training starts from random weights. Held-out accuracy passes 0.9 within 30–40 steps and reaches 1.0 by step 300 (seeds 1–3), though it can dip below 0.9 on the way. In the flow view the arrows reorganise and the 40 coloured trajectories bend until the two classes end on opposite sides; the decision field then shows a curved boundary across the input plane.',
    try: [
      {
        label: 'Concentric circles',
        note: 'An exact planar flow cannot pull the inner disc out of the ring, yet held-out accuracy still reaches 0.75–0.99 after 200 steps (seeds 1–3), less steadily than on moons. The flow stretches the plane so that most held-out points end on the correct side of a line, and an 8-step solver need not follow the exact flow.',
        patch: { dataset: 'circles' },
        train: 200,
      },
      {
        label: 'Circles, one extra dimension',
        note: 'With a third coordinate the inner class can lift out of the plane: accuracy reaches 0.94–1.0 after 200 steps (seeds 1–3), against 0.75–0.99 without it. The portrait shows only the projection.',
        patch: { dataset: 'circles', augment: 1 },
        train: 200,
      },
      {
        label: 'Single Euler step',
        note: 'One Euler step is a residual layer, h + f(h): moons are still learned (accuracy 0.98–1.0 after 200 steps, seeds 1–3), but every trajectory is one straight segment.',
        patch: { method: 'euler', steps: 1 },
        train: 200,
      },
    ],
  },
  configSchema: [
    { key: 'dataset', label: 'Dataset', kind: 'select', options: PLANAR_OPTIONS, affects: 'data' },
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
      key: 'augment',
      label: 'Augmented dimensions',
      kind: 'int',
      min: 0,
      max: 3,
      affects: 'architecture',
      help: 'Zeros appended to x before the flow (augmented neural ODE).',
    },
    { key: 'width', label: 'Width of f', kind: 'int', min: 4, max: 64, affects: 'architecture' },
    { key: 'timeInput', label: 'f depends on t', kind: 'bool', affects: 'architecture' },
    {
      key: 'activation',
      label: 'Activation in f',
      kind: 'select',
      affects: 'forward',
      options: [
        { value: 'tanh', label: 'tanh' },
        { value: 'softplus', label: 'softplus' },
        { value: 'silu', label: 'SiLU' },
      ],
    },
    { key: 'method', label: 'Solver', kind: 'select', options: ODE_METHODS, affects: 'forward' },
    { key: 'steps', label: 'Solver steps', kind: 'int', min: 1, max: 40, affects: 'forward' },
    {
      key: 'time',
      label: 'Integration time T',
      kind: 'float',
      min: 0.25,
      max: 3,
      step: 0.25,
      affects: 'forward',
    },
  ],
  defaults: {
    dataset: 'moons',
    noise: 0.08,
    augment: 0,
    width: 24,
    timeInput: true,
    activation: 'tanh',
    method: 'rk4',
    steps: 8,
    time: 1,
  },
  init(c, rng) {
    const b = new ParamBuilder(rng);
    const D = 2 + c.augment;
    declareLinear(b, 'f.fc1', D + (c.timeInput ? 1 : 0), c.width);
    declareLinear(b, 'f.fc2', c.width, c.width);
    declareLinear(b, 'f.out', c.width, D);
    declareLinear(b, 'classifier', D, 2);
    return b.store;
  },
  datasets: [planarDataset<NodeConfig>((c) => c.dataset, { noise: (c) => c.noise })],
  forward(batch, c) {
    const x = input('x', batch.inputs.x);
    const { h0, sol, logits } = run(x, c);
    const probs = softmax(logits, -1);
    const loss = batch.labels
      ? scope('loss', () => crossEntropy(logits, batch.labels!))
      : undefined;
    return {
      outputs: { h0, hT: sol.h, logits, probs },
      loss,
      metrics: {
        ...(batch.labels ? { accuracy: accuracy(probs.value, batch.labels) } : {}),
        NFE: sol.nfe,
      },
    };
  },
  training: {
    support: 'full',
    note: 'Gradients are obtained by backpropagating through the solver steps (discretise-then-optimise), not by the adjoint method.',
    optimizer: { ...defaultOptimizer, kind: 'adam', lr: 0.01 },
    batchSize: 32,
    steps: 300,
    validateEvery: 25,
  },
  procedures: [
    fieldProcedure<NodeConfig>(
      (b, c) => ({ outputs: { logits: run(input('x', b.inputs.x, false), c).logits } }),
      (c) => c.dataset,
      (r) => logitMargin(r.outputs.logits.value),
      (c) => c.noise,
    ),
    {
      id: 'flow',
      label: 'Flow of the training points',
      description:
        'Integrates the 400 training points with the configured solver, records the solver states of 40 of them, and evaluates f_θ(h, t) on a 21 × 21 grid at the chosen time.',
      args: [
        {
          key: 't',
          label: 'Field at time t',
          kind: 'float',
          min: 0,
          max: 3,
          step: 0.05,
          affects: 'forward',
        },
      ],
      defaults: { t: 0 },
      run(ctx) {
        const c = ctx.config;
        const D = 2 + c.augment;
        const pts = planarPoints(c.dataset, 400, 11, c.noise);
        const x = constant(new Tensor([400, 2], pts.x), 'x');
        const h0 = augmentInput(x, c);
        const f = vectorField(c);
        const states: Tensor[] = [h0.value];
        let h = h0;
        const dt = c.time / c.steps;
        let nfe = 0;
        for (let k = 0; k < c.steps; k++) {
          if (ctx.shouldStop()) break;
          h = odeStep(f, h, k * dt, dt, c.method);
          nfe += STAGES[c.method];
          states.push(h.value);
          ctx.progress((k + 1) / c.steps);
        }
        const K = 40;
        const T1 = states.length;
        const traj = new Float32Array(K * T1 * 2);
        for (let i = 0; i < K; i++)
          for (let k = 0; k < T1; k++) {
            const row = Math.floor((i * 400) / K);
            traj[(i * T1 + k) * 2] = states[k].data[row * D];
            traj[(i * T1 + k) * 2 + 1] = states[k].data[row * D + 1];
          }
        const trajLabels = Float32Array.from(
          { length: K },
          (_, i) => pts.y[Math.floor((i * 400) / K)],
        );
        const fin = new Float32Array(400 * 2);
        for (let i = 0; i < 400; i++) {
          fin[i * 2] = h.value.data[i * D];
          fin[i * 2 + 1] = h.value.data[i * D + 1];
        }
        // Vector field on the plane (extra coordinates zero), row 0 at the bottom.
        const [x0, x1, y0, y1] = PLANAR_EXTENT;
        const g = new Float32Array(GRID * GRID * D);
        for (let i = 0; i < GRID; i++)
          for (let j = 0; j < GRID; j++) {
            g[(i * GRID + j) * D] = x0 + ((j + 0.5) / GRID) * (x1 - x0);
            g[(i * GRID + j) * D + 1] = y0 + ((i + 0.5) / GRID) * (y1 - y0);
          }
        const fv = f(constant(new Tensor([GRID * GRID, D], g), 'grid'), Number(ctx.args.t)).value;
        const field = new Float32Array(GRID * GRID * 2);
        for (let i = 0; i < GRID * GRID; i++) {
          field[i * 2] = fv.data[i * D];
          field[i * 2 + 1] = fv.data[i * D + 1];
        }
        return {
          tensors: {
            field: new Tensor([GRID, GRID, 2], field),
            trajectories: new Tensor([K, T1, 2], traj),
            trajectoryLabels: new Tensor([K], trajLabels),
            final: new Tensor([400, 2], fin),
            labels: new Tensor([400], Float32Array.from(pts.y)),
          },
          scalars: { NFE: nfe, Δt: dt },
        };
      },
    },
  ],
  views: [
    {
      kind: 'scatter',
      title: 'Decision field over the input plane and training points',
      source: 'proc:field:points',
      colorBy: 'proc:field:labels',
      field: 'proc:field:field',
      extent: PLANAR_EXTENT,
    },
    {
      kind: 'phase',
      title: 'Vector field f_θ(h, t) and solver trajectories of 40 training points (colour: class)',
      field: 'proc:flow:field',
      trajectories: 'proc:flow:trajectories',
      colorBy: 'proc:flow:trajectoryLabels',
      extent: PLANAR_EXTENT,
      note: 'Arrows: f_θ(h, t) at the chosen time t on a 21 × 21 grid, length ∝ ‖f‖ (normalised to the largest); with augmentation, on the plane where the extra coordinates are zero. Curves: the solver states h(t_k) of 40 training points, from x (start) to h(T), coloured by class. The field depends on t, so a path need not follow the arrows drawn for one t.',
    },
    {
      kind: 'scatter',
      title: 'Training points after the flow, h(T) (first two coordinates)',
      source: 'proc:flow:final',
      colorBy: 'proc:flow:labels',
    },
    {
      kind: 'probs',
      title: 'Class probabilities (first example)',
      source: 'out:probs',
      classNames: ['class 0', 'class 1'],
    },
  ],
  math: {
    augment: {
      tex: 'h(0) = [x;\\ 0_{a}]',
      text: 'Augmented neural ODE: the state carries a extra coordinates that start at zero.',
    },
    ode: {
      tex: `h(T) = h(0) + \\int_0^T f_\\theta(h(t), t)\\,dt,\\qquad \\text{RK4: } ${METHOD_TEX.rk4}`,
      text: 'Solved with a fixed number of equal steps. Euler uses 1, midpoint 2 and RK4 4 evaluations of f per step (NFE = steps × stages).',
    },
    f: {
      tex: 'f_\\theta(h, t) = W_3\\,\\sigma(W_2\\,\\sigma(W_1 [h; t] + b_1) + b_2) + b_3',
      text: 'With “f depends on t” off, the input is h alone and the ODE is autonomous.',
    },
    classifier: { tex: 'z = W h(T) + b,\\quad p = \\operatorname{softmax}(z)' },
    loss: {
      tex: 'L = -\\tfrac1N\\textstyle\\sum_n \\log p_{n,y_n},\\qquad \\frac{\\partial L}{\\partial \\theta} \\text{ by reverse mode through every solver stage}',
    },
  },
});

export default neuralOde;
